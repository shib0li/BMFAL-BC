import numpy as np
import torch
import torch.nn as nn
import torch.distributions as distributions
from torch.optim import LBFGS
import time

from models.mfnet import DeepMFNet

import time
import datetime
from infras.misc import *
from infras.randutils import *

import dataset_active as dataset

class SubmodBatchDMFAL(DeepMFNet):
    
    def __init__(self, opt, synD):
        super().__init__(opt, synD)

        self.batch_size = opt.batch_size
        self.costs = np.array(opt.penalty)
        
        self.concat_weights_mean = []
        self.concat_weights_std = []
        
        for i in range(self.M):
            concat_mu = torch.cat((self.nns_list[i].W_mu, self.nns_list[i].b_mu), dim=0)   # concatenate mean
            concat_std = torch.cat((self.nns_list[i].W_std, self.nns_list[i].b_std), dim=0) # concatenate std
            self.concat_weights_mean.append(concat_mu)
            self.concat_weights_std.append(concat_std)
        #
        
        self.V_param_list = []
        self.param_dims = []
        for m in range(self.M):
            V_param = self.eval_params_var(m)
            self.V_param_list.append(V_param)
            self.param_dims.append(V_param.shape[0])


    def init_query_points(self, m, Nq=1):
        lb, ub = self.data.get_N_bounds(m)
        scale = (ub-lb).reshape([1,-1])
        uni_noise = np.random.uniform(size=[Nq, self.input_dims[m]])
        
        np_Xq_init = uni_noise*scale + lb
        
        Xq = torch.tensor(np_Xq_init, device=self.device, dtype=self.torch_type, requires_grad=True)
        
        return Xq

    def eval_params_var(self, m):
        # flatten the variance
        std_list = self.concat_weights_std[:m+1]
        flat_std_list = []
        for std in std_list:
            flat_std = std.reshape([-1])
            #print(flat_var.shape)
            flat_std_list.append(flat_std)
        #
        stack_flat_std = torch.cat(flat_std_list, dim=0)
        
        V_param = torch.diag(torch.square(stack_flat_std))
        
        return V_param
    
    def single_nonlinear_base(self, X, m, weights_list):
        # first fidelity
        W = weights_list[0][0:-1, :]
        b = weights_list[0][-1, :].reshape([1,-1])

        base_m = self.nns_list[0].forward_base_by_sample(X, W, b)
        
        # propagate to the other fidelity levels
        for i in range(1,m+1):
            W = weights_list[i][0:-1, :]
            b = weights_list[i][-1, :].reshape([1,-1])

            X_concat = torch.cat((base_m, X), dim=1)
            base_m = self.nns_list[i].forward_base_by_sample(X_concat, W, b)
        #
        return base_m
    
    def eval_output_jacob(self, X, m):
        
        weights_list = self.concat_weights_mean[:m+1]
        
        if m == 0:
            obj_func = lambda Wcat0 : self.single_nonlinear_base(X, m, [Wcat0])
        elif m == 1:
            obj_func = lambda Wcat0, Wcat1 : self.single_nonlinear_base(X, m, [Wcat0, Wcat1])
        elif m == 2:
            obj_func = lambda Wcat0, Wcat1, Wcat2 : self.single_nonlinear_base(X, m, [Wcat0, Wcat1, Wcat2])
        #
        
        jacobians = torch.autograd.functional.jacobian(obj_func, tuple(weights_list), strict=True, create_graph=True)
        
        # stack the jacobians
        stack_jacobian_list = []
        for Jm in list(jacobians):
            N = Jm.shape[0]
            K = Jm.shape[1]
            mat_flat_Jm = Jm.reshape([N*K, -1])
            stack_jacobian_list.append(mat_flat_Jm)
        #
        J = torch.cat(stack_jacobian_list, dim=1)
        
        return J

    def eval_batch_base_variance_jacobians(self, X_batch, m_batch, J_batch):

        hf = np.max(np.array(m_batch))
        
        V_param = self.V_param_list[hf]
        target_dim = self.param_dims[hf]
        
        pad_J_batch = []
        
        for J in J_batch:
            dim = J.shape[1]
            if dim < target_dim:
                padding = nn.ZeroPad2d((0, target_dim-dim, 0, 0))
                pad_J = padding(J)
                pad_J_batch.append(pad_J)
            else:
                pad_J_batch.append(J)
            #
        
        #
        
        stack_J = torch.cat(pad_J_batch, dim=0)
        V_base = stack_J @ V_param @ stack_J.T
        
        return V_base


    def eval_batch_output_entropy(self, X_batch, m_batch, J_batch):
        
        V_batch_base = self.eval_batch_base_variance_jacobians(X_batch, m_batch, J_batch)
        
        Am_list = []
        Am_hat_list = []
        log_tau_m_list = []
        K_list = []
        
        D_list = []
        D_log_tau_list = []
        
        for m in m_batch:
            Am_list.append(self.nns_list[m].A)
            log_tau_m_list.append(self.log_tau_list[m])
            Am_hat_list.append(torch.exp(self.log_tau_list[m])*self.nns_list[m].A)
            K_list.append(self.base_dims[m])
            
            D_list.append(self.output_dims[m])
            D_log_tau_list.append(self.output_dims[m]*self.log_tau_list[m])
        #
        
        A = torch.block_diag(*Am_list)
        A_hat = torch.block_diag(*Am_hat_list)
        
        A_hat_A_tr = torch.matmul(A_hat, A.T)
        
        I_KN = torch.eye(sum(K_list), device=self.device, dtype=self.torch_type)
        
        output_var = torch.matmul(A_hat_A_tr, V_batch_base) + I_KN
        
        log_det = torch.logdet(output_var)
        
        entropy = sum(D_list)*np.log(2*np.pi*np.e) - 0.5*sum(D_log_tau_list) + 0.5*log_det
        
        return entropy
    
    def eval_batch_mutual_info(self, X_batch, m_batch, J_m_batch, J_M_batch, J_joint_batch):
        
        if len(X_batch) == 0:
            H_batch_m = 0.0
        else:
            H_batch_m = self.eval_batch_output_entropy(X_batch, m_batch, J_m_batch)
        
        M_batch = [self.M-1]*len(self.X_S_batch)
        
        H_batch_M = self.eval_batch_output_entropy(self.X_S_batch, M_batch, J_M_batch)
        
        H_batch_mM = self.eval_batch_output_entropy(X_batch+self.X_S_batch, m_batch+M_batch, J_joint_batch)
        
        return H_batch_m + H_batch_M - H_batch_mM

    def opt_submod_query(self, X_batch, m_batch, J_m_batch, J_M_batch, m):

        Xq = self.init_query_points(m)

        np_lb, np_ub = self.data.get_N_bounds(m)
        bounds = torch.tensor(np.vstack((np_lb, np_ub)), device=self.device, dtype=self.torch_type)

        lbfgs = LBFGS([Xq], lr=self.opt_lr, max_iter=20, max_eval=None, 
            tolerance_grad=1e-8, tolerance_change=1e-12, history_size=100)

        def closure():
            lbfgs.zero_grad()  

            Jm = self.eval_output_jacob(Xq, m)

            curr_J = J_m_batch + [Jm]
            curr_J_hf = J_M_batch
            curr_J_joint = curr_J+J_M_batch

            mutual_info_before = self.eval_batch_mutual_info(
                X_batch, m_batch, J_m_batch, J_M_batch, J_m_batch+J_M_batch)

            mutual_info_after = self.eval_batch_mutual_info(
                X_batch+[Xq], m_batch+[m], curr_J, curr_J_hf, curr_J_joint)

            gain_mi = mutual_info_after - mutual_info_before

            loss = -gain_mi
            
            loss.backward(retain_graph=True)

            with torch.no_grad():
                for j, (lb, ub) in enumerate(zip(*bounds)):
                    Xq.data[..., j].clamp_(lb, ub) # need to do this on the data not X itself
                #
            #
            return loss

        lbfgs.step(closure)

        Jm = self.eval_output_jacob(Xq, m)

        curr_J = J_m_batch + [Jm]
        curr_J_hf = J_M_batch
        curr_J_joint = curr_J+J_M_batch

        mutual_info_before = self.eval_batch_mutual_info(
            X_batch, m_batch, J_m_batch, J_M_batch, J_m_batch+J_M_batch)

        mutual_info_after = self.eval_batch_mutual_info(
            X_batch+[Xq], m_batch+[m], curr_J, curr_J_hf, curr_J_joint)

        gain_mi = mutual_info_after - mutual_info_before

        self.logger.write("  - info AFTER " + str(gain_mi.data.cpu().numpy()) + '\n')
        self.logger.write("  - Xq   AFTER " + str(Xq.data.cpu().numpy()) + '\n')

        return gain_mi, Xq
    
    def submod_eval_next(self, X_batch, m_batch, J_m_batch, J_M_batch):

        fidelity_info = []
        fidelity_query = []
        fidelity_costs = []
        
        prev_batch_costs = [self.costs[b] for b in m_batch]

        for m in range(self.M):
            info, xq = self.opt_submod_query(
                X_batch, m_batch, J_m_batch, J_M_batch, m)
            
            fidelity_info.append(info.data.cpu().numpy())
            fidelity_query.append(xq)
            
            fidelity_costs.append(self.costs[m])
        #

        reg_info = np.array(fidelity_info) / np.array(fidelity_costs)

        argm = np.argmax(reg_info)
        argx = fidelity_query[argm]
        
 
        self.logger.write('argm='+str(argm)+'\n')
        self.logger.write('argx='+str(argx.data.cpu().numpy())+'\n')
        self.logger.flush()
        
        Jm = self.eval_output_jacob(argx, argm)
          
        J_m_batch.append(Jm)
        X_batch.append(argx)
        m_batch.append(argm)
        
        return X_batch, m_batch, J_m_batch
    
    
    def submod_batch_query(self, Ns=20):
        
        lb, ub = self.data.get_N_bounds(self.M-1)
        np_X_hf = generate_with_bounds(Ns, lb, ub, method='lhs')
        batch_X_hf = torch.tensor(
            np_X_hf, device=self.device, dtype=self.torch_type, requires_grad=False)
        
        self.X_S_batch = []
        for s in range(Ns):
            self.X_S_batch.append(batch_X_hf[s,:].reshape([-1,1]))
        #
        
        J_m_batch = []
        J_M_batch = []

        for s in range(Ns):
            Js = self.eval_output_jacob(batch_X_hf[s,:].reshape([1,-1]), self.M-1)
            J_M_batch.append(Js)
        #
        
        X_batch = []
        m_batch = []
        
        B = self.batch_size
        query_costs = 0
        
        while query_costs < B:
            
            X_batch, m_batch, J_m_batch = self.submod_eval_next(
                X_batch, m_batch, J_m_batch, J_M_batch)
            
            current_costs = np.array([self.costs[m] for m in m_batch]).sum()
            query_costs = current_costs
        #

        np_X_batch = []
        for xq in X_batch:
            np_X_batch.append(xq.data.cpu().numpy())

        return np_X_batch, m_batch
    

    
    
    