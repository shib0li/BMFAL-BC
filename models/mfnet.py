import numpy as np
import torch
from torch.optim import Adam
import torch.distributions as distributions
from scipy.stats import truncnorm
import time

import dataset_active as dataset

class AdaptiveBaseNet:
    
    def __init__(self, layers, activation, device, torch_type):
        
        self.device = device
        self.torch_type = torch_type
        
        self.layers = layers
        self.num_layers = len(self.layers)
        self.activation = {'tanh': torch.nn.Tanh(), 'relu': torch.nn.ReLU(), 'sigmoid': torch.nn.Sigmoid()}[activation]
        
        self.input_dim = self.layers[0]
        self.output_dim = self.layers[-1]
        self.latent_dim = self.layers[-3]
        self.base_dim = self.layers[-2]
        
        self.latent_weights = []
        self.latent_biases = []
        
        for l in range(self.num_layers-3):
            W = self._xavier_init(self.layers[l], self.layers[l+1])
            b = torch.zeros([1,self.layers[l+1]], dtype=self.torch_type, device=self.device, requires_grad=True)
            self.latent_weights.append(W)
            self.latent_biases.append(b)
        
        self.W_mu = self._xavier_init(self.latent_dim, self.base_dim)
        self.b_mu = torch.zeros([1,self.base_dim], dtype=self.torch_type, device=self.device, requires_grad=True)
        
        self.W_rho = torch.zeros([self.latent_dim, self.base_dim], dtype=self.torch_type, device=self.device, requires_grad=True)
        self.W_std = torch.log(1 + torch.exp(self.W_rho))
        
        self.b_rho = torch.zeros([1, self.base_dim], dtype=self.torch_type, device=self.device, requires_grad=True)
        self.b_std = torch.log(1 + torch.exp(self.b_rho))
        
        self.A = self._xavier_init(self.base_dim, self.output_dim)
        self.A_b = torch.zeros([1,self.output_dim], dtype=self.torch_type, device=self.device, requires_grad=True)
        
        self.normal = distributions.normal.Normal(
            torch.tensor([0.0], dtype=self.torch_type, device=self.device), 
            torch.tensor([1.0], dtype=self.torch_type, device=self.device)
        )


        self.kld = self._eval_kld()
        self.reg = self._eval_reg()
        
    
    def _sample_from_posterior(self,):
        epsi_W = torch.squeeze(self.normal.sample(self.W_mu.shape))
        W_sample = self.W_mu + self.W_std*epsi_W
        
        epsi_b = torch.squeeze(self.normal.sample(self.b_mu.shape), dim=2)
        b_sample = self.b_mu + self.b_std*epsi_b
        
        return W_sample, b_sample
    
    def forward(self, X, sample=False):
        
        H = X
        for l in range(self.num_layers-3):
            W = self.latent_weights[l]
            b = self.latent_biases[l]
            H = torch.add(torch.matmul(H, W), b)
            
            # scale before the nonlinear-op
            in_d = self.layers[l]
            H = H/np.sqrt(in_d+1)
            H = self.activation(H) 
        
        # project the latent base to base
        if sample:
            W_sample, b_sample = self._sample_from_posterior()
            H = torch.add(torch.matmul(H, W_sample), b_sample)
        else:
            H = torch.add(torch.matmul(H, self.W_mu), self.b_mu)
        #
        
        base = H/np.sqrt(self.latent_dim+1)
        
        Y = torch.add(torch.matmul(base, self.A), self.A_b)
        Y = Y/np.sqrt(self.base_dim+1)

        return Y, base
    
    def forward_base_by_sample(self, X, W_sample, b_sample):
        
        H = X
        for l in range(self.num_layers-3):
            W = self.latent_weights[l]
            b = self.latent_biases[l]
            H = torch.add(torch.matmul(H, W), b)
            
            # scale before the nonlinear-op
            in_d = self.layers[l]
            H = H/np.sqrt(in_d+1)
            H = self.activation(H) 
        #
        
        H = torch.add(torch.matmul(H, W_sample), b_sample)
        
        base = H/np.sqrt(self.latent_dim+1)

        return base
        
        
    def _eval_reg(self,):
        L2_norm_list = []
        for w in self.latent_weights:
            L2_norm_list.append(torch.sum(torch.square(w)))
        #
        for b in self.latent_biases:
            L2_norm_list.append(torch.sum(torch.square(b)))
        #
        L2_norm_list.append(torch.sum(torch.square(self.A)))
        L2_norm_list.append(torch.sum(torch.square(self.A_b)))
        
        return sum(L2_norm_list)
        
    
    def _eval_kld(self,):
        kld_W = torch.sum(-torch.log(self.W_std) + 0.5*(torch.square(self.W_std) + torch.square(self.W_mu)) - 0.5)
        kld_b = torch.sum(-torch.log(self.b_std) + 0.5*(torch.square(self.b_std) + torch.square(self.b_mu)) - 0.5)
        
        return kld_W+kld_b
    
    def _xavier_init(self, in_dim, out_dim):
        xavier_stddev = np.sqrt(2.0/(in_dim + out_dim))
        W = torch.normal(size=(in_dim, out_dim), mean=0.0, std=xavier_stddev, requires_grad=True, device=self.device, dtype=self.torch_type)
        return W
    
    def _msra_init(self, in_dim, out_dim):
        xavier_stddev = np.sqrt(2.0/(in_dim))
        W = torch.normal(size=(in_dim, out_dim), mean=0.0, std=xavier_stddev, requires_grad=True, device=self.device, dtype=self.torch_type)
        return W
    
    def parameters(self,):
        params = {}
        params['latent_weights'] = self.latent_weights
        params['latent_biases'] = self.latent_biases
        params['W_mu'] = self.W_mu
        params['W_rho'] = self.W_rho
        params['b_mu'] = self.b_mu
        params['b_rho'] = self.b_rho
        params['A'] = self.A
        params['A_b'] = self.A_b
        
        return params

class DeepMFNet:
    
    def __init__(self, opt, synD):
        
        self.device = torch.device(opt.placement)
        self.torch_type = opt.torch_type
        
        self.data = synD
        
        self.logger = opt.logger
        self.verbose = opt.verbose
        
        self.M = opt.M
        self.input_dims = opt.input_dim_list
        self.output_dims = opt.output_dim_list
        self.base_dims = opt.base_dim_list
        self.hlayers = opt.hlayers_list

        self.max_epochs = opt.max_epochs
        self.print_freq = opt.print_freq
        
        self.reg_strength = opt.reg_strength
        self.learning_rate = opt.learning_rate
        self.activation = opt.activation
        self.opt_lr = opt.opt_lr
        
        self.nns_list, self.nns_params_list, self.log_tau_list = self.init_model_params()

    
    def init_model_params(self,):
        nns_list = []
        nns_params_list = []
        log_tau_list = []
        
        for m in range(self.M):
            if m == 0:
                in_dim = self.input_dims[m]
            else:
                in_dim = self.input_dims[m] + self.base_dims[m-1]
            #
            layers = [in_dim] + self.hlayers[m] + [self.base_dims[m]] + [self.output_dims[m]]
            if self.verbose:
                print(layers)
            #
            nn = AdaptiveBaseNet(layers, self.activation, self.device, self.torch_type)
            nn_params = nn.parameters()
            log_tau = torch.tensor(0.0, device=self.device, requires_grad=True, dtype=self.torch_type)
            
            nns_list.append(nn)
            nns_params_list.append(nn_params)
            log_tau_list.append(log_tau)
        #
        
        return nns_list, nns_params_list, log_tau_list
    
    def forward(self, X, m, sample=False):
        # first fidelity
        Y_m, base_m = self.nns_list[0].forward(X, sample)
        
        # propagate to the other fidelity levels
        for i in range(1,m+1):
            X_concat = torch.cat((base_m, X), dim=1)
            Y_m, base_m = self.nns_list[i].forward(X_concat, sample)
        #
        
        return Y_m, base_m

    
    def eval_llh(self, X, Y, m):
        Ns = 1
        llh_samples_list = []
   
        for ns in range(Ns):
            pred_sample, _ = self.forward(X, m, sample=True)
            
            log_prob_sample = torch.sum(-0.5*torch.square(torch.exp(self.log_tau_list[m]))*torch.square(pred_sample-Y) +\
                                  self.log_tau_list[m] - 0.5*np.log(2*np.pi))
            
            llh_samples_list.append(log_prob_sample)
        #
        
        return sum(llh_samples_list)

    def batch_eval_llh(self, X_list, Y_list):
        llh_list = []
        for m in range(self.M):
            llh_m = self.eval_llh(X_list[m], Y_list[m], m)
            llh_list.append(llh_m)
        #
        return sum(llh_list)
    
    def batch_eval_kld(self,):
        kld_list = []
        for m in range(self.M):
            kld_list.append(self.nns_list[m]._eval_kld())
        #
        return sum(kld_list)
    
    def batch_eval_reg(self,):
        reg_list = []
        for m in range(self.M):
            reg_list.append(self.nns_list[m]._eval_reg())
        #
        return sum(reg_list)

    def eval_rmse(self, m, N_X, N_Y, train=True):
        # inputs are normalized
        N_pred, _ = self.forward(N_X, m, sample=False)
        scales = self.data.get_scales(m, train)
        
        Y = N_Y*scales['y_std'] + scales['y_mean']
        pred = N_pred*scales['y_std'] + scales['y_mean']
        
        rmse = torch.sqrt(torch.mean(torch.square(Y-pred)))
        n_rmse = rmse/scales['y_std']
        
        return rmse.data.cpu().numpy(), n_rmse.data.cpu().numpy()
    
    def eval_rmse_ground(self, m, N_X, np_y_ground, train=True):
        # inputs are normalized
        N_pred, _ = self.forward(N_X, m, sample=False)
        scales = self.data.get_scales(m, train)
        
        mu = np.mean(np_y_ground)
        sig = np.std(np_y_ground)

        np_N_pred = N_pred.data.cpu().numpy()
        interp_np_N_pred = self.data.interp_to_ground(np_N_pred, m)
        
        interp_np_pred = interp_np_N_pred*sig + mu
        
        rmse = np.sqrt(np.mean(np.square(np_y_ground-interp_np_pred)))
        n_rmse = rmse/sig
        
        return rmse, n_rmse

    def init_train_optimizer(self, lr, weight_decay):
        opt_params = []
        
        for m in range(self.M):
            
            for nn_param_name, nn_param in self.nns_params_list[m].items():
                opt_params.append({'params':nn_param, 'lr':lr})
            #
            opt_params.append({'params':self.log_tau_list[m], 'lr':lr})
            
        #
        
        return Adam(opt_params, lr=lr, weight_decay=weight_decay)

    def train(self,):
        
        if self.verbose:
            print('train the model ...')
        
        X_train_list = []
        y_train_list = []
        np_y_train_ground_list = []
        
        X_test_list = []
        y_test_list = []
        np_y_test_ground_list = []
        
        for m in range(self.M):

            np_X_train, np_y_train, np_y_train_ground = self.data.get_data(m,train=True, normalize=True, noise=0.01)
            np_X_test, np_y_test, np_y_test_ground = self.data.get_data(m,train=False, normalize=True, noise=0.00)
        
            X_train_list.append(torch.tensor(np_X_train, device=self.device, dtype=self.torch_type))
            y_train_list.append(torch.tensor(np_y_train, device=self.device, dtype=self.torch_type))
            np_y_train_ground_list.append(np_y_train_ground)
            
            X_test_list.append(torch.tensor(np_X_test, device=self.device, dtype=self.torch_type))
            y_test_list.append(torch.tensor(np_y_test, device=self.device, dtype=self.torch_type))
            np_y_test_ground_list.append(np_y_test_ground)
        #

        hist_test_rmse = []
        hist_test_ground_rmse = []
        
        optimizer_train = self.init_train_optimizer(self.learning_rate, 0.0)
        
        start_time = time.time()
        
        for epoch in range(self.max_epochs+1):

            optimizer_train.zero_grad()
            loss = -self.batch_eval_llh(X_train_list, y_train_list) + self.batch_eval_kld() + self.reg_strength*self.batch_eval_reg()
            loss.backward(retain_graph=True)
            optimizer_train.step()
            
            if epoch % self.print_freq == 0:
                
                self.logger.write('=============================================================\n')
                self.logger.write(str(epoch) + '-th epoch: loss=' + str(loss.data.cpu().numpy()) +\
                                  ', time_elapsed:' + str(time.time()-start_time) + '\n')
                self.logger.write('=============================================================\n')
                
                buff_test_nRmse = []
                buff_test_nRmse_ground = []

                for m in range(self.M):

                    train_rmse, n_train_rmse = self.eval_rmse(m, X_train_list[m], y_train_list[m], train=True)
                    test_rmse, n_test_rmse = self.eval_rmse(m, X_test_list[m], y_test_list[m], train=False)
                    
                    train_ground_rmse, n_train_ground_rmse = self.eval_rmse_ground(
                        m, X_train_list[m], np_y_train_ground_list[m], train=True)
                    test_ground_rmse, n_test_ground_rmse = self.eval_rmse_ground(
                        m, X_test_list[m], np_y_test_ground_list[m], train=False)
                    
                    buff_test_nRmse.append(n_test_rmse)
                    buff_test_nRmse_ground.append(n_test_rmse)

                    self.logger.write('m='+str(m)+'\n')
                    self.logger.write('  * (origin) train_rmse='+str(n_train_rmse)+', test_rmse='+str(n_test_rmse)+'\n')
                    self.logger.write('  * (ground) train_rmse='+str(n_train_ground_rmse)+',test_rmse='+str(n_test_ground_rmse)+'\n')
                    self.logger.write('  * log_tau_m='+str(self.log_tau_list[m].data.cpu().numpy())+'\n')
                # for m
                
                hist_test_rmse.append(np.array(buff_test_nRmse))
                hist_test_ground_rmse.append(np.array(buff_test_nRmse))
                
            # if epoch
            self.logger.flush()
        # for epoch
        
        N_pred, _ = self.forward(X_test_list[-1], self.M-1, sample=False)
        
        res = {}
        res['test_rmse'] = np.array(hist_test_rmse)
        res['test_ground_rmse'] = np.array(hist_test_ground_rmse)
        res['N_predict'] = N_pred.data.cpu().numpy()
        
        

        return res

