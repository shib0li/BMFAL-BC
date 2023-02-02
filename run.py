import os

import fire
from tqdm.auto import trange 
import pickle
from hdf5storage import loadmat
import h5py

import numpy as np
import torch
import torch.distributions as distributions
from torch.optim import Adam
from torch.optim import LBFGS
import time
import datetime

from config import opt

import dataset_active as dataset
from models.submod_dmfal import SubmodBatchDMFAL

def create_path(path): 
    try:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        #
        print("Directory '%s' created successfully" % (path))
    except OSError as error:
        print("Directory '%s' can not be created" % (path))
    #
    
def dump_pred_to_h5f(hist_pred, h5fname, comp=None):
 
    try:
        hf = h5py.File(h5fname, 'w')
        Npred = len(hist_pred)
        
        group_pred_test = hf.create_group('hist_N_pred')
        
        for n in range(Npred):
            key = 'predict_at_t' + str(n)
            pred = hist_pred[n]
            print(pred.shape)
            group_pred_test.create_dataset(key, data=pred, compression=comp)
        #
    except:
        print('ERROR occurs when WRITE the h5 object...')
    finally:
        hf.close()

def evaluation(**kwargs):
    
    opt._parse(kwargs)
    
    res_path = os.path.join('__results__', opt.domain)
    log_path = os.path.join('__log__', opt.domain)
    
    create_path(res_path)
    create_path(log_path)

    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    res_file_name = 'res_'+opt.heuristic + '_trail' + str(opt.trial) + '.pickle'
    pred_file_name = 'pred_'+opt.heuristic + '_trail' + str(opt.trial) + '.h5'
    log_file_name = 'log_'+opt.heuristic + '_trail' + str(opt.trial) + '.txt'
    
    logger = open(os.path.join(log_path, log_file_name), 'w+') 
    
    opt.logger = logger
    
    hlayers_list = []
    for i in range(len(opt.hlayers_w)):
        w = opt.hlayers_w[i]
        d = opt.hlayers_d[i]
        hlayers_list.append([w]*d)  
    opt.hlayers_list = hlayers_list

    synD = dataset.Dataset(opt.domain, opt.trial)
        
    model = SubmodBatchDMFAL(opt, synD)

    hist_cost = []
    hist_test_nRmse = []
    hist_test_nRmse_ground = []
    hist_N_pred = []
    
    accum_cost = np.sum(np.array(opt.penalty) * np.array(synD.Ntrain_list))
    hist_cost.append(accum_cost)

    for t in trange(opt.T):
        
        t_init = time.time()
        opt.logger.write('#############################################################\n')
        opt.logger.write('                     Active Step #' + str(t)+'\n')
        opt.logger.write('#############################################################\n')
        opt.logger.flush()
        
        
        curr_res = model.train()
        t_train = time.time()-t_init;
        
        np_X_batch, m_batch = model.submod_batch_query()

        t_query = time.time()-t_init-t_train

        for j in range(len(m_batch)):
            synD.append(np_X_batch[j], m_batch[j])
        #
        t_append = time.time()-t_init-t_train-t_query

        opt.logger.write('------------------------------------------------\n')
        opt.logger.write(' * Train time:' + str(t_train) + '\n')
        opt.logger.write(' * Total query time:' + str(t_query) + '\n')
        opt.logger.write(' * Append time:' + str(t_append) + '\n')
        opt.logger.write(' * Time cost per query:' + str((t_query+t_train)/len(np_X_batch)) + '\n')
        opt.logger.write(' * Time cost per append:' + str(t_append/len(np_X_batch)) + '\n')
        opt.logger.write('------------------------------------------------\n')
            
        accum_cost = np.sum(np.array(opt.penalty) * np.array(synD.Ntrain_list))
        hist_cost.append(accum_cost)
        hist_test_nRmse.append(curr_res['test_rmse'])
        hist_test_nRmse_ground.append(curr_res['test_ground_rmse'])
        
        act_res = {}
        act_res['hist_cost'] = hist_cost
        act_res['hist_test_nRmse'] = hist_test_nRmse
        act_res['hist_test_nRmse_ground'] = hist_test_nRmse_ground
        
        dump_file = open(os.path.join(res_path, res_file_name), "wb")
        pickle.dump(act_res, dump_file)
        dump_file.close()
        
        #dump_pred_to_h5f(hist_N_pred, os.path.join(res_path, pred_file_name))

        
if __name__=='__main__':
    fire.Fire()