#!/bin/sh

# MAX_EPOCHS=4000
# BATCH_SIZE=20
# HORIZON=20
# DEVICE='cuda:0'

MAX_EPOCHS=10
BATCH_SIZE=5
HORIZON=20
DEVICE='cuda:0'
    
python run.py evaluation -heuristic 'GreedySubmod' -domain 'Poisson2' -M 2  -trial 1 \
    -input_dim_list [5,5] -output_dim_list [256,1024] -base_dim_list [32,32] \
    -hlayers_w [40,40] -hlayers_d [2,2] \
    -activation 'tanh' -penalty [1,2] \
    -print_freq 10 -max_epochs=$MAX_EPOCHS -learning_rate 1e-3 -reg_strength 1e-5 \
    -opt_lr 1e-1 -batch_size=$BATCH_SIZE -T $HORIZON \
    -placement $DEVICE 
    
   