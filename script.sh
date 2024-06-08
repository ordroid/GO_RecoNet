#!/bin/bash

# Setup env
#Install envoriment under the name <ENV_NAME>:
#conda activate <ENV_NAME>

python main.py --lr 0.01 --num-workers 16 --report-interval 1 --batch-size 16 --drop-rate 0.4
#--mask-lr 0.1 --learn-mask