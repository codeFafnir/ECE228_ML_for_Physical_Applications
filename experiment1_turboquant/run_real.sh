#!/bin/bash
cd /Users/raghu/coding/ECE228/ECE228_ML_for_Physical_Applications/PINN_channel-estimation-main
/Users/raghu/coding/myenv/bin/python \
  /Users/raghu/coding/ECE228/ECE228_ML_for_Physical_Applications/experiment1_turboquant/eval_all.py \
  --checkpoint /Users/raghu/coding/ECE228/ECE228_ML_for_Physical_Applications/simple_ls_0_val.pth \
  --synthetic \
  --n_cal 128 \
  --n_val_synthetic 256 \
  --batch_size 8 \
  --bits 3 \
  --skip_tq
