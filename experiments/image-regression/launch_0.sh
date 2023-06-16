#!/bin/bash
CUDA_VISIBLE_DEVICES=2
python run_usa_vars_exp.py config_file=/mnt/SSD2/nils/uq-method-box/experiments/image-regression/configs/usa_vars/gaussian_nll.yaml experiment.seed=0 trainer.devices=[0]