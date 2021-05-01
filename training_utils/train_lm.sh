#!/bin/bash

# CUDA_LAUNCH_BLOCKING=1 python train_roberta_lm.py \
# 	/home/alka/Documents/zindi_challenges/zindi_text_classif/inputs/tunisian_sent_an_arabizi/roberta_lm_configs/config1.json

# CUDA_LAUNCH_BLOCKING=1 python train_roberta_lm.py\
# 	/home/alka/Documents/zindi_challenges/zindi_text_classif/inputs/tunisian_sent_an_arabizi/roberta_lm_configs/config2.json

CUDA_LAUNCH_BLOCKING=1 python train_roberta_lm.py \
	/home/alka/Documents/zindi_challenges/zindi_text_classif/inputs/tunisian_sent_an_arabizi/roberta_lm_configs/config1_aug.json