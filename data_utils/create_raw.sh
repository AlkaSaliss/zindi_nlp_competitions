#!/bin/bash

python create_pretraining_data.py --output_path /home/alka/Documents/zindi_challenges/zindi_text_classif/outputs/data/raw \
	--flag noproc

python create_pretraining_data.py \
	--procs clean_noise to_lower clean_word_rep clean_char_rep clean_whitespace \
	--output_path /home/alka/Documents/zindi_challenges/zindi_text_classif/outputs/data/raw \
	--flag fullproc
