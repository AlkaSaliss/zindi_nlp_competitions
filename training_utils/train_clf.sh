# !/bin/bash

python train_roberta_clf.py ../../inputs/tunisian_sent_an_arabizi/roberta_classif_configs/configs/conf_noproc.json
python train_roberta_clf.py ../../inputs/tunisian_sent_an_arabizi/roberta_classif_configs/configs/conf_fullproc.json
# python train_roberta_clf.py ../../inputs/tunisian_sent_an_arabizi/roberta_classif_configs/configs/conf_noproc_sched.json
# python train_roberta_clf.py ../../inputs/tunisian_sent_an_arabizi/roberta_classif_configs/configs/conf_fullproc_sched.json