
## Section1
# With and Without Data Augmentation
AUG=0 ROT=0 EXP_NAME=WO_AUG  python ./q1_q2_classification/train_q1.py
AUG=1 ROT=0 EXP_NAME=W_AUG  python ./q1_q2_classification/train_q1.py

# log: ./runs/WO_AUG
# log: ./runs/W_AUG

tensorboard --logdir runs

# Best model 
AUG=0 ROT=0 EXP_NAME=WO_AUG_e10_lr1e-3  python q1_q2_classification/train_q1.py
AUG=1 ROT=0 EXP_NAME=W_AUG_e10_lr1e-3  python q1_q2_classification/train_q1.py
AUG=1 ROT=0 EXP_NAME=W_AUG_e20_lr1e-3  python q1_q2_classification/train_q1.py

# Best: test map = 0.2829612980392344
AUG=1 ROT=0 EXP_NAME=W_AUG_e30_lr1e-3  python q1_q2_classification/train_q1.py

AUG=1 ROT=0 EXP_NAME=W_AUG_e40_lr2e-3  python q1_q2_classification/train_q1.py
AUG=1 ROT=0 EXP_NAME=W_AUG_e40_lr1e-3  python q1_q2_classification/train_q1.py
AUG=1 ROT=0 EXP_NAME=W_AUG_e40_lr5e-4  python q1_q2_classification/train_q1.py

tensorboard --logdir runs