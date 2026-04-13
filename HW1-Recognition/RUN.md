
## Section1
# With and Without Data Augmentation
AUG=0 ROT=0 EXP_NAME=WO_AUG  python ./q1_q2_classification/train_q1.py
AUG=1 ROT=0 EXP_NAME=W_AUG  python ./q1_q2_classification/train_q1.py

# log: ./runs/WO_AUG
# log: ./runs/W_AUG

tensorboard --logdir runs

# Best model 
# test map = 0.31636201919230833
AUG=1 ROT=0 EXP_NAME=W_AUG_e50_lr1e-3_stepsize12 python q1_q2_classification/train_q1.py

# test map = 0.301437851904029
AUG=1 ROT=0 EXP_NAME=W_AUG_e30_lr6e-4  python q1_q2_classification/train_q1.py
# test map = 0.29125182061402394
AUG=1 ROT=0 EXP_NAME=W_AUG_e30_lr8e-4_b8  python q1_q2_classification/train_q1.py
# test map = 0.3113117084827654
AUG=1 ROT=0 EXP_NAME=W_AUG_e30_lr8e-4  python q1_q2_classification/train_q1.py
# test map = 0.31005256029929057
AUG=1 ROT=0 EXP_NAME=W_AUG_e30_lr8e-4_stepsize6  python q1_q2_classification/train_q1.py
# test map = 0.2829612980392344
AUG=1 ROT=0 EXP_NAME=W_AUG_e30_lr1e-3  python q1_q2_classification/train_q1.py
# test map = 0.2499471592149447
AUG=1 ROT=0 EXP_NAME=W_AUG_e30_lr3e-3  python q1_q2_classification/train_q1.py
# test map = 0.2782583664590525
AUG=1 ROT=0 EXP_NAME=W_AUG_e50_lr8e-4 python q1_q2_classification/train_q1.py
# test map = 0.283219275756173
AUG=1 ROT=0 EXP_NAME=W_AUG_e50_lr1e-3 python q1_q2_classification/train_q1.py
# test map = 0.31180441489611704
AUG=1 ROT=0 EXP_NAME=W_AUG_e50_lr1e-3_stepsize6 python q1_q2_classification/train_q1.py
# test map = 0.31539310761239236
AUG=1 ROT=0 EXP_NAME=W_AUG_e50_lr1e-3_stepsize9 python q1_q2_classification/train_q1.py
# test map = 0.31636201919230833
AUG=1 ROT=0 EXP_NAME=W_AUG_e50_lr1e-3_stepsize12 python q1_q2_classification/train_q1.py
# test map = 0.2910344952928778
AUG=1 ROT=0 EXP_NAME=W_AUG_e50_lr1e-3_stepsize12_b8 python q1_q2_classification/train_q1.py
# test map = 0.30399014746642955
AUG=1 ROT=0 EXP_NAME=W_AUG_e50_lr1e-3_stepsize12_b32 python q1_q2_classification/train_q1.py
# test map = 0.3116855141707781
AUG=1 ROT=0 EXP_NAME=W_AUG_e50_lr1e-3_stepsize15 python q1_q2_classification/train_q1.py
# test map = 0.25877240551318326
AUG=1 ROT=0 EXP_NAME=W_AUG_e50_lr3e-3_stepsize6  python q1_q2_classification/train_q1.py
# test map = 0.2357070755956814
AUG=1 ROT=0 EXP_NAME=W_AUG_e50_lr3e-3  python q1_q2_classification/train_q1.py
# test map = 0.15117998014427836 
AUG=1 ROT=0 EXP_NAME=W_AUG_e50_lr4e-3 python q1_q2_classification/train_q1.py
# test map = 0.18338498440472478 
AUG=1 ROT=0 EXP_NAME=W_AUG_e50_lr4e-3_stepsize6  python q1_q2_classification/train_q1.py

tensorboard --logdir runs

## Section2
# best model
# test map = 0.7952437246140882
AUG=1 ROT=1 EXP_NAME=W_AUG_ROT1_50_lr1e-2_bs32_stepsize15_g0p1_wdec2e-4_mom0p9_SGD_saveatend_224x224  python q1_q2_classification/train_q2.py

# test map = 0.7666439966807295 
AUG=1 ROT=0 EXP_NAME=W_AUG_50_lr1e-4_bs32_stepsize15_g0p1_224x224  python q1_q2_classification/train_q2.py

# test map = 0.7680421465149706
AUG=1 ROT=1 ROT_DEG=10 EXP_NAME=W_AUG_ROTDEG10_50_lr3e-5_bs32_stepsize5_g0p3_224x224  python q1_q2_classification/train_q2.py

# test map = 0.7768856466076997
AUG=1 ROT=1 ROT_DEG=20 EXP_NAME=W_AUG_ROTDEG20_50_lr8e-5_bs32_stepsize10_g0p3_224x224  python q1_q2_classification/train_q2.py

# test map = 0.7769190310988161
AUG=1 ROT=1 ROT_DEG=20 EXP_NAME=W_AUG_ROTDEG20_50_lr8e-5_bs32_stepsize10_g0p3_wdec2e-4_224x224  python q1_q2_classification/train_q2.py

# test map = 0.7777169617967711
AUG=1 ROT=1 ROT_DEG=20 EXP_NAME=W_AUG_ROTDEG20_50_lr8e-5_bs32_stepsize10_g0p3_wdec5e-4_224x224  python q1_q2_classification/train_q2.py

# test map = 0.7764419259881116
AUG=1 ROT=1 ROT_DEG=20 EXP_NAME=W_AUG_ROTDEG20_50_lr8e-5_bs32_stepsize10_g0p3_wdec8e-4_224x224  python q1_q2_classification/train_q2.py

# test map = 0.7633995497493679
AUG=1 ROT=1 ROT_DEG=20 EXP_NAME=W_AUG_ROTDEG20_50_lr8e-5_bs32_stepsize12_g0p5_224x224  python q1_q2_classification/train_q2.py

# test map = 0.7923992114361829
AUG=1 ROT=0 EXP_NAME=W_AUG_50_lr1e-2_bs32_stepsize15_g0p1_wdec1e-4_mom0p9_SGD_224x224  python q1_q2_classification/train_q2.py

# test map = 0.7905566231261512
AUG=1 ROT=0 EXP_NAME=W_AUG_50_lr1e-2_bs32_stepsize10_g0p1_wdec2e-4_mom0p9_SGD_224x224  python q1_q2_classification/train_q2.py

# test map = 0.792712092973759
AUG=1 ROT=0 EXP_NAME=W_AUG_50_lr1e-2_bs32_stepsize15_g0p1_wdec2e-4_mom0p9_SGD_224x224  python q1_q2_classification/train_q2.py

# test map = 0.7948790271382029
AUG=1 ROT=1 EXP_NAME=W_AUG_ROT1_50_lr1e-2_bs32_stepsize15_g0p1_wdec2e-4_mom0p9_SGD_224x224  python q1_q2_classification/train_q2.py

# test map = 0.794860565263314
AUG=1 ROT=1 EXP_NAME=W_AUG_ROT1_50_lr1e-2_bs32_stepsize15_g0p1_wdec3e-4_mom0p9_SGD_224x224  python q1_q2_classification/train_q2.py

# test map = 0.7905757615663559
AUG=1 ROT=0 EXP_NAME=W_AUG_50_lr1e-2_bs32_stepsize20_g0p1_wdec2e-4_mom0p9_SGD_224x224  python q1_q2_classification/train_q2.py

# test map = 0.7856927109653921
AUG=1 ROT=0 EXP_NAME=W_AUG_50_lr5e-3_bs32_stepsize15_g0p1_wdec1e-4_mom0p9_SGD_224x224  python q1_q2_classification/train_q2.py

# test map = 0.7855772836016388
AUG=1 ROT=0 EXP_NAME=W_AUG_50_lr2e-2_bs32_stepsize15_g0p1_wdec1e-4_mom0p9_SGD_224x224  python q1_q2_classification/train_q2.py