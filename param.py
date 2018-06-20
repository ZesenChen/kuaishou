import numpy as np


reg_log = 'user_register_log.txt'
video_log = 'video_create_log.txt'
launch_log = 'app_launch_log.txt'
act_log = 'user_activity_log.txt'

TRAIN_PREDICT_DAY = range(24,31)
TRAIN_REGISTER_DAT = range(1,24)
TRAIN_ACT_DAT = range(8,24)

TEST_PREDICT_DAY = range(31,38)
TEST_REGISTER_DAT = range(1,31)
TEST_ACT_DAT = range(15,31)

WEIGHT = np.array([i for i in range(len(TEST_ACT_DAT))])
