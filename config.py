# file contains parameters for test_createmodels.py and test_predict.py


# classifier params
n_components = 12
cov_type = 'diag'
n_iter = 1000

# MFCC params
nmfcc = 40
nfft = 1024


##TRAIN##

# define folder which contains songs for training or subfolders of different classes with songs for training
train_folder = "../Trains"

# folder for saving models
save_folder = "../Saves/"

# debug mode for training
debug_train = True

# max number of threads for training
max_threads = 2


##TEST##

# folder which containst testing songs
test_folder = "../Tests"

# folder which contains saved models
models_folder = save_folder

# debug mode
debug_mode = True

# output scores for test songs under every model
table_score = False

# file for logs. if it is None, no file will be created
result_file = "result.txt"
