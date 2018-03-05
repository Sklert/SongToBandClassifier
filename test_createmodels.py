from trainerholder import TrainHolder
from parallelwork import parallel_train

# define folder which contains songs for training or subfolders of different classes with songs for training
train_folder = "../Trains"

# folder for saving model
save_folder = "../Saves/"

# create classificator
clf = TrainHolder(n_components=20, cov_type='diag',
                  n_iter=1000, nfft=1024, nmfcc=40)

# define folder for saving models during training
clf.save_folder = save_folder

# create threads for different classes
parallel_train(clf, train_folder, debug_mode=True, max_threads=2)

# just train in one threads
#clf.train(train_folder,debug_mode=True)

# save trained models mannually (don't require if clf.save_folder is not None)
#clf.save(save_folder, True)
