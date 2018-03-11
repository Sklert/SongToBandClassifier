from trainerholder import TrainHolder
from parallelwork import parallel_train
import config

# define folder which contains songs for training or subfolders of different classes with songs for training
train_folder = config.train_folder

# folder for saving model
save_folder = config.save_folder

# create classificator
clf = TrainHolder(n_components=config.n_components, cov_type=config.cov_type,
                  n_iter=config.n_iter, nfft=config.nfft, nmfcc=config.nmfcc)

# define folder for saving models during training
clf.save_folder = save_folder

# create threads for different classes
parallel_train(clf, train_folder, debug_mode=config.debug_train,
               max_threads=config.max_threads)

# just train in one threads
# clf.train(train_folder,debug_mode=config.debug_train)

# save trained models mannually (don't require if clf.save_folder is not None)
#clf.save(save_folder, config.debug_train)
