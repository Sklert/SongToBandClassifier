# SongToBandClassifier
Classificate songs to bands by Hidden Markov Models

# Train Models
Fast start:
1) Put every train song with **'.wav'** format in folder with the name of corresponding band(class).
2) Move these folder-classes to common parent folder. Parent folder should contain only subfolder-classes.
3) In file [config.py](https://github.com/Sklert/SongToBandClassifier/blob/master/config.py)
set parent folder from step 2 to `train_folder` and output models folder to `save_folder`.
4) Run [test_createmodels.py](https://github.com/Sklert/SongToBandClassifier/blob/master/test_createmodels.py) 


# Test 
Fast start:
1) Put every test song with **'.wav'** format in one folder
2) In file [config.py](https://github.com/Sklert/SongToBandClassifier/blob/master/config.py)
set this folder to `test_folder` and folder with trained models to `models_folder`.
3) Run [test_predict.py](https://github.com/Sklert/SongToBandClassifier/blob/master/test_predict.py)

# Advanced
If you want to increase training **speed**, you can

in [config.py](https://github.com/Sklert/SongToBandClassifier/blob/master/config.py):
* increase number of max threads for classes (`max_threads`)
* decrease number of components (`n_components`)
* decrease number of mfcc or fourier transform points (`nmfcc`, `nfft`)  -- not recommended

in [test_createmodels.py](https://github.com/Sklert/SongToBandClassifier/blob/master/test_createmodels.py):
* set `shuffle=False`

If you want to increase **accuracy**, you can 

in [config.py](https://github.com/Sklert/SongToBandClassifier/blob/master/config.py):
* increase number of components (`n_components`), 20 is optimal
* set `cov_type="full"` or other converge parameters, look at [hmmlearn](https://github.com/hmmlearn/hmmlearn)

in [test_createmodels.py](https://github.com/Sklert/SongToBandClassifier/blob/master/test_createmodels.py) :
* set `shuffle=True`, if it became `False` (default is `True`)

For more output info and other features look at [config.py](https://github.com/Sklert/SongToBandClassifier/blob/master/config.py), 
[test_createmodels.py](https://github.com/Sklert/SongToBandClassifier/blob/master/test_createmodels.py) 
and [test_predict.py](https://github.com/Sklert/SongToBandClassifier/blob/master/test_predict.py)
