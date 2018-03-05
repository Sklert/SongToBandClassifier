import os
import numpy as np
from scipy.io import wavfile
from hmm_trainer import *
from features_calculator import FeauturesCalculator


class TrainHolder:
    """
    TrainHolder creates, fits, saves, loads HMM

    Attributes
    ----------
    _hmmParams : HMMParams
        Parameters of Gaussian HMM

    _mfccCalculator : FeaturesCalculator
        Opens '.wav' file and calculates mfcc-features

    _models : List of HmmTraines
        Contains HMMs

    save_folder : str
        Path for saving models during training
        If  is None
            models are not saving during training

    """

    def __init__(self, n_components=4, cov_type='diag', n_iter=1000, nfft=512, nmfcc=20, save_folder=None):
        self._hmmParams = HMMParams(n_components, cov_type, n_iter)
        self._mfccCalculator = FeauturesCalculator(nfft=nfft, nmfcc=nmfcc)

        self._models = []
        self.save_folder = save_folder

    def train(self, dataFolder, shuffle=True, seed=None, debug_mode=False):
        """
        dataFolder : str
            path to folder which contains songs for training or
             subfolders with songs for training
            the name of folder/subfolder refers to the name of class

        shuffle : bool
            shuffle mfcc features  of different songs before fitting or not
            shuffling usually increases robustness of model, but also reduces training speed

        seed : int
            seed for random shuffling, useless if shuffle is False

        debug_mode : bool
            prints debug info is true

        """

        np.random.seed(seed)

        if debug_mode:
            print('Current folder is ' + dataFolder)

        lstdirs = [os.path.join(dataFolder, dirname) for dirname in os.listdir(
            dataFolder) if os.path.isdir(os.path.join(dataFolder, dirname))]

        noSubFolders = not lstdirs
        if noSubFolders:
            lstdirs.append(dataFolder)

        for subfolder in lstdirs:
            if not os.path.isdir(subfolder):
                continue

            label = subfolder[subfolder.rfind('/') + 1:]
            featureMatrix = np.array([])

            for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')]:
                filepath = os.path.join(subfolder, filename)

                if debug_mode:
                    print('Current file is ' + filename)

                features = self.getFeaturesfromWaV(filepath)
                featureMatrix = np.append(featureMatrix, features, axis=0) if len(
                    featureMatrix) != 0 else features

            if debug_mode:
                print(subfolder + ' training is started!')

            hmm_trainer = HMMTrainer(hmmParams=self._hmmParams)

            np.random.shuffle(featureMatrix)
            hmm_trainer.train(featureMatrix)

            self._models.append((hmm_trainer, label))

            if self.save_folder is not None:
                hmm_trainer.save(self.save_folder, label, debug_mode)

            if debug_mode:
                print(subfolder + ' training is completed!')
                self._debConvergeInfo(hmm_trainer, label)

        if noSubFolders and debug_mode:
            print(dataFolder + ' training is completed!')

    def test(self, folder_name, result_file=None, table_score=False, debug_mode=False):
        """
        folder_name : str
            path to folder which contains songs for predict

        result_file : str
            path to file for logs. if is None logs are printing to cmd

        table_score : bool
            prints scores of every model on current song

        debug_mode : bool
            prints debug info is true

        """

        if debug_mode:
            print('Tests are started!')

        resultStr = ""
        for input_file in [x for x in os.listdir(folder_name) if x.endswith('.wav')]:
            if debug_mode:
                print('Current file is ' + input_file)
            # Read input file
            filepath = os.path.join(folder_name, input_file)
            self.testFile(filepath, res=resultStr, table_score=table_score)

        if debug_mode:
            print('Concluded!\n Printing results...\n')

        # if result_file = None => print to cmd
        if result_file is None:
            print(resultStr)
        else:
            with open(result_file, 'w') as f:
                f.write(resultStr)

        if debug_mode:
            print('Results are printed!')

    def testFile(self, filepath, result_file=None, res=None, table_score=False):
        """
        filepath : str
            path to song for predict

        result_file : str
            path to file for logs. if is None logs are printing to cmd

        res : str
            string for logs

        table_score : bool
            prints scores of every model on current song

        """
        if res is None:
            res = ""

        features = self.getFeaturesfromWaV(filepath)

        scores = {}
        for hmm_model, label in self._models:
            score = hmm_model.get_score(features)
            scores[label] = score
            if table_score:
                res += '    Score ' + str(score) + ' for ' + label + '\n'

        similarity = sorted(scores.items(), key=lambda t: t[1], reverse=True)
        i = 0
        for item in similarity:
            i += 1
            res += str(i) + ' ' + item[0] + '\n'

        # Print the output
        res += os.path.basename(filepath) + ' to ' + \
            str(similarity[0][0]) + '\n'

        if result_file is None:
            print(res)
        else:
            with open(result_file, 'w') as f:
                f.write(res)

    def save(self, folder_name, debug_mode=False):
        for model, label in self._models:
            model.save(folder_name, label, debug_mode)

    def load(self, folder_name, class_names, debug_mode=False):
        for label in class_names:
            model = HMMTrainer()
            model.load(folder_name, label, debug_mode)
            self._models.append((model, label))

    def getFeaturesfromWaV(self, filename):
        return self._mfccCalculator.getFeaturesfromWaV(filename)

    def _debConvergeInfo(self, model, label):
        print(label)
        print(model.get_monitorInfo())
        print('\n')
