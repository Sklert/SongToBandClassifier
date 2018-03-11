import os
from hmmlearn import hmm
from sklearn.externals import joblib
import numpy as np
import warnings
warnings.simplefilter('ignore', DeprecationWarning)


class HMMParams:
    """
    Defines significant params for HMM
    """

    def __init__(self, n_components=4, cov_type='diag', n_iter=1000, tol=1e-4):
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.tol = tol


class HMMTrainer:
    """
    Wrapper for GaussianHMM 
    """

    def __init__(self, hmmParams=HMMParams()):

        self._hmm = hmm.GaussianHMM(n_components=hmmParams.n_components,
                                    covariance_type=hmmParams.cov_type, n_iter=hmmParams.n_iter, tol=hmmParams.tol)

    def train(self, X):
        np.seterr(all='ignore')
        self._hmm.fit(X)

    def get_score(self, input_data):
        return self._hmm.score(input_data)

    def get_score_samples(self, input_data):
        return self._hmm.score_samples(input_data)

    def get_monitorInfo(self):
        return self._hmm.monitor_

    def save(self, folder_name, class_name, debug_mode=False):
        """
        folder_name : str
            folder which contains output

        class_name : str
            name of output file

        debug_mode : bool
            prints debug info is true

        """

        if folder_name[-1] != '/':
            folder_name += '/'

        filename = folder_name + class_name + '.pkl'

        if debug_mode:
            print('Start saving to ' + filename)

        joblib.dump(self._hmm, filename)

        if debug_mode:
            print('Saving completed ' + filename)

    def load(self, folder_name, class_name, debug_mode=False):
        """
        folder_name : str
            folder which contains output

        class_name : str
            name of output file

        debug_mode : bool
            prints debug info is true

        """

        for filename in [x for x in os.listdir(folder_name) if (x.endswith('.pkl') and (class_name in x))]:
            filepath = os.path.join(folder_name, filename)
            if debug_mode:
                print('Start loading  ' + filename)

            self._hmm = joblib.load(filepath)

            if debug_mode:
                print('Loading completed ' + filename)
