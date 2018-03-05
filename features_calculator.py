import librosa
from librosa.feature import mfcc
import sklearn.preprocessing


class FeauturesCalculator:
    """
    Opens file, calculates features and returns them

    Attributes
    ----------
    _res_type : str
        parameter for librosa.load
        defines resample type

    _nfft : int
    _nmfcc : int
    _hop_length : int
        info for calculating mfcc (for librosa.feature.mfcc)

    _scale : bool
        scale mfcc or not

    """

    def __init__(self, nfft, nmfcc, hop_length=512, res_type='scipy', scale=True):
        self._res_type = res_type

        self._nfft = nfft
        self._nmfcc = nmfcc
        self._hop_length = hop_length

        self._scale = scale

    def getFeaturesfromWaV(self, filename):
        audio, sampling_freq = librosa.load(
            filename, sr=None, res_type=self._res_type)

        features = librosa.feature.mfcc(
            audio, sampling_freq, n_mfcc=self._nmfcc, n_fft=self._nfft, hop_length=self._hop_length)

        if self._scale:
            features = sklearn.preprocessing.scale(features)

        return features.T
