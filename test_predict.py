from trainerholder import TrainHolder
import config


class TestPredict:
    """
    Sample describes how to use TrainHolder

    Parameters
    ----------

    models_folder : str
        path to folder which contains saved models

    class_names : list of str
        names of classes. saved model files must contain these names
    """

    def __init__(self, models_folder, class_names):
        self.clf = TrainHolder(n_components=config.n_components, cov_type=config.cov_type,
                               n_iter=config.n_iter, nfft=config.nfft, nmfcc=config.nmfcc)
        self.clf.load(models_folder, class_names, config.debug_mode)

    def predictFolder(self, test_folder):
        self.clf.test(test_folder, table_score=config.table_score,
                      debug_mode=config.debug_mode, result_file=config.result_file)

    def predictFile(self, test_file):
        self.clf.testFile(test_file, table_score=config.table_score,
                          result_file=config.result_file)

    def checkConverges(self):
        self.clf.whoConverged()


if __name__ == "__main__":

    # folder which contains trained models
    models_folder = config.models_folder

    # folder which contains test songs
    test_folder = config.test_folder

    # define here which classes should be loaded
    # name of file with model should contain name of corresponding class
    class_names = ['Metallica', 'Nirvana', 'Motorhead',
                   'Pink Floyd', 'Anathema', 'Hollywood Undead', 'The XX']

    predictor = TestPredict(models_folder, class_names)
    predictor.predictFolder(test_folder)
