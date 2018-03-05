from trainerholder import TrainHolder


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
        self.clf = TrainHolder(
            n_components=20, cov_type='diag', n_iter=1000, nfft=1024, nmfcc=40)
        self.clf.load(models_folder, class_names, False)

    def predictFolder(self, test_folder):
        self.clf.test(test_folder, table_score=True,
                      debug_mode=False, result_file="result.txt")

    def predictFile(self, test_file):
        self.clf.testFile(test_file, table_score=False,
                          result_file="result.txt")

    def checkConverges(self):
        self.clf.whoConverged()


if __name__ == "__main__":
    models_folder = "../Saves"
    test_folder = "../Tests"

    class_names = ['Metallica', 'Nirvana', 'Motorhead',
                   'Pink Floyd', 'Anathema', 'Hollywood Undead', 'The XX']

    predictor = TestPredict(models_folder, class_names)
    predictor.predictFolder(test_folder)
