from sklearn.metrics import f1_score, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error, mean_absolute_error


class Test:

    def __init__(self, pred, test):
        """
        :param pred: pd.Series or pd.DataFrame
        :param test: pd.Series or pd.DataFrame
        """
        self.pred = pred
        self.test = test
        self.f1 = None
        self.accuracy = None
        self.confusion = None
        self.mse = None
        self.mae = None
        self.rmse = None

    def get_f1(self):
        """
        Shows and returns f1 score. It can calculate score if it don't yet calculated.

        :return: float
        """

       pass

    def get_accuracy(self):
        """
        Shows and returns accuracy score. It can calculate score if it don't yet calculated.
        :return: flat
        """
        if self.

    def get_confusion(self):
        """
        Shows confusion matrix. It can calculate score if it don't yet calculated.
        :return:
        """
        pass

    def quick_clf(self):
        """
        Used to get quick information about classifier's quality: shows f1 and accuracy scores and displays confusion
        matrix.

        :return: None
        """
        pass

    def get_mse(self):
        """
        Shows and returns MSE score. It can calculate score if it don't yet calculated.

        :return: float
        """
        pass

    def get_mae(self):
        """
        Shows and returns MAE score. It can calculate score if it don't yet calculated.

        :return:
        """
        pass

    def get_rmse(self):
        """
        Shows and returns RMSE score. It can calculate score if it don't yet calculated.

        :return: float
        """
        pass

    def quick_reg(self):
        """
        Used to get quick info about regressor's quality: shows mse, mae and rmse scores.

        :return: None
        """
        pass

