import math

from sklearn.metrics import f1_score, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd


class Test:

    def __init__(self, test, pred):
        """
        :param pred: pd.Series or pd.DataFrame
        :param test: pd.Series or pd.DataFrame
        """
        self.pred = pred
        self.test = test
        self.f1 = None
        self.accuracy = None
        self.mse = None
        self.mae = None
        self.rmse = None

    def get_f1(self):
        """
        Shows and returns f1 score. It can calculate score if it don't yet calculated.

        :return: float
        """

        if self.f1 is None:
            self.f1 = f1_score(self.test, self.pred)

        print(f'f1: \t{self.f1: .2f}')
        return self.f1

    def get_accuracy(self):
        """
        Shows and returns accuracy score. It can calculate score if it don't yet calculated.
        :return: float
        """
        if self.accuracy is None:
            self.accuracy = accuracy_score(self.test, self.pred)

        print(f'accuracy: \t{self.accuracy: .2f}')
        return self.accuracy

    def get_confusion(self):
        """
        Shows confusion matrix. It can calculate score if it don't yet calculated.

        :return: sklearn.metrics.ConfusionMatrixDisplay
        """

        ConfusionMatrixDisplay.from_predictions(self.test, self.pred)
        plt.show()

    def quick_clf(self):
        """
        Used to get quick information about classifier's quality: shows f1 and accuracy scores and displays confusion
        matrix.

        :return: None
        """

        self.get_f1()
        self.get_accuracy()
        self.get_confusion()

    def get_mse(self):
        """
        Shows and returns MSE score. It can calculate score if it don't yet calculated.

        :return: float
        """

        if self.mse is None:
            self.mse = mean_squared_error(self.test, self.pred)

        print(f'MSE: \t{self.mse: .2f}')
        return self.mse

    def get_mae(self):
        """
        Shows and returns MAE score. It can calculate score if it don't yet calculated.

        :return:
        """
        if self.mae is None:
            self.mae = mean_absolute_error(self.test, self.pred)

        print(f'MAE: \t{self.mae: .2f}')

    def get_rmse(self):
        """
        Shows and returns RMSE score. It can calculate score if it don't yet calculated.

        :return: float
        """
        if self.rmse is None:
            self.rmse = math.sqrt(self.mse)

        print(f'RMSE: \t{self.rmse: .2f}')
        return self.rmse

    def quick_reg(self):
        """
        Used to get quick info about repressor's quality: shows mse, mae and rmse scores.

        :return: None
        """

        self.get_mse()
        self.get_mae()
        self.get_rmse()


if __name__ == '__main__':

    y_test = pd.Series([0, 0, 1, 0, 1, 1, 0])
    y_pred = pd.Series([0, 1, 0, 0, 1, 1, 1])

    t = Test(y_test, y_pred)
    t.get_f1()
    t.get_accuracy()
