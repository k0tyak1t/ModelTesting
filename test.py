import math

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


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
        self.confusion = None

        self.mse = None
        self.mae = None
        self.rmse = None

    def get_f1(self):
        """
        Returns f1 score. It can calculate score if it don't yet calculated.

        :return: float
        """

        if self.f1 is None:
            self.f1 = f1_score(self.test, self.pred)

        return self.f1

    def get_accuracy(self):
        """
        Returns accuracy score. It can calculate score if it don't yet calculated.
        :return: float
        """
        if self.accuracy is None:
            self.accuracy = accuracy_score(self.test, self.pred)

        return self.accuracy

    def get_confusion(self):
        """
        Returns confusion matrix
        :return: np.array
        """
        if self.confusion is None:
            self.confusion = confusion_matrix(self.test, self.pred)

        return self.confusion

    def show_confusion(self):
        """
        Shows confusion matrix

        :return: None
        """
        try:
            ConfusionMatrixDisplay(self.get_confusion()).plot()
            plt.show()
        except AttributeError:
            if matplotlib.get_backend() == 'TkAgg':
                raise Exception('Critical Error! Please report.')
            matplotlib.use('TkAgg')
            self.show_confusion()

    def quick_clf(self):
        """
        Used to get quick information about classifier's quality: shows f1 and accuracy scores and displays confusion
        matrix.

        :return: None
        """

        self.show('f1', self.get_f1())
        self.show('accuracy', self.get_accuracy())
        self.show_confusion()

    def get_mse(self):
        """
        Returns MSE score. It can calculate score if it don't yet calculated.

        :return: float
        """

        if self.mse is None:
            self.mse = mean_squared_error(self.test, self.pred)

        return self.mse

    def get_mae(self):
        """
        Returns MAE score. It can calculate score if it don't yet calculated.

        :return: float
        """
        if self.mae is None:
            self.mae = mean_absolute_error(self.test, self.pred)

        return self.mae

    def get_rmse(self):
        """
        Returns RMSE score. It can calculate score if it don't yet calculated.

        :return: float
        """
        if self.rmse is None:
            self.rmse = math.sqrt(self.mse)

        return self.rmse

    @staticmethod
    def show(name, value):
        """
        Prints value in the standard form: name:value (x.xx)

        :param name: str
        :param value: float
        :return: None
        """
        print(f'{name}:    {value: .2f}')

    def quick_reg(self):
        """
        Used to get quick info about repressor's quality: shows mse, mae and rmse scores.

        :return: None
        """

        self.show('MSE', self.get_mse())
        self.show('MAE', self.get_mae())
        self.show('RMSE', self.get_rmse())

# ============================================== #
# The code below is the only for module testing! #
# Should be used like module                     #
# ============================================== #


def testcases():
    y_test = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1])

    t = Test(y_test, y_pred)
    t.quick_reg()
    print('=' * 10)
    t.quick_clf()


if __name__ == '__main__':
    testcases()


