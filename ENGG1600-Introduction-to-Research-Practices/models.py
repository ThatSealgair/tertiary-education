from sklearn.linear_model import *
from sklearn.cluster import *
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn import neighbors
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


class SupportVectorAlgorithm:
    def __init__(self, x, y, test_size=0.2, shuffle=True):
        """
        :param x: Dataset
        :param y: Labels
        :param test_size: Percentage of data used for reference.
        :param shuffle: Is the data randomised?
        """
        self._X = x
        self._y = y
        self._test_size = test_size
        self._shuffle = shuffle
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._X,
                                                                                    self._y,
                                                                                    test_size=self._test_size,
                                                                                    shuffle=self._shuffle)
        self._model = svm.SVC()
        self._fit_data = self._model.fit(self._X_train, self._y_train)
        self._prediction = self._model.predict(self._X_test)
        self._accuracy = accuracy_score(self._y_test, self._prediction)
        self._f1 = f1_score(self._y_test, self._prediction, average='weighted')
        self._recall = recall_score(y_true=self._y_test, y_pred=self._prediction, average='weighted')
        self._precision = precision_score(y_true=self._y_test, y_pred=self._prediction, average='weighted')
        self._kappa = cohen_kappa_score(self._y_test, self._prediction)
        self._f2 = fbeta_score(self._y_test, self._prediction, beta=0.5, average='weighted')

    def predict(self):
        return self._prediction

    def actual(self):
        return self._y_test

    def accuracy(self):
        return self._accuracy

    def precision(self):
        return self._precision

    def f1(self):
        return self._f1

    def f2(self):
        return self._f2

    def label(self):
        return self._y

    def recall(self):
        return self._recall

    def kappa(self):
        return self._kappa

    def overview(self):
        print(f"Labels: {self._y}")
        print(f"Predictions: {self._prediction}")
        print(f"Actual Values: {self._y_test}")
        print(f"F1: {self._f1}")
        print(f"Accuracy: {self._accuracy}")

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class KernalNearestNeighbour:
    def __init__(self, x, y, n_neighbors=1, weights='uniform', test_size=0.2, shuffle=True):
        """
        :param x: Dataset
        :param y: Labels
        :param n_clusters: Number of clusters
        :param weights: Weighting scheme
        :param test_size: Percentage of data used for reference.
        :param shuffle: Is the data randomised?
        """
        self._X = x
        self._y = y
        self._test_size = test_size
        self._shuffle = shuffle
        self._n_neighbors = n_neighbors
        self._weights = weights
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._X,
                                                                                    self._y,
                                                                                    test_size=self._test_size,
                                                                                    shuffle=self._shuffle)
        self._model = neighbors.KNeighborsClassifier(n_neighbors=self._n_neighbors, weights=self._weights)
        self._fit = self._model.fit(self._X_train, self._y_train)
        self._prediction = self._model.predict(self._X_test)
        self._accuracy = accuracy_score(self._y_test, self._prediction)
        self._f1 = f1_score(self._y_test, self._prediction, average='weighted')
        self._recall = recall_score(y_true=self._y_test, y_pred=self._prediction, average='weighted')
        self._precision = precision_score(y_true=self._y_test, y_pred=self._prediction, average='weighted')
        self._kappa = cohen_kappa_score(self._y_test, self._prediction)
        self._f2 = fbeta_score(self._y_test, self._prediction, beta=0.5, average='weighted')

    def predict(self):
        return self._prediction

    def actual(self):
        return self._y_test

    def accuracy(self):
        return self._accuracy

    def precision(self):
        return self._precision

    def f1(self):
        return self._f1

    def f2(self):
        return self._f2

    def label(self):
        return self._y

    def recall(self):
        return self._recall

    def kappa(self):
        return self._kappa

    def knn_overview(self):
        print(f"Labels: {self._y}")
        print(f"Predictions: {self._prediction}")
        print(f"Actual Values: {self._y_test}")
        print(f"F1: {self._f1}")
        print(f"Accuracy: {self._accuracy}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class LinearRegressionModel:
    def __init__(self, x, y, test_size=0.2, shuffle=True):
        """
        This runs the Linear Regression Machine Learning Model.
        :param x: Dataset to be tested.
        :param y: Target values of the dataset
        :param test_size: Percentage of data used for reference. 20% considered best practice.
        """
        self._X = x
        self._y = y
        self._test_size = test_size
        self._shuffle = shuffle
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._X,
                                                                                    self._y,
                                                                                    test_size=self._test_size,
                                                                                    shuffle=self._shuffle)
        self._model = LinearRegression()
        self._model.fit(self._X_train, self._y_train)
        self._prediction = self._model.predict(self._X_test)
        self._r_squared = self._model.score(self._X, self._y)
        self._coeff = self._model.coef_
        self._intercept = self._model.intercept_
        self._f1 = f1_score(self._y_test, self._prediction, average='weighted')

    def predict(self):
        return self._prediction

    def actual(self):
        return self._y_test

    def accuracy(self):
        return self._accuracy

    def precision(self):
        return self._precision

    def f1(self):
        return self._f1

    def f2(self):
        return self._f2

    def label(self):
        return self._y

    def recall(self):
        return self._recall

    def kappa(self):
        return self._kappa

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def overview(self):
        print(f"Predictions: {self._prediction}")
        print(f"Actual Values: {self._y_test}")
        print(f"R Squared: {self._r_squared}")
        print(f"Coefficient: {self._coeff}")
        print(f"F1: {self._f1}")


class KernalMeans:
    def __init__(self, x, y, n_clusters=10, test_size=0.2, shuffle=True):
        """
        :param x: Dataset
        :param y: Labels
        :param n_clusters: Number of clusters
        :param test_size: Percentage of data used for reference.
        :param shuffle: Is the data randomised?
        """
        self._X = x
        self._y = y
        self._test_size = test_size
        self._shuffle = shuffle
        self._n_clusters = n_clusters
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._X,
                                                                                    self._y,
                                                                                    test_size=self._test_size,
                                                                                    shuffle=self._shuffle)
        self._model = KMeans(n_clusters=self._n_clusters)
        self._fit = self._model.fit(self._X_train, self._y_train)
        self._prediction = self._model.predict(self._X_test)
        self._label = self._model.labels_
        self._accuracy = accuracy_score(self._y_test, self._prediction)
        self._f1 = f1_score(self._y_test, self._prediction, average='weighted')
        self._recall = recall_score(y_true=self._y_test, y_pred=self._prediction, average='weighted')
        self._precision = precision_score(y_true=self._y_test, y_pred=self._prediction, average='weighted')
        self._kappa = cohen_kappa_score(self._y_test, self._prediction)
        self._f2 = fbeta_score(self._y_test, self._prediction, beta=0.5, average='weighted')

    def predict(self):
        return self._prediction

    def actual(self):
        return self._y_test

    def accuracy(self):
        return self._accuracy

    def precision(self):
        return self._precision

    def f1(self):
        return self._f1

    def f2(self):
        return self._f2

    def label(self):
        return self._y

    def recall(self):
        return self._recall

    def kappa(self):
        return self._kappa

    def km_overview(self):
        print(f"Labels: {self._label}")
        print(f"Predictions: {self._prediction}")
        print(f"Actual Values: {self._y_test}")
        print(f"F1: {self._f1}")
        print(f"Accuracy {self._accuracy}")

    def __repr__(self) -> str:
        """Return a representation of this entity

        Returns:
              (str): A string representation of the KernalMeans class"""
        return f"{self.__class__.__name__}()"


class RForestClassifier:
    def __init__(self, x, y, n_estimators=100, test_size=0.2, shuffle=True):
        """
        :param x: Dataset.
        :param y: Labels.
        :param n_estimators: Number of trees in the forest.
        :param test_size: Percentage of data split for training and testing.
        :param shuffle: Randomization of data
        """
        self._X = x
        self._y = y
        self._test_size = test_size
        self._shuffle = shuffle
        self._n_estimators = n_estimators
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._X,
                                                                                    self._y,
                                                                                    test_size=self._test_size,
                                                                                    shuffle=self._shuffle)
        self._model = RandomForestClassifier(self._n_estimators)
        self._fit = self._model.fit(self._X_train, self._y_train)
        self._prediction = self._model.predict(self._X_test)
        self._accuracy = accuracy_score(self._y_test, self._prediction)
        self._f1 = f1_score(self._y_test, self._prediction, average='weighted')
        self._recall = recall_score(y_true=self._y_test, y_pred=self._prediction, average='weighted')
        self._precision = precision_score(y_true=self._y_test, y_pred=self._prediction, average='weighted')
        self._kappa = cohen_kappa_score(self._y_test, self._prediction)
        self._f2 = fbeta_score(self._y_test, self._prediction, beta=0.5, average='weighted')

    def predict(self):
        return self._prediction

    def actual(self):
        return self._y_test

    def accuracy(self):
        return self._accuracy

    def precision(self):
        return self._precision

    def f1(self):
        return self._f1

    def f2(self):
        return self._f2

    def label(self):
        return self._y

    def recall(self):
        return self._recall

    def kappa(self):
        return self._kappa

    def rfc_overview(self):
        print(f"Labels: {self._y}")
        print(f"Predictions: {self._prediction}")
        print(f"Actual Values: {self._y_test}")
        print(f"F1: {self._f1}")
        print(f"Accuracy {self._accuracy}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


if __name__ == '__main__':
    print("Incredibile work implementing this functions, I swear!")