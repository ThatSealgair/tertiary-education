import pandas as pd
import scipy as sp


class DataEncoding:
    def __init__(self, data, target: str, supervised=False, feature_selection=None):
        """
        This encodes packet data in a CSV file parsed through so that it runs correctly with Scikit Learn, Numpy, and Pandas.

        :param data: The dataset
        :param target: The target of the dataset
        :param supervised: Boolean choice whether labels are included in the data
        :param feature_selection: List or string of features to be tested.
        """
        self._data = data
        self._target = target
        self._supervised = supervised
        self._feature_selection = feature_selection

        """Algorithm to encode all values within the dataset to float values which Scikit Learn can use."""
        for col in self._data:
            enum = {}

            for crypt, value in enumerate(self._data[col].unique()):
                enum[value] = float(crypt)

            self._data[col].replace(enum, inplace=True)

        """Seperates the target data from the testing data."""

        if self._feature_selection is not None and self._supervised:
            self._X = self._data[self._feature_selection, self._target]
            self._y = self._data[self._target]
        elif self._feature_selection is not None and self._supervised is False:
            self._X = self._data[self._feature_selection]
            self._y = self._data[self._target]
        elif self._feature_selection is None and self._supervised:
            self._X = self._data
            self._y = self._data[self._target]
        elif self._feature_selection is None and self._supervised is False:
            self._X = self._data.drop(labels=self._target, axis=1)
            self._y = self._data[self._target]
        else:
            print(f"Slection feature:{self._feature_selection} and target {self._target} has run into an issue. \n"
                  f"Please keep combination and ammend this.")

    def X(self):
        """
        :return: Returns encoded test data
        """
        return self._X

    def y(self):
        """
        :return: Returns encoded target data
        """
        return self._y

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Entropy_Calculator:
    def __init__(self, data: str, window: float, offset: float):
        """
        This class calculates the entropy of the data.

        :param data: The dataset from a CSV file
        :param window: The window size
        :param offset: The offset size
        """
        self._data = pd.read_csv(data)
        self._window = window
        self._offset = offset
        
        self._columns = ['timestamp', 'smac', 'dmac', 'sip', 'dip', 'request', 'fc', 'error', 'address', 'info', 'elabel', 'label']

        prev_digit = 0

        """ Loops through unique timestamp values and calculates the entropy of each window."""
        for timestamp in self.data['Time'].unique():
            if str(timestamp)[0] != prev_digit:
                print(timestamp)
                prev_digit = str(timestamp)[0]

            """ Filters results into window."""
            temp_partition = self.data[(self.data['Time'] <= timestamp + window + window * offset)]
            partition = temp_partition[(temp_partition['Time'] >= timestamp - window - window * offset)]

            entropy_arr = []
            for datacol in partition.columns:
                data_entropy = sp.stats.entropy(partition[datacol].value_counts())
                entropy_arr.append(data_entropy)
            entropy_arr[0] = timestamp

            all_labels = self.data[self.data['Time'] == timestamp]['label'].values
            if len(all_labels) == 1:
                entropy_arr.append(all_labels[0])
            elif list(all_labels).count('FORCE_ERROR_ATTACK') != 0:
                entropy_arr.append('FORCE_ERROR_ATTACK')
            elif list(all_labels).count('REPLAY_ATTACK') != 0:
                entropy_arr.append('REPLAY_ATTACK')
            elif list(all_labels).count('RECOGNITION_ATTACK') != 0:
                entropy_arr.append('RECOGNITION_ATTACK')
            elif list(all_labels).count('MITM_UNALTERED') != 0:
                entropy_arr.append('MITM_UNALTERED')
            elif list(all_labels).count('READ_ATTACK') != 0:
                entropy_arr.append('READ_ATTACK')
            elif list(all_labels).count('WRITE_ATTACK') != 0:
                entropy_arr.append('WRITE_ATTACK')
            elif list(all_labels).count('RESPONSE_ATTACK') != 0:
                entropy_arr.append('RESPONSE_ATTACK')
            else:
                entropy_arr.append('NORMAL')

            single_df = pd.DataFrame([entropy_arr], columns=self._columns)
            target = target.append(single_df, ignore_index=True)
        
        export = pd.DataFrame(columns=self._columns)
        export.to_csv(
            './EntropyCalculated/' + 'window_' + str(int(window)) + "_" + str(offset) + '_' + source.replace('/', '~'),
            index=False)


if __name__ == '__main__':
    print("Data Encoding? What is this?")