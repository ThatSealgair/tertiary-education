from models import *
from data_encoding import *
import time
import pandas as pd

class Evaluation:
    def __init__(self, x, y, algorithm='Kernal Means', test_count=1, n=1, record=False, verbose=False, note='', *kwargs):
        """
        Provides ability to test the accuracy of the different machine learning algorithms

        :param x: Data
        :param y: Labels
        :param algorithm: Chosen Algorithm
        :param test_count: Number of tests run
        :param n: Number of clusters or neighbors
        :param record: boolean for saving a CSV file of results
        :param verbose: boolean for returning current values during test
        :param kwargs:

        Current Supported Algorithms
        Kernal Means 'Kernal Means'
        Linear Regression 'Linear Regression'
        """
        self._x = x
        self._y = y
        self._algorithm = algorithm
        self._test_count = test_count
        self._n = n
        self._record = record
        self._verbose = verbose
        self._note = note

    def test(self):
        sum_accuracy = 0
        sum_f1score = 0
        sum_f2score = 0
        sum_time = 0
        sum_recall = 0
        sum_precision = 0
        sum_kappa = 0

        accuracy_distance = 0
        f1_distance = 0
        f2_distance = 0
        time_distance = 0
        recall_distance = 0
        precision_distance = 0
        kappa_distance = 0

        accuracy_points = []
        f1_points = []
        f2_points = []
        times = []
        recall_points = []
        precision_points = []
        kappa_points = []

        if self._algorithm == 'Kernal Means':
            for i in range(self._test_count):
                start = time.perf_counter()
                km = KernalMeans(self._x, self._y, n_clusters=self._n)
                stop = time.perf_counter()
                time_taken = stop - start

                accuracy_value = km.accuracy()
                f1_value = km.f1()
                f2_value = km.f2()
                recall_value = km.recall()
                precision_value = km.precision()
                kappa_value = km.kappa()

                accuracy_points.append(accuracy_value)
                f1_points.append(f1_value)
                f2_points.append(f2_value)
                times.append(time_taken)
                recall_points.append(recall_value)
                precision_points.append(precision_value)
                kappa_points.append(kappa_value)

                sum_accuracy += accuracy_value
                sum_f1score += f1_value
                sum_f2score += f2_value
                sum_time += time_taken
                sum_recall += recall_value
                sum_precision += precision_value
                sum_kappa += kappa_value

                if self._verbose:
                    print(f"KM {self._note} Testing Time: {time_taken} F1: {f1_value} Accuracy {accuracy_value}")

            mean_accuracy = sum_accuracy / self._test_count
            mean_f1 = sum_f1score / self._test_count
            mean_f2 = sum_f2score / self._test_count
            mean_time = sum_time / self._test_count
            mean_recall = sum_recall / self._test_count
            mean_precision = sum_precision / self._test_count
            mean_kappa = sum_kappa / self._test_count

            for i in range(self._test_count):
                accuracy_distance += (abs(accuracy_points[i] - mean_accuracy)) ** 2
                f1_distance += (abs(f1_points[i] - mean_f1)) ** 2
                f2_distance += (abs(f2_points[i] - mean_f2)) ** 2
                time_distance += (abs(times[i] - mean_time)) ** 2
                recall_distance += (abs(recall_points[i] - mean_recall)) ** 2
                precision_distance += (abs(precision_points[i] - mean_precision)) ** 2
                kappa_distance += (abs(kappa_points[i] - mean_kappa)) ** 2

            accuracy_standard_deviation = (accuracy_distance / self._test_count) ** 0.5
            f1_standard_deviation = (f1_distance / self._test_count) ** 0.5
            f2_standard_deviation = (f2_distance / self._test_count) ** 0.5
            time_standard_deviation = (time_distance / self._test_count) ** 0.5
            recall_standard_deviation = (recall_distance / self._test_count) ** 0.5
            precision_standard_deviation = (precision_distance / self._test_count) ** 0.5
            kappa_standard_deviation = (kappa_distance / self._test_count) ** 0.5

            if self._record:
                listdata = {'Time': times,
                            'F1': f1_points,
                            'F2': f2_points,
                            'Accuracy': accuracy_points,
                            'Recall': recall_points,
                            'Precision': precision_points,
                            'Kappa': kappa_points,
                            }
                list_data_frame = pd.DataFrame(listdata, columns=['Time',
                                                                  'F1',
                                                                  'F2',
                                                                  'Accuracy',
                                                                  'Recall',
                                                                  'Precision',
                                                                  'Kappa',
                                                                  ])
                csvdata = f'KM_List{self._note}' + str(abs(hash((mean_accuracy * mean_time * mean_time)))) + '.csv'
                list_data_frame.to_csv(rf'data/{csvdata}')

                data_totals = {'Cluster Count': [self._n],
                               'Sum Time': [sum_time],
                               'Mean Time': [mean_time],
                               'SD Time': [time_standard_deviation],
                               'Sum F1 Score': [sum_f1score],
                               'Mean F1 Score': [mean_f1],
                               'SD F1 Score': [f1_standard_deviation],
                               'Sum F2 Score': [sum_f2score],
                               'Mean F2 Score': [mean_f2],
                               'SD F2 Score': [f2_standard_deviation],
                               'Sum Accuracy': [sum_accuracy],
                               'Mean Accuracy': [mean_accuracy],
                               'SD Accuracy': [accuracy_standard_deviation],
                               'Sum Recall': [sum_recall],
                               'Mean Recall': [mean_recall],
                               'SD Recall': [recall_standard_deviation],
                               'Sum Precision': [sum_precision],
                               'Mean Precision': [mean_precision],
                               'SD Precision': [precision_standard_deviation],
                               'Sum Kappa Score': [sum_kappa],
                               'Mean Kappa Score': [mean_kappa],
                               'SD Kappa Score': [kappa_standard_deviation],
                               }

                data_totals_frame = pd.DataFrame(data_totals, columns=['Cluster Count',
                                                                       'Sum Time',
                                                                       'Mean Time',
                                                                       'SD Time',
                                                                       'Sum F1 Score',
                                                                       'Mean F1 Score',
                                                                       'SD F1 Score',
                                                                       'Sum F2 Score',
                                                                       'Mean F2 Score',
                                                                       'SD F2 Score',
                                                                       'Sum Accuracy',
                                                                       'Mean Accuracy',
                                                                       'SD Accuracy',
                                                                       'Sum Recall',
                                                                       'Mean Recall',
                                                                       'SD Recall',
                                                                       'Sum Precision',
                                                                       'Mean Precision',
                                                                       'SD Precision',
                                                                       'Sum Kappa Score',
                                                                       'Mean Kappa Score',
                                                                       'SD Kappa Score',
                                                                       ])
                csvtitle = f'KM{self._note}' + str(abs(hash((mean_accuracy * mean_time * mean_time)))) + '.csv'
                data_totals_frame.to_csv(rf'data/{csvtitle}')

            print(f"Mean accuracy after {self._test_count} counts for {self._n} clusters is: {mean_accuracy}. With a "
                  f"Standard Deviation of: {accuracy_standard_deviation}.")
            print(f"Mean F1 score after {self._test_count} counts for {self._n} clusters is: {mean_f1}. With a "
                  f"Standard Deviation of: {f1_standard_deviation}.")
            print(f"Mean time after {self._test_count} counts for {self._n} clusters is: {mean_time}. With a Standard "
                  f"Deviation of: {time_standard_deviation}.")

        elif self._algorithm == 'Support Vector Machine':
            for i in range(self._test_count):
                start = time.perf_counter()
                sva = SupportVectorAlgorithm(self._x, self._y)
                stop = time.perf_counter()
                time_taken = stop - start

                accuracy_value = sva.accuracy()
                f1_value = sva.f1()
                f2_value = sva.f2()
                recall_value = sva.recall()
                precision_value = sva.precision()
                kappa_value = sva.kappa()

                accuracy_points.append(accuracy_value)
                f1_points.append(f1_value)
                f2_points.append(f2_value)
                times.append(time_taken)
                recall_points.append(recall_value)
                precision_points.append(precision_value)
                kappa_points.append(kappa_value)

                sum_accuracy += accuracy_value
                sum_f1score += f1_value
                sum_f2score += f2_value
                sum_time += time_taken
                sum_recall += recall_value
                sum_precision += precision_value
                sum_kappa += kappa_value


                if self._verbose:
                    print(f"SVM {self._note} Testing Time: {time_taken} F1: {f1_value} Accuracy {accuracy_value}")

            mean_accuracy = sum_accuracy / self._test_count
            mean_f1 = sum_f1score / self._test_count
            mean_f2 = sum_f2score / self._test_count
            mean_time = sum_time / self._test_count
            mean_recall = sum_recall / self._test_count
            mean_precision = sum_precision / self._test_count
            mean_kappa = sum_kappa / self._test_count

            for i in range(self._test_count):
                accuracy_distance += (abs(accuracy_points[i] - mean_accuracy)) ** 2
                f1_distance += (abs(f1_points[i] - mean_f1)) ** 2
                f2_distance += (abs(f2_points[i] - mean_f1)) ** 2
                time_distance += (abs(times[i] - mean_time)) ** 2
                recall_distance += (abs(recall_points[i] - mean_recall)) ** 2
                precision_distance += (abs(precision_points[i] - mean_precision)) ** 2
                kappa_distance += (abs(kappa_points[i] - mean_kappa)) ** 2

            accuracy_standard_deviation = (accuracy_distance / self._test_count) ** 0.5
            f1_standard_deviation = (f1_distance / self._test_count) ** 0.5
            f2_standard_deviation = (f2_distance / self._test_count) ** 0.5
            time_standard_deviation = (time_distance / self._test_count) ** 0.5
            recall_standard_deviation = (recall_distance / self._test_count) ** 0.5
            precision_standard_deviation = (precision_distance / self._test_count) ** 0.5
            kappa_standard_deviation = (kappa_distance / self._test_count) ** 0.5

            if self._record:
                listdata = {'Time': times,
                            'F1': f1_points,
                            'F2': f2_points,
                            'Accuracy': accuracy_points,
                            'Recall': recall_points,
                            'Precision': precision_points,
                            'Kappa': kappa_points,
                            }
                list_data_frame = pd.DataFrame(listdata, columns=['Time',
                                                                  'F1',
                                                                  'F2',
                                                                  'Accuracy',
                                                                  'Recall',
                                                                  'Precision',
                                                                  'Kappa',
                                                                  ])
                csvdata = f'SVM_List{self._note}' + str(abs(hash((mean_accuracy * mean_time * mean_time)))) + '.csv'
                list_data_frame.to_csv(rf'data/{csvdata}')

                data_totals = {'Cluster Count': [self._n],
                               'Sum Time': [sum_time],
                               'Mean Time': [mean_time],
                               'SD Time': [time_standard_deviation],
                               'Sum F1 Score': [sum_f1score],
                               'Mean F1 Score': [mean_f1],
                               'SD F1 Score': [f1_standard_deviation],
                               'Sum F2 Score': [sum_f2score],
                               'Mean F2 Score': [mean_f2],
                               'SD F2 Score': [f2_standard_deviation],
                               'Sum Accuracy': [sum_accuracy],
                               'Mean Accuracy': [mean_accuracy],
                               'SD Accuracy': [accuracy_standard_deviation],
                               'Sum Recall': [sum_recall],
                               'Mean Recall': [mean_recall],
                               'SD Recall': [recall_standard_deviation],
                               'Sum Precision': [sum_precision],
                               'Mean Precision': [mean_precision],
                               'SD Precision': [precision_standard_deviation],
                               'Sum Kappa Score': [sum_kappa],
                               'SD Kappa Score': [kappa_standard_deviation],
                               'Mean Kappa Score': [mean_kappa],
                               }

                data_totals_frame = pd.DataFrame(data_totals, columns=['Cluster Count',
                                                                       'Sum Time',
                                                                       'Mean Time',
                                                                       'SD Time',
                                                                       'Sum F1 Score',
                                                                       'Mean F1 Score',
                                                                       'SD F1 Score',
                                                                       'Sum F2 Score',
                                                                       'Mean F2 Score',
                                                                       'SD F2 Score',
                                                                       'Sum Accuracy',
                                                                       'Mean Accuracy',
                                                                       'SD Accuracy',
                                                                       'Sum Recall',
                                                                       'Mean Recall',
                                                                       'SD Recall',
                                                                       'Sum Precision',
                                                                       'Mean Precision',
                                                                       'SD Precision',
                                                                       'Sum Kappa Score',
                                                                       'Mean Kappa Score',
                                                                       'SD Kappa Score',
                                                                       ])
                csvtitle = f'SVM{self._note}' + str(abs(hash((mean_accuracy * mean_time * mean_time)))) + '.csv'
                data_totals_frame.to_csv(rf'data/{csvtitle}')

            print(f"Mean accuracy after {self._test_count} counts is: {mean_accuracy}. With a "
                  f"Standard Deviation of: {accuracy_standard_deviation}.")
            print(f"Mean F1 score after {self._test_count} counts is: {mean_f1}. With a "
                  f"Standard Deviation of: {f1_standard_deviation}.")
            print(f"Mean time after {self._test_count} counts is: {mean_time}. With a Standard "
                  f"Deviation of: {time_standard_deviation}.")

        elif self._algorithm == 'Kernal Nearest Neighbor':
            for i in range(self._test_count):
                start = time.perf_counter()
                knn = KernalNearestNeighbour(self._x, self._y, n_neighbors=self._n)
                stop = time.perf_counter()
                time_taken = stop - start

                accuracy_value = knn.accuracy()
                f1_value = knn.f1()
                f2_value = knn.f2()
                recall_value = knn.recall()
                precision_value = knn.precision()
                kappa_value = knn.kappa()

                accuracy_points.append(accuracy_value)
                f1_points.append(f1_value)
                f2_points.append(f2_value)
                times.append(time_taken)
                recall_points.append(recall_value)
                precision_points.append(precision_value)
                kappa_points.append(kappa_value)

                sum_accuracy += accuracy_value
                sum_f1score += f1_value
                sum_f2score += f2_value
                sum_time += time_taken
                sum_recall += recall_value
                sum_precision += precision_value
                sum_kappa += kappa_value

                if self._verbose:
                    print(f"KNN {self._note} Testing Time: {time_taken} F1: {f1_value} Accuracy {accuracy_value}")

            mean_accuracy = sum_accuracy / self._test_count
            mean_f1 = sum_f1score / self._test_count
            mean_f2 = sum_f2score / self._test_count
            mean_time = sum_time / self._test_count
            mean_recall = sum_recall / self._test_count
            mean_precision = sum_precision / self._test_count
            mean_kappa = sum_kappa / self._test_count

            for i in range(self._test_count):
                accuracy_distance += (abs(accuracy_points[i] - mean_accuracy)) ** 2
                f1_distance += (abs(f1_points[i] - mean_f1)) ** 2
                f2_distance += (abs(f2_points[i] - mean_f2)) ** 2
                time_distance += (abs(times[i] - mean_time)) ** 2
                recall_distance += (abs(recall_points[i] - mean_time)) ** 2
                precision_distance += (abs(precision_points[i] - mean_precision)) ** 2
                kappa_distance += (abs(kappa_points[i] - mean_kappa)) ** 2

            accuracy_standard_deviation = (accuracy_distance / self._test_count) ** 0.5
            f1_standard_deviation = (f1_distance / self._test_count) ** 0.5
            f2_standard_deviation = (f2_distance / self._test_count) ** 0.5
            time_standard_deviation = (time_distance / self._test_count) ** 0.5
            recall_standard_deviation = (recall_distance / self._test_count) ** 0.5
            precision_standard_deviation = (precision_distance / self._test_count) ** 0.5
            kappa_standard_deviation = (kappa_distance / self._test_count) ** 0.5

            if self._record:
                listdata = {'Time': times,
                            'F1': f1_points,
                            'F2': f2_points,
                            'Accuracy': accuracy_points,
                            'Recall': recall_points,
                            'Precision': precision_points,
                            'Kappa': kappa_points,
                            }
                list_data_frame = pd.DataFrame(listdata, columns=['Time',
                                                                  'F1',
                                                                  'F2',
                                                                  'Accuracy',
                                                                  'Recall',
                                                                  'Precision',
                                                                  'Kappa',
                                                                  ])
                csvdata = f'KNN_List{self._note}' + str(abs(hash((mean_accuracy * mean_time * mean_time)))) + '.csv'
                list_data_frame.to_csv(rf'data/{csvdata}')

                data_totals = {'Cluster Count': [self._n],
                               'Sum Time': [sum_time],
                               'Mean Time': [mean_time],
                               'SD Time': [time_standard_deviation],
                               'Sum F1 Score': [sum_f1score],
                               'Mean F1 Score': [mean_f1],
                               'SD F1 Score': [f1_standard_deviation],
                               'Sum F2 Score': [sum_f2score],
                               'Mean F2 Score': [mean_f2],
                               'SD F2 Score': [f2_standard_deviation],
                               'Sum Accuracy': [sum_accuracy],
                               'Mean Accuracy': [mean_accuracy],
                               'SD Accuracy': [accuracy_standard_deviation],
                               'Sum Recall': [sum_recall],
                               'Mean Recall': [mean_recall],
                               'SD Recall': [recall_standard_deviation],
                               'Sum Precision': [sum_precision],
                               'Mean Precision': [mean_precision],
                               'SD Precision': [precision_standard_deviation],
                               'Sum Kappa Score': [sum_kappa],
                               'Mean Kappa Score': [mean_kappa],
                               'SD Kappa Score': [kappa_standard_deviation],
                               }

                data_totals_frame = pd.DataFrame(data_totals, columns=['Cluster Count',
                                                                       'Sum Time',
                                                                       'Mean Time',
                                                                       'SD Time',
                                                                       'Sum F1 Score',
                                                                       'Mean F1 Score',
                                                                       'SD F1 Score',
                                                                       'Sum F2 Score',
                                                                       'Mean F2 Score',
                                                                       'SD F2 Score',
                                                                       'Sum Accuracy',
                                                                       'Mean Accuracy',
                                                                       'SD Accuracy',
                                                                       'Sum Recall',
                                                                       'Mean Recall',
                                                                       'SD Recall',
                                                                       'Sum Precision',
                                                                       'Mean Precision',
                                                                       'SD Precision',
                                                                       'Sum Kappa Score',
                                                                       'Mean Kappa Score',
                                                                       'SD Kappa Score',
                                                                       ])
                csvtitle = f'KNN{self._note}' + str(abs(hash((mean_accuracy * mean_time * mean_time)))) + '.csv'
                data_totals_frame.to_csv(rf'data/{csvtitle}')

            print(f"Mean accuracy after {self._test_count} counts for {self._n} clusters is: {mean_accuracy}. With a "
                  f"Standard Deviation of: {accuracy_standard_deviation}.")
            print(f"Mean F1 score after {self._test_count} counts for {self._n} clusters is : {mean_f1}. With a "
                  f"Standard Deviation of: {f1_standard_deviation}")
            print(f"Mean time after {self._test_count} counts for {self._n} clusters is: {mean_time}. With a Standard "
                  f"Deviation of: {time_standard_deviation}.")

        elif self._algorithm == 'Random Forest':
            for i in range(self._test_count):
                start = time.perf_counter()
                rfc = RForestClassifier(self._x, self._y)
                stop = time.perf_counter()
                time_taken = stop - start

                accuracy_value = rfc.accuracy()
                f1_value = rfc.f1()
                f2_value = rfc.f2()
                recall_value = rfc.recall()
                precision_value = rfc.precision()
                kappa_value = rfc.kappa()

                accuracy_points.append(accuracy_value)
                f1_points.append(f1_value)
                f2_points.append(f2_value)
                times.append(time_taken)
                recall_points.append(recall_value)
                precision_points.append(precision_value)
                kappa_points.append(kappa_value)

                sum_accuracy += accuracy_value
                sum_f1score += f1_value
                sum_f2score += f2_value
                sum_time += time_taken
                sum_recall += recall_value
                sum_precision += precision_value
                sum_kappa += kappa_value


                if self._verbose:
                    print(f"RF {self._note} Testing Time: {time_taken} F1: {f1_value} Accuracy {accuracy_value}")

            mean_accuracy = sum_accuracy / self._test_count
            mean_f1 = sum_f1score / self._test_count
            mean_f2 = sum_f2score / self._test_count
            mean_time = sum_time / self._test_count
            mean_recall = sum_recall / self._test_count
            mean_precision = sum_precision / self._test_count
            mean_kappa = sum_kappa / self._test_count

            for i in range(self._test_count):
                accuracy_distance += (abs(accuracy_points[i] - mean_accuracy)) ** 2
                f1_distance += (abs(f1_points[i] - mean_f1)) ** 2
                f2_distance += (abs(f2_points[i] - mean_f2)) ** 2
                time_distance += (abs(times[i] - mean_time)) ** 2
                recall_distance += (abs(recall_points[i] - mean_recall)) ** 2
                precision_distance += (abs(precision_points[i] - mean_precision)) ** 2
                kappa_distance += (abs(kappa_points[i] - mean_kappa)) ** 2

            accuracy_standard_deviation = (accuracy_distance / self._test_count) ** 0.5
            f1_standard_deviation = (f1_distance / self._test_count) ** 0.5
            f2_standard_deviation = (f2_distance / self._test_count) ** 0.5
            time_standard_deviation = (time_distance / self._test_count) ** 0.5
            recall_standard_deviation = (recall_distance / self._test_count) ** 0.5
            precision_standard_deviation = (precision_distance / self._test_count) ** 0.5
            kappa_standard_deviation = (kappa_distance / self._test_count) ** 0.5

            if self._record:
                listdata = {'Time': times,
                            'F1': f1_points,
                            'F2': f2_points,
                            'Accuracy': accuracy_points,
                            'Recall': recall_points,
                            'Precision': precision_points,
                            'Kappa': kappa_points,
                            }
                list_data_frame = pd.DataFrame(listdata, columns=['Time',
                                                                  'F1',
                                                                  'F2',
                                                                  'Accuracy',
                                                                  'Recall',
                                                                  'Precision',
                                                                  'Kappa',
                                                                  ])
                csvdata = f'RFC_List{self._note}' + str(abs(hash((mean_accuracy * mean_time * mean_time)))) + '.csv'
                list_data_frame.to_csv(rf'data/{csvdata}')

                data_totals = {'Cluster Count': [self._n],
                               'Sum Time': [sum_time],
                               'Mean Time': [mean_time],
                               'SD Time': [time_standard_deviation],
                               'Sum F1 Score': [sum_f1score],
                               'Mean F1 Score': [mean_f1],
                               'SD F1 Score': [f1_standard_deviation],
                               'Sum F2 Score': [sum_f2score],
                               'Mean F2 Score': [mean_f2],
                               'SD F2 Score': [f2_standard_deviation],
                               'Sum Accuracy': [sum_accuracy],
                               'Mean Accuracy': [mean_accuracy],
                               'SD Accuracy': [accuracy_standard_deviation],
                               'Sum Recall': [sum_recall],
                               'Mean Recall': [mean_recall],
                               'SD Recall': [recall_standard_deviation],
                               'Sum Precision': [sum_precision],
                               'Mean Precision': [mean_precision],
                               'SD Precision': [precision_standard_deviation],
                               'Sum Kappa Score': [sum_kappa],
                               'Mean Kappa Score': [mean_kappa],
                               'SD Kappa Score': [kappa_standard_deviation],
                               }

                data_totals_frame = pd.DataFrame(data_totals, columns=['Cluster Count',
                                                                       'Sum Time',
                                                                       'Mean Time',
                                                                       'SD Time',
                                                                       'Sum F1 Score',
                                                                       'Mean F1 Score',
                                                                       'SD F1 Score',
                                                                       'Sum F2 Score',
                                                                       'Mean F2 Score',
                                                                       'SD F2 Score',
                                                                       'Sum Accuracy',
                                                                       'Mean Accuracy',
                                                                       'SD Accuracy',
                                                                       'Sum Recall',
                                                                       'Mean Recall',
                                                                       'SD Recall',
                                                                       'Sum Precision',
                                                                       'Mean Precision',
                                                                       'SD Precision',
                                                                       'Sum Kappa Score',
                                                                       'Mean Kappa Score',
                                                                       'SD Kappa Score',
                                                                       ])
                csvtitle = f'RFC{self._note}' + str(abs(hash((mean_accuracy * mean_time * mean_time)))) + '.csv'
                data_totals_frame.to_csv(rf'data/{csvtitle}')

            print(f"Mean accuracy after {self._test_count} counts is: {mean_accuracy}. With a "
                  f"Standard Deviation of: {accuracy_standard_deviation}.")
            print(f"Mean F1 score after {self._test_count} counts is: {mean_f1}. With a "
                  f"Standard Deviation of: {f1_standard_deviation}.")
            print(f"Mean time after {self._test_count} counts is: {mean_time}. With a Standard "
                  f"Deviation of: {time_standard_deviation}.")

        elif self._algorithm == "Linear Regression":
            for i in range(self._test_count):
                start = time.perf_counter()
                lr = LinearRegressionModel(self._x, self._y)
                stop = time.perf_counter()
                time_taken = stop - start

                f1_value = lr.f1()

                f1_points.append(f1_value)
                times.append(time_taken)

                if self._verbose:
                    print(f"Testing Time: {time_taken} F1: {f1_value}")

            mean_f1 = sum_f1score / self._test_count
            mean_time = sum_time / self._test_count

            for i in range(self._test_count):
                f1_distance += (abs(f1_points[i] - mean_f1)) ** 2
                time_distance += (abs(times[i] - mean_time)) ** 2

            f1_standard_deviation = (f1_distance / self._test_count) ** 0.5
            time_standard_deviation = (time_distance / self._test_count) ** 0.5

            if self._record:
                data = {'Sum Time': [sum_time],
                        'Mean Time': [mean_time],
                        'SD Time': [time_standard_deviation],
                        'Sum F1 Score': [sum_f1score],
                        'Mean F1 Score': [mean_f1],
                        'SD F1 Score': [f1_standard_deviation]}

                data_frame = pd.DataFrame(data, columns=['Cluster Count',
                                                         'Sum Time',
                                                         'Mean Time',
                                                         'SD Time',
                                                         'Sum F1 Score',
                                                         'Mean F1 Score',
                                                         'SD F1 Score'])
                csvtitle = 'LR' + str(abs(hash((f1_standard_deviation * mean_time * mean_time)))) + '.csv'
                data_frame.to_csv(rf'data/{csvtitle}')

            print(f"Mean F1 score after {self._test_count} counts is: {mean_f1}. With a Standard Deviation "
                  f"of {f1_standard_deviation}")
            print(f"Mean time after {self._test_count} counts is: {mean_time}. With a Standard Deviation "
                  f"of {time_standard_deviation}")

        else:
            return f"You need to select a algorithm"

    def __repr__(self):
        return f"{self.__class__.__name__}()"


if __name__ == '__main__':
    print("Mama Mia! That's a lot of speghetti!")