import numpy as np
import pandas as pd
import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class Evaluator(object):
    def __init__(self):
        self._accuracy_scores = []
        self._precision_scores = []
        self._recall_scores = []
        self._f1_scores = []
        self._training_times = []

    # def __repr__():
    #     template = "\n".join([
    #         "+------------+-----------------------------------------+",
    #         "|   METRIC   |        MEAN        | STANDARD DEVIATION |",
    #         "+============+====================+====================+",
    #         "| ACCURACY   | {accuracy_mean}    | {accuracy_stddev}  |",
    #         "+------------+--------------------+--------------------+",
    #         "| PRECISION  | {precision_mean}   | {precision_stddev} |",
    #         "+------------+--------------------+--------------------+",
    #         "|   RECALL   | {recall_mean}      | {recall_stddev}    |",
    #         "+------------+--------------------+--------------------+",
    #         "|     F1     | {f1_mean}          | {f1_stddev}        |",
    #         "+------------+--------------------+--------------------+"
    #     ])

        # return template.format()

    def _update_accuracy_score(self, Y_true, Y_pred, normalize):
        self._accuracy_scores.append(accuracy_score(Y_true,
                                                    Y_pred,
                                                    normalize = normalize))
        print("ACCURACY:  ", self._accuracy_scores[-1])


    def _update_precision_score(self, Y_true, Y_pred, average):
        self._precision_scores.append(precision_score(Y_true,
                                                      Y_pred,
                                                      average = average))
        print("PRECISION: ", self._precision_scores[-1])


    def _update_recall_score(self, Y_true, Y_pred, average):
        self._recall_scores.append(recall_score(Y_true,
                                                Y_pred,
                                                average = average))
        print("RECALL:    ", self._recall_scores[-1])


    def _update_f1_score(self, Y_true, Y_pred, average):
        self._f1_scores.append(f1_score(Y_true,
                                        Y_pred,
                                        average = average))
        print("F1:        ", self._f1_scores[-1])


    def begin(self):
        self._training_times = []
        self._previous_time = self._begin_time = datetime.datetime.now()

    def finish(self):
        self._finish_time = datetime.datetime.now()

        self._accuracy_mean = np.mean(self._accuracy_scores)
        self._precision_mean = np.mean(self._precision_scores)
        self._recall_mean = np.mean(self._recall_scores)
        self._f1_mean = np.mean(self._f1_scores)

        self._accuracy_std = np.std(self._accuracy_scores)
        self._precision_std = np.std(self._precision_scores)
        self._recall_std = np.std(self._recall_scores)
        self._f1_std = np.std(self._f1_scores)


    def evaluate(self, Y_true, Y_pred, average = "weighted", normalize = True):
        current_time = datetime.datetime.now()
        self._training_times.append(current_time - self._previous_time)
        print("TIME:      ", self._training_times[-1])
        self._previous_time = current_time
        self._update_accuracy_score(Y_true, Y_pred, normalize)
        self._update_precision_score(Y_true, Y_pred, "weighted")
        self._update_recall_score(Y_true, Y_pred, "weighted")
        self._update_f1_score(Y_true, Y_pred, "weighted")
