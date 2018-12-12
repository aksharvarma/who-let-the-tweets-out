import numpy as np
import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class Evaluator(object):
    def __init__(self, n_splits = 5):
        self._scores = np.zeros((n_splits, 4), dtype=np.float64)
        self._n_splits = n_splits
        self._split_index = 0

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

    def begin(self):
        self._previous_time = self._begin_time = datetime.datetime.now()

    def finish(self):
        self._finish_time = datetime.datetime.now()

    def format_score(self, scores):
        return " ".join(map(lambda score: "{0:.2f}".format(round(score,2)),
                            scores))

    def evaluate(self, Y_true, Y_pred, average = "weighted", normalize = True):
        current_time = datetime.datetime.now()
        training_time = current_time - self._previous_time
        self._previous_time = current_time
        self._scores[self._split_index, 0] = accuracy_score(Y_true,
                                                            Y_pred,
                                                            normalize = normalize)
        self._scores[self._split_index, 1] = precision_score(Y_true,
                                                             Y_pred,
                                                             average = "weighted")
        self._scores[self._split_index, 2] = recall_score(Y_true,
                                                          Y_pred,
                                                          average = "weighted")
        self._scores[self._split_index, 3] = f1_score(Y_true,
                                                      Y_pred,
                                                      average = "weighted")

        print(self.format_score(self._scores[self._split_index,:]), end = " ")
        print(training_time.total_seconds(), end = " Seconds\n")
        self._split_index += 1

    def save(self, raw_score_filename, aggregated_score_filename):
        np.savetxt(raw_score_filename,
                   self._scores,
                   comments = "",
                   delimiter = ",",
                   header = "Accuracy, Precision, Recall, F1")
        aggregated_score = np.array([
            np.mean(self._scores, axis = 0),
            np.std(self._scores, axis = 0)
        ])
        np.savetxt(aggregated_score_filename,
                   aggregated_score,
                   comments = "",
                   delimiter = ",",
                   header = "Accuracy, Precision, Recall, F1")
        print(self.format_score(aggregated_score[0]), end = " \n")
        print(self.format_score(aggregated_score[1]), end = " \n")
