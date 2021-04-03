from pandas import *


def confusion_matrix(y_true, y_pred):
    possible_val = []
    for val in y_true:
        if val not in possible_val:
            possible_val.append(val)

    for val in y_pred:
        if val not in possible_val:
            possible_val.append(val)

    dic = {}
    possible_val.sort()
    for i in range(len(possible_val)):
        dic[possible_val[i]] = i

    mat = [[0 for j in range(len(possible_val))]
           for i in range(len(possible_val))]

    for i in range(len(y_true)):
        mat[dic[y_true[i]]][dic[y_pred[i]]] += 1

    return mat


class Metrics:
    def __init__(self, y_true, y_pred):
        # self.y_true = y_true
        # self.y_pred = y_pred
        self.confusion_matrix = confusion_matrix(y_true, y_pred)
        self.labels = []
        for val in y_true:
            if val not in self.labels:
                self.labels.append(val)

        for val in y_pred:
            if val not in self.labels:
                self.labels.append(val)

        self.labels.sort()

    def get_label_tp_tn_fp_fn(self, label):
        tp = self.confusion_matrix[label][label]
        tn, fp, fn = 0, 0, 0
        n = len(self.confusion_matrix)
        for i in range(n):
            for j in range(n):
                val = self.confusion_matrix[i][j]
                if (i != label and j != label):
                    tn += val
                if (j == label and i != label):
                    fp += val
                if (i == label and j != label):
                    fn += val
        return tp, tn, fp, fn

    def overall_accuracy(self):
        tp = 0
        tot = 0
        n = len(self.confusion_matrix)
        for i in range(n):
            tp += self.confusion_matrix[i][i]
            for j in range(n):
                tot += self.confusion_matrix[i][j]
        return tp / tot

    # label yg dimaksud itu indeks labelnya
    # label = ["a", "b", "c"]
    # kalo mau accuracy "a" manggil self.accuracy(0)
    def accuracy(self, label):
        tp, tn, fp, fn = self.get_label_tp_tn_fp_fn(label)
        if(tp + tn == 0):
            return 0
        return (tp + tn) / (tp + tn + fn + fp)

    def all_accuracy(self):
        n = len(self.confusion_matrix)
        tot = 0
        for i in range(n):
            tot += self.accuracy(i)

        if(tot == 0):
            return 0
        return tot / n

    def precision(self, label):
        tp, tn, fp, fn = self.get_label_tp_tn_fp_fn(label)
        if(tp == 0):
            return 0
        return tp / (tp + fp)

    def all_precision(self):
        n = len(self.confusion_matrix)
        tot = 0
        for i in range(n):
            tot += self.precision(i)
        if(tot == 0):
            return 0
        return tot / n

    def recall(self, label):
        tp, tn, fp, fn = self.get_label_tp_tn_fp_fn(label)
        if(tp == 0):
            return 0
        return tp / (tp + fn)

    def all_recall(self):
        n = len(self.confusion_matrix)
        tot = 0
        for i in range(n):
            tot += self.recall(i)
        if(tot == 0):
            return 0
        return tot / n

    def f1_score(self, label):
        tp, tn, fp, fn = self.get_label_tp_tn_fp_fn(label)
        precision = self.precision(label)
        recall = self.recall(label)
        if(precision * recall == 0):
            return 0
        return 2 * precision * recall / (precision + recall)

    def all_f1_score(self):
        n = len(self.confusion_matrix)
        tot = 0
        for i in range(n):
            tot += self.f1_score(i)
        if(tot == 0):
            return 0
        return tot / n

    def report(self, digits=3):
        # accuracy, precision, recall, f1
        n = len(self.confusion_matrix)
        ret = [[0 for j in range(4)] for i in range(n)]
        for i in range(n):
            ret[i][0] = round(self.accuracy(i), digits)
            ret[i][1] = round(self.precision(i), digits)
            ret[i][2] = round(self.recall(i), digits)
            ret[i][3] = round(self.f1_score(i), digits)

        print(prettify(ret, self.labels, [
              "accuracy", "precision", "recall", "f1"]))
        print(f'overall accuracy: {self.all_accuracy():.3f}')
        print(f'overall precision: {self.all_precision():.3f}')
        print(f'overall recall: {self.all_recall():.3f}')
        print(f'overall f1_score: {self.all_f1_score():.3f}')


def prettify(data, index, columns):
    ret = DataFrame(data)
    ret.index = index
    ret.columns = columns
    return ret


if __name__ == "__main__":
    # Constants
    C = "Cat"
    F = "Fish"
    H = "Hen"

    # True values
    y_true = [C, C, C, C, C, C, F, F, F, F, F,
              F, F, F, F, F, H, H, H, H, H, H, H, H, H]
    # Predicted values
    y_pred = [C, C, C, C, H, F, C, C, C, C, C,
              C, H, H, F, F, C, C, C, H, H, H, H, H, H]
    # y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
    # y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]

    metrics = Metrics(y_true, y_pred)
    print(prettify(confusion_matrix(y_true, y_pred), metrics.labels, metrics.labels))
    metrics.report()
