import torch 
from torch import nn 
import abc
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class Metric(metaclass=abc.ABCMeta):
    """
    Base class of Metric
    Overwrite function: compute_metric, return metric of updated samples
    Overwrite function: update_state, save listed data into pred_record and true_record
    """
    def __init__(self, name=None):
        self.pred_record = list()
        self.true_record = list()
        self.cur_metric = None
        self.name = name
    
    @abc.abstractmethod
    def update_state(self, y_pred, y_true):
        pass
    
    @abc.abstractmethod
    def compute_metric(self):
        pass
    
    def result(self):
        self.cur_metric = self.compute_metric()
        return self.cur_metric

    def display(self):
        cur_metric = self.result()
        print("Validation Metric {}: {}".format(self.name, cur_metric))
    
    def get_cur_metric(self):
        return self.cur_metric

    def clear(self):
        self.pred_record = list()
        self.true_record = list()
    

class AccuracyMetric(Metric):
    """
    Compute accuracy of all sample updated in
    """
    def __init__(self, normalize=True, sample_weight=None):
        super(AccuracyMetric, self).__init__(name='accuracy')
        self.normalize = normalize
        self.sample_weight = sample_weight
    
    def update_state(self, y_pred, y_true):
        y_pred = y_pred.argmax(dim=-1).cpu().view(-1).tolist()
        y_true = y_true.cpu().view(-1).tolist()
        self.pred_record += y_pred 
        self.true_record += y_true

    def compute_metric(self):
        score = accuracy_score(
            self.true_record, 
            self.pred_record, 
            normalize=self.normalize, 
            sample_weight=self.sample_weight
        )
        return score


class F1Metric(Metric):
    """
    Compute F1 score of all sample updated in
    """
    def __init__(self, labels=None, pos_label=1, average='binary', sample_weight=None):
        super(F1Metric, self).__init__(name='{} f1'.format(average))
        self.labels = labels
        self.pos_label=pos_label
        self.average = average
        self.sample_weight = sample_weight

    def update_state(self, y_pred, y_true):
        y_pred = y_pred.argmax(dim=-1).cpu().view(-1).tolist()
        y_true = y_true.cpu().view(-1).tolist()
        self.pred_record += y_pred 
        self.true_record += y_true

    def compute_metric(self):
        score = f1_score(
            self.true_record, 
            self.pred_record, 
            labels=self.labels, 
            pos_label=self.pos_label, 
            average=self.average, 
            sample_weight=self.sample_weight
        )
        return score
