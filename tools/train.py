import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from collections import deque
from tools.io import load_by_step


class CallBack(object):
    """Base class of callbacks
    """

    def __init__(self):
        pass

    def step(self):
        pass


class EarlyStoping(CallBack):
    """Stop training when metric on validation set not imporved
    params:
        mode: str, 'min' or 'max', metric used to define 'improved'
        delta: float, if new metric is in delta range of old metric, no improved
        patience: int, times to wait for improved
    """

    def __init__(self, model, mode='max', delta=0, patience=5, path='./checkpoint/last_best.pt'):
        super(EarlyStoping, self).__init__()
        self.mode = mode
        self.delta = delta
        self.model = model
        self.path = path
        self.ori_patience = patience
        self.cur_patience = patience

        self.last_best = None

    def step(self, value):
        # when first call step
        if self.last_best is None:
            self.last_best = value
            return

        if self.mode == 'max':
            if value - self.last_best > self.delta:
                # better than before
                self.cur_patience = self.ori_patience
                torch.save(self.model.state_dict(),self.path)
                self.last_best = value
            else:
                self.cur_patience -= 1
                if self.cur_patience <= 0:
                    raise StopIteration
        if self.mode == 'min':
            if self.last_best - value > self.delta:
                # better than before
                self.cur_patience = self.ori_patience
                torch.save(self.model.state_dict(),self.path)
                self.last_best = value
            else:
                self.cur_patience -= 1
                if self.cur_patience <= 0:
                    raise StopIteration


class Trainer(object):
    def __init__(
        self,
        device=None,
        loss_record_length=50
    ):
        self.device = device if device is not None else \
            torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.loss_record_length = loss_record_length
        self.setup = False

    def build(self, model, optimizer, criterion, callbacks=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.callbacks = callbacks
        self.setup = True

    def fit(
        self,
        train_loader,
        num_epoch=None,
        validate_loader=None,
        metric=None
    ):
        # check inputs 
        assert self.setup == True, 'You should call setup() before fit()'
        if any([validate_loader, metric]):
            assert all([validate_loader, metric])
        
        # to record something during training
        early_stop = False
        losses = deque([], maxlen=self.loss_record_length)

        # train by epoch
        for epoch in range(num_epoch):
            losses = []
            for data in tqdm(train_loader):
                y_pred = self.train_batch(data)
                loss = self.criterion(y_pred, y_true)
                losses.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print('\ncurrent train loss: {}'.format(np.mean(losses)))

            # validation part
            if validate_loader is not None:
                # use model to do prediction and update metric
                with torch.no_grad():
                    metric.clear()
                    for data in validate_loader:
                        y_pred, y_true = self.validate_batch(data)
                        metric.update_state(y_pred, y_true)
                    metric.display()
                    # check callbacks one by one
                    if self.callbacks is not None:
                        for callback in self.callbacks:
                            try:
                                callback.step(metric.cur_metric)
                            except StopIteration:
                                early_stop = True
                                break

            if early_stop:
                self.model.load_state_dict(torch.load('./checkpoint/last_best.pt'))
                print('early_stop at epoch {}'.format(epoch))
                break


    def fit_by_step(        
        self,
        train_loader,
        steps,
        validate_loader=None,
        metric=None,
        validate_per_step=None
    ):
        # check inputs 
        assert self.setup == True, 'You should call setup() before fit()'
        assert validate_per_step < steps, 'validate per step must be smaller than step num'
        if any([validate_loader, metric, validate_per_step]):
            assert all([validate_loader, metric, validate_per_step])
        
        # to record something during training
        early_stop = False
        losses = deque([], maxlen=self.loss_record_length)
        train_loader = load_by_step(train_loader, steps)
        losses = []

        # train by epoch
        for i, data in tqdm(enumerate(train_loader)):
            y_pred, y_true = self.train_batch(data)
            loss = self.criterion(y_pred, y_true)
            losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # validation part
            if i % validate_per_step == 0:
                print('\ncurrent train loss: {}'.format(np.mean(losses)))
                if validate_loader is not None :
                    # use model to do prediction and update metric
                    with torch.no_grad():
                        metric.clear()
                        for data in validate_loader:
                            y_pred, y_true = self.validate_batch(data)
                            metric.update_state(y_pred, y_true)
                        metric.display()
                        # check callbacks one by one
                        if self.callbacks is not None:
                            for callback in self.callbacks:
                                try:
                                    callback.step(metric.cur_metric)
                                except StopIteration:
                                    early_stop = True
                                    break

                if early_stop:
                    self.model.load_state_dict(torch.load('./checkpoint/last_best.pt'))
                    print('early stop at step {}'.format(i))
                    break

    def predict(self):
        pass

    def train_batch(self):
        pass

    def validate_batch(self):
        pass
