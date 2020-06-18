import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from transformers import BertTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import Baseline, BertPooling, BertBiLSTMPooling
from modules.losses import FocalLoss
from tools.io import load_by_step
from tools.metrics import AccuracyMetric, F1Metric
from tools.train import EarlyStoping, Trainer

BATCH_SIZE = 64
NUM_STEP = 4000
VAL_PER_STEP = 200
EPOCH = 10
PAD_LENGTH = 140
BERTPATH = './embeddings'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TextDataset(torch.utils.data.Dataset):
    """
    """

    def __init__(self, x, y=None):
        super(TextDataset, self).__init__()
        self.x = x
        self.y = y
        self.length = len(x)
        self.tokenizer = BertTokenizer.from_pretrained(BERTPATH)
        self.map = {'0':0,'1':1,'-1':2}

    def __getitem__(self, index):
        if self.y is not None:
            y = torch.tensor(self.map[self.y[index]])
            return self._process(self.x[index]), y
        return self._process(self.x[index])

    def __len__(self):
        return self.length

    def _process(self, sentence):
        tk_output = self.tokenizer.encode_plus(
            sentence, max_length=PAD_LENGTH, pad_to_max_length=True
        )
        ids, seg, mask = list(map(lambda x:torch.LongTensor(x),[tk_output['input_ids'], \
            tk_output['token_type_ids'], tk_output['attention_mask']]))
        return ids, seg, mask


class MyTrainer(Trainer):
    def __init__(self):
        super(MyTrainer, self).__init__()
    
    def train_batch(self, data):
        (ids, seg, mask), y = data
        ids, mask, seg, y = list(map(lambda x:x.to(device), [ids, mask, seg, y]))
        y_pred = self.model(ids, mask, seg)
        return y_pred, y
    
    def validate_batch(self, data):
        (ids, seg, mask), y = data
        ids, mask, seg, y = list(map(lambda x:x.to(device), [ids, mask, seg, y]))
        y_pred = self.model(ids, mask, seg)
        return y_pred, y
    
    def predict(self, test_loader):
        self.model.load_state_dict(torch.load('./checkpoint/last_best.pt'))
        with torch.no_grad():
            y_preds = []
            for ids, seg, mask in test_loader:
                ids, mask, seg = list(map(lambda x:x.to(device), [ids, mask, seg]))
                y_pred = self.model(ids, mask, seg)
                y_preds += y_pred.argmax(dim=-1).cpu().numpy().tolist()
            return y_preds


def create_trainer():
    model = BertBiLSTMPooling(
        bert_path=BERTPATH, hidden_dim=768, output_dim=3
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-5
    )
    callbacks = [ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3),EarlyStoping(model, patience=5)]
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = FocalLoss(num_classes=3)
    # criterion = LabelSmoothingLoss()
    
    trainer = MyTrainer()
    trainer.build(model, optimizer, criterion, callbacks=callbacks)
    return trainer

def load_data(path, labeled=True):
    with open(path, 'r', encoding='utf-8') as f:
        output = f.readlines()
    if not labeled:
        return output
    sentences = list(map(lambda x:x.split('LABELIS:')[0].replace('\n',''), output))
    targets = list(map(lambda x:x.split('LABELIS:')[1].replace('\n',''), output))
    return np.asarray(sentences), np.asarray(targets)



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
metric = F1Metric(average='macro')

train_x, train_y = load_data('./data/v1_data/train_labeled.txt', labeled=True)
test_x = load_data('./data/v1_data/test.txt', labeled=False)
test_dataset = TextDataset(test_x)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

kfold = StratifiedKFold(n_splits=7, random_state=233, shuffle=True)
kfold_output = []
for train_index, val_index in kfold.split(train_x, train_y):
    tx, ty = train_x[train_index], train_y[train_index]
    cvx, cvy = train_x[val_index], train_y[val_index]

    train_dataset = TextDataset(x=tx, y=ty)
    val_dataset = TextDataset(x=cvx, y=cvy)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

    trainer = create_trainer()
    
    trainer.fit_by_step(
        train_loader=train_loader, 
        validate_loader=val_loader, 
        steps=NUM_STEP,
        validate_per_step=VAL_PER_STEP, 
        metric=metric
    )
    fold_y_pred = trainer.predict(test_loader)
    kfold_output.append(fold_y_pred)

pd.DataFrame(np.array(kfold_output).T).to_csv('output.csv')
