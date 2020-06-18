import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer
from sklearn.model_selection import StratifiedKFold
from models import BertPooing, Baseline
from tools.metrics import F1Metric, AccuracyMetric
from modules.losses import FocalLoss

BATCH_SIZE = 128
NUM_EPOCH = 2
PAD_LENGTH = 140
BERTPATH = './embeddings'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'


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


class Trainer(object):
    def __init__(self, model, optr, criterion, num_epoch, device=None):
        self.model = model
        self.optr = optr
        self.criterion = criterion
        self.num_epoch = num_epoch
        self.acc_metric = AccuracyMetric()
        self.device = device if device is not None else 'cpu'


    def fit(self, train_laoder, val_loader=None, metric=None):
        for epoch in range(self.num_epoch):
            for (ids, seg, mask), y in tqdm(train_loader):
                ids, seg, mask, y = list(map(lambda x: x.to(self.device), [ids, seg, mask, y]))
                y_pred = self.model(ids, mask, seg)
                loss = self.criterion(y_pred, y)
                self.optr.zero_grad()
                loss.backward()
                self.optr.step()
            
            if val_loader is not None:
                with torch.no_grad():
                    metric.clear()
                    self.acc_metric.clear()
                    for (ids, seg, mask), y in val_loader:
                        ids, seg, mask, y = list(map(lambda x: x.to(self.device), [ids, seg, mask, y]))
                        y_pred = self.model(ids, mask, seg)
                        metric.update_state(y_pred, y)
                        self.acc_metric.update_state(y_pred, y)
                    metric.display()
                    self.acc_metric.display()

    def predict(self, test_loader):
        outputs = []
        for ids, seg, mask in test_loader:
            ids, seg, mask = list(map(lambda x: x.to(self.device),[ids, seg, mask]))
            y_pred = self.model(ids, mask, seg)
            y_pred = torch.argmax(y_pred, dim=-1).cpu().tolist()
            outputs += y_pred
        return outputs


def create_trainer():
    model = Baseline(
        bert_vocab_num=24000, 
        emb_dim=300, 
        hidden_dim=256, 
        output_dim=3
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3
    )
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = FocalLoss(num_classes=3)
    
    trainer = Trainer(model, optimizer, criterion, NUM_EPOCH, device)
    return trainer

def load_data(path, labeled=True):
    with open(path, 'r', encoding='utf-8') as f:
        output = f.readlines()
    if not labeled:
        return output
    sentences = list(map(lambda x:x.split('LABELIS:')[0].replace('\n',''), output))
    targets = list(map(lambda x:x.split('LABELIS:')[1].replace('\n',''), output))
    return np.asarray(sentences), np.asarray(targets)




metric = F1Metric(average='macro')

train_x, train_y = load_data('./data/v1_data/train_labeled.txt', labeled=True)
test_x = load_data('./data/v1_data/test.txt', labeled=False)
test_dataset = TextDataset(test_x)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

kfold = StratifiedKFold(n_splits=5, random_state=233, shuffle=True)
kfold_output = []
for train_index, val_index in kfold.split(train_x, train_y):
    tx, ty = train_x[train_index], train_y[train_index]
    cvx, cvy = train_x[val_index], train_y[val_index]

    train_dataset = TextDataset(x=tx, y=ty)
    val_dataset = TextDataset(x=cvx, y=cvy)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

    trainer = create_trainer()
    
    trainer.fit(train_loader, val_loader=val_loader, metric=metric)
    fold_y_pred = trainer.predict(test_loader)
    kfold_output.append(fold_y_pred)

pd.DataFrame(np.array(kfold_output).T).to_csv('output.csv')