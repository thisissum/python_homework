import torch 
from torch import nn
from modules.layers import Mask, PackingRNN, SamePaddingConv1d
from transformers import BertModel


class Baseline(nn.Module):
    def __init__(self, bert_vocab_num, emb_dim, hidden_dim, output_dim):
        super(Baseline, self).__init__()
        self.bert_vocab_num = bert_vocab_num
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.embedding_layer = nn.Embedding(
            num_embeddings=bert_vocab_num,
            embedding_dim=emb_dim,
            padding_idx=0,
        )
        self.gru_layer = PackingRNN(
            nn.GRU(
                input_size=emb_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
        )
        self.conv_layer = SamePaddingConv1d(emb_dim, hidden_dim, kernel_size=3)
        self.max_pooler = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, ids, mask, seg):
        emb = self.embedding_layer(ids)
        rnn_out = self.gru_layer(emb, mask)
        conv_out = self.conv_layer(emb.permute(0,2,1)).permute(0,2,1)
        rnn_max_pooling = self.max_pooler(rnn_out.permute(0,2,1)).squeeze()
        conv_max_pooling = self.max_pooler(conv_out.permute(0,2,1)).squeeze()
        out = torch.cat([rnn_max_pooling, conv_max_pooling], dim=-1)
        out = self.fc(out)
        return out



class BertPooling(nn.Module):
    def __init__(self, bert_path, hidden_dim=768, output_dim=3):
        super(BertPooling, self).__init__()
        self.bert_path = bert_path
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.embedding = BertModel.from_pretrained(bert_path)
        self.max_pooler = nn.AdaptiveMaxPool1d(1)
        self.avg_pooler = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim*2, output_dim)
        )
    
    def forward(self, ids, mask, seg):
        bert_out = self.embedding(ids, mask, seg)
        max_p = self.max_pooler(bert_out[0].permute(0,2,1)).squeeze()
        avg_p = self.avg_pooler(bert_out[0].permute(0,2,1)).squeeze()
        out = torch.cat([max_p, avg_p], dim=-1)
        out = self.fc(out)
        return out


class BertBiLSTMPooling(nn.Module):
    def __init__(self, bert_path, hidden_dim=768, output_dim=3):
        super(BertBiLSTMPooling, self).__init__()
        self.bert_path = bert_path
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.embedding = BertModel.from_pretrained(bert_path)
        self.gru_layer = PackingRNN(
            nn.LSTM(
                input_size=hidden_dim, 
                hidden_size=hidden_dim//2,
                batch_first=True, 
                bidirectional=True
            )
        )
        self.max_pooler = nn.AdaptiveMaxPool1d(1)
        self.avg_pooler = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_dim*2, output_dim)
        )
    
    def forward(self, ids, mask, seg):
        bert_out = self.embedding(ids, mask, seg)
        out = self.gru_layer(bert_out[0], mask)
        max_p = self.max_pooler(out.permute(0,2,1)).squeeze()
        avg_p = self.avg_pooler(out.permute(0,2,1)).squeeze()
        out = torch.cat([max_p, avg_p], dim=-1)
        out = self.fc(out)
        return out
