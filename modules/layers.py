import torch 
from torch import nn
import torch.nn.functional as F

class Mask(nn.Module):
    """A mask layer whose output matrix == 1 if not padded else 0
    """
    def __init__(self, pad=0):
        super(Mask, self).__init__()
        self.pad = pad

    def forward(self, inputs):
        return (inputs!=self.pad).float()


class Flatten(nn.Module):
    """Flatten all dimension except for first(batch) dimension"""
    def __init__(self):
        super(Flatten, self).__init__()
    
    def forward(self, inputs):
        batch_size = inputs.size(0)
        return inputs.view(batch_size, -1)



class PackingRNN(nn.Module):
    """use packing and unpacking functions to avoid applying rnn to mask part
    Only used when the 'c' memory is not used for next step
    args:
        rnn_instance: a LSTM or GRU instance
    """

    def __init__(self, rnn_instance):
        super(PackingRNN, self).__init__()
        self.rnn_instance = rnn_instance

    def forward(self, inputs, mask=None):
        if mask is not None:
            sorted_inputs, sorted_length, backpointer = self.sort_by_length(
                inputs, mask
            )
            sorted_inputs = nn.utils.rnn.pack_padded_sequence(
                sorted_inputs,
                sorted_length,
                batch_first=True
            )  # pack seqs for rnn
            out, _ = self.rnn_instance(sorted_inputs, None)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True
            )  # pad back for next step
            original_order_out = out.index_select(index=backpointer,dim=0)
        else: # do nothing
            original_order_out, _ = self.rnn_instance(inputs)
        return original_order_out

    def sort_by_length(self, inputs, mask):
        original_length = mask.long().sum(dim=1)
        sorted_length, sorted_ind = original_length.sort(descending=True)
        sorted_inputs = inputs.index_select(index=sorted_ind, dim=0)
        _, backpointer = sorted_ind.sort(descending=False)
        return sorted_inputs, sorted_length, backpointer


class SamePaddingConv1d(nn.Module):
    """Apply Conv1d to sequence with seq_len unchanged
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1, bias=True):
        super(SamePaddingConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.bias = bias

        self.conv1d = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            dilation=dilation, 
            bias=bias
        )
        pad_plus = True if (kernel_size - 1) * dilation % 2 else False
        pad_one_side = (kernel_size - 1) * dilation // 2
        if pad_plus:
            self.pad = (pad_one_side, pad_one_side+1)
        else:
            self.pad = (pad_one_side, pad_one_side)
    
    def forward(self, x):
        return self.conv1d(F.pad(x, self.pad))


class DGCNN(nn.Module):
    """Implementation of DGCNN from Jianlin Su
    Dilate Gated CNN, same api as nn.Conv1d
    """

    def __init__(self, in_channels, kernel_size, dilation=1, bias=True):
        super(DGCNN, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.out_channels = in_channels * 2
        
        self.conv = SamePaddingConv1d(
            in_channels=in_channels, 
            out_channels=in_channels * 2, 
            kernel_size=kernel_size, 
            dilation=dilation, 
            bias=bias
        )
    
    def forward(self, x):
        conv_x = self.conv(x)
        conv_x, gate = conv_x.split(self.in_channels, dim=-2)
        gate = torch.sigmoid(gate)
        return conv_x * gate + (1 - gate) * x


class LabelAttBiLSTM(nn.Module):
    """Much simple label attention network from LAN
    """

    def __init__(self, input_dim, hidden_dim, label_num):
        super(LabelAttBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.label_num = label_num

        self.bilstm_layer = PackingRNN(
            nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim, 
                num_layers=1,
                batch_first=True, 
                bidirectional=True
            )
        )
        self.label_emb = nn.Parameter(
            nn.init.orthogonal_(torch.Tensor(hidden_dim*2, label_num)),
            requires_grad=True
        )
    
    def forward(self, inputs, mask=None):
        # shape(inputs) = batch_size, seq_len, dk
        # shape(mask) = batch_size, seq_len
        # shape(label_emb) = dk, label_num
        lstm_out = self.bilstm_layer(inputs, mask)
        att = torch.matmul(lstm_out, self.label_emb)

        # shape(att) = batch_size, seq_len, label_num
        weight = torch.softmax(att,dim=-1)

        if mask is not None:
            mask = mask.unsqueeze(-1).expand_as(weight)
            weight = weight.masked_fill(mask==0, 0)
        
        label_att = torch.matmul(weight, self.label_emb.permute(1,0))
        return torch.cat([lstm_out, label_att], dim=-1), weight


class LocalAttention(nn.Module):
    """Implement Local attention with two same and valid mode
    """

    def __init__(self, hidden_dim, kernel_size=3, mode='same'):
        super(LocalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.mode = mode
        self.v = nn.Parameter(
            nn.init.orthogonal_(
                torch.Tensor(hidden_dim, 1)
            )
        )
        self.w1 = nn.Parameter(
            nn.init.orthogonal_(
                torch.Tensor(hidden_dim, hidden_dim)
            )
        )
        self.w2 = nn.Parameter(
            nn.init.orthogonal_(
                torch.Tensor(hidden_dim, hidden_dim)
            )
        )
        pad_plus = True if (kernel_size - 1)  % 2 else False
        pad_one_side = (kernel_size - 1) // 2
        if pad_plus:
            self.pad = (pad_one_side, pad_one_side+1)
        else:
            self.pad = (pad_one_side, pad_one_side)
    
    def pad_sequence(self, x):
        return F.pad(x.permute(0,2,1), self.pad).permute(0,2,1)
    
    def extract_local(self, inputs):
        # shape(inputs)=batch_size, padded_seq_len, dk
        # shape(outupt)=batch_size, seq_len, kernel_size, dk
        local = [inputs[:, i:i+self.kernel_size, :].unsqueeze(1) \
            for i in range(inputs.size(1)-self.kernel_size+1)]
        return torch.cat(local, dim=1)
    
    def forward(self, inputs, mask=None):
         # pad sequence to get same length output
        if mask is not None:
            mask = self.pad_sequence(mask.unsqueeze(-1))
        x = self.pad_sequence(inputs)
        # use window with kernel_size to extract local inputs (and mask)
        local = self.extract_local(x) # batch_size, seq_len, kernel_size, dk
        if mask is not None:
            mask_local = self.extract_local(mask)
        
        # compute attention score with 'add attention'
        score = torch.tanh(
            local.matmul(self.w2) + inputs.matmul(self.w1).unsqueeze(2)
        ).matmul(self.v).squeeze()

        # shape(weight)=batch_size, seq_len, kernel_size, 1
        weight = torch.softmax(score, dim=-1).unsqueeze(-1)
        if mask is not None:
            weight.masked_fill_(mask_local==0, 0)
        
        output = local.permute(0,1,3,2).matmul(weight).squeeze()
        return output



