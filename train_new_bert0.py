import json
import os
import warnings
import argparse

if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
warnings.filterwarnings('ignore')


import numpy as np
import torch
from fastNLP import cache_results, prepare_torch_dataloader
from fastNLP import print
from fastNLP import Trainer, Evaluator
from fastNLP import TorchGradClipCallback, MoreEvaluateCallback
from fastNLP import FitlogCallback
from fastNLP import SortedSampler, BucketedBatchSampler
from fastNLP import TorchWarmupCallback
import fitlog
# fitlog.debug()


#from model import CNNNer
#from metrics import NERMetric
#from ner_pipe import SpanNerPipe
#from padder import Torch3DMatrixPadder


from torch import nn
from transformers import AutoModel
from fastNLP import seq_len_to_mask
from torch_scatter import scatter_max
import torch
import torch.nn.functional as F
#from .cnn import MaskCNN
#from .multi_head_biaffine import MultiHeadBiaffine

import random

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0.5, logits=True, reduce=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce


    def forward(self, inputs, targets):
        if self.logits:
            #print("i",inputs, targets)
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            #print("1111111111",BCE_loss)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        #print("BCE_loss",BCE_loss)

        x=BCE_loss
        x_min, x_max = min(x), max(x)
        #归一化
        xx = (x - x_min) / (x_max-x_min)

        pt = torch.exp(-xx)
        #print("pt ",pt )
        #print("maxl ",max(BCE_loss) )
        #print("minl ",min(BCE_loss) )

        #print("maxpt ",max(pt) )
        #print("minpt ",min(pt) )
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def sigmoid_focal_loss(inputs, targets, alpha=0.7, gamma=1, reduction="none"):
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p = torch.sigmoid(inputs)
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss

class CNNNer(nn.Module):
    def __init__(self, model_name, num_ner_tag, cnn_dim=200, biaffine_size=200,
                 size_embed_dim=0, logit_drop=0, kernel_size=3, n_head=4, cnn_depth=3):
        super(CNNNer, self).__init__()
        self.pretrain_model = AutoModel.from_pretrained(model_name)
        hidden_size = self.pretrain_model.config.hidden_size

        if size_embed_dim!=0:
            n_pos = 30
            self.size_embedding = torch.nn.Embedding(n_pos, size_embed_dim)
            _span_size_ids = torch.arange(512) - torch.arange(512).unsqueeze(-1)
            _span_size_ids.masked_fill_(_span_size_ids < -n_pos/2, -n_pos/2)
            _span_size_ids = _span_size_ids.masked_fill(_span_size_ids >= n_pos/2, n_pos/2-1) + n_pos/2
            self.register_buffer('span_size_ids', _span_size_ids.long())
            hsz = biaffine_size*2 + size_embed_dim + 2
        else:
            hsz = biaffine_size*2+2
        biaffine_input_size = hidden_size

        self.head_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(biaffine_input_size, biaffine_size),
            nn.LeakyReLU(),
        )
        self.tail_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(biaffine_input_size, biaffine_size),
            nn.LeakyReLU(),
        )

        self.dropout = nn.Dropout(0.4)
        if n_head>0:
            self.multi_head_biaffine = MultiHeadBiaffine(biaffine_size, cnn_dim, n_head=n_head)
        else:
            self.U = nn.Parameter(torch.randn(cnn_dim, biaffine_size, biaffine_size))
            torch.nn.init.xavier_normal_(self.U.data)
        self.W = torch.nn.Parameter(torch.empty(cnn_dim, hsz))
        torch.nn.init.xavier_normal_(self.W.data)
        if cnn_depth>0:
            self.cnn = MaskCNN(cnn_dim, cnn_dim, kernel_size=kernel_size, depth=cnn_depth)

        self.down_fc = nn.Linear(cnn_dim, num_ner_tag)
        self.logit_drop = logit_drop
        #self.a = FocalLoss()

    def forward(self, input_ids, bpe_len, indexes, matrix):
        attention_mask = seq_len_to_mask(bpe_len)  # bsz x length x length
        outputs = self.pretrain_model(input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_states = outputs['last_hidden_state']
        state = scatter_max(last_hidden_states, index=indexes, dim=1)[0][:, 1:]  # bsz x word_len x hidden_size
        lengths, _ = indexes.max(dim=-1)

        head_state = self.head_mlp(state)
        tail_state = self.tail_mlp(state)
        if hasattr(self, 'U'):
            scores1 = torch.einsum('bxi, oij, byj -> boxy', head_state, self.U, tail_state)
        else:
            scores1 = self.multi_head_biaffine(head_state, tail_state)
        head_state = torch.cat([head_state, torch.ones_like(head_state[..., :1])], dim=-1)
        tail_state = torch.cat([tail_state, torch.ones_like(tail_state[..., :1])], dim=-1)
        affined_cat = torch.cat([self.dropout(head_state).unsqueeze(2).expand(-1, -1, tail_state.size(1), -1),
                                 self.dropout(tail_state).unsqueeze(1).expand(-1, head_state.size(1), -1, -1)], dim=-1)

        if hasattr(self, 'size_embedding'):
            size_embedded = self.size_embedding(self.span_size_ids[:state.size(1), :state.size(1)])
            affined_cat = torch.cat([affined_cat,
                                     self.dropout(size_embedded).unsqueeze(0).expand(state.size(0), -1, -1, -1)], dim=-1)

        scores2 = torch.einsum('bmnh,kh->bkmn', affined_cat, self.W)  # bsz x dim x L x L
        scores = scores2 + scores1   # bsz x dim x L x L

        if hasattr(self, 'cnn'):
            mask = seq_len_to_mask(lengths)  # bsz x length x length
            mask = mask[:, None] * mask.unsqueeze(-1)
            pad_mask = mask[:, None].eq(0)
            u_scores = scores.masked_fill(pad_mask, 0)
            if self.logit_drop != 0:
                u_scores = F.dropout(u_scores, p=self.logit_drop, training=self.training)
            # bsz, num_label, max_len, max_len = u_scores.size()
            u_scores = self.cnn(u_scores, pad_mask)
            scores = u_scores + scores

        scores = self.down_fc(scores.permute(0, 2, 3, 1))

        assert scores.size(-1) == matrix.size(-1)

        if self.training:
            flat_scores = scores.reshape(-1)
            flat_matrix = matrix.reshape(-1)
            mask = flat_matrix.ne(-100).float().view(input_ids.size(0), -1)
            flat_loss = F.binary_cross_entropy_with_logits(flat_scores, flat_matrix.float(), reduction='none')
            #flat_loss=sigmoid_focal_loss(flat_scores, flat_matrix.float())
            loss =((flat_loss.view(input_ids.size(0), -1)*mask).sum(dim=-1)).mean()
            return {'loss': loss}

        return {'scores': scores}

import torch
from fastNLP import Metric
import numpy as np
#from .metrics_utils import _compute_f_rec_pre, decode
def _compute_f_rec_pre(tp, rec, pre):
    pre = tp/(pre+1e-6)
    rec = tp/(rec+1e-6)
    f = 2*pre*rec/(pre+rec+1e-6)
    return round(f*100, 2), round(rec*100, 2), round(pre*100, 2)


def _spans_from_upper_triangular(seq_len: int):
    """Spans from the upper triangular area.
    """
    for start in range(seq_len):
        for end in range(start, seq_len):
            yield (start, end)


def decode(scores, length, allow_nested=False, thres=0.5):
    batch_chunks = []
    for idx, (curr_scores, curr_len) in enumerate(zip(scores, length.cpu().tolist())):
        curr_non_mask = scores.new_ones(curr_len, curr_len, dtype=bool).triu()
        tmp_scores = curr_scores[:curr_len, :curr_len][curr_non_mask].cpu().numpy()  # -1 x 2

        confidences, label_ids = tmp_scores, tmp_scores>=thres
        labels = [i for i in label_ids]
        chunks = [(label, start, end) for label, (start, end) in zip(labels, _spans_from_upper_triangular(curr_len)) if label != 0]
        confidences = [conf for label, conf in zip(labels, confidences) if label != 0]

        assert len(confidences) == len(chunks)
        chunks = [ck for _, ck in sorted(zip(confidences, chunks), reverse=True)]
        chunks = filter_clashed_by_priority(chunks, allow_nested=allow_nested)
        if len(chunks):
            batch_chunks.append(set([(s, e, l) for l,s,e in chunks]))
        else:
            batch_chunks.append(set())
    return batch_chunks


def is_overlapped(chunk1: tuple, chunk2: tuple):
    (_, s1, e1), (_, s2, e2) = chunk1, chunk2
    return (s1 < e2 and s2 < e1)


def is_nested(chunk1: tuple, chunk2: tuple):
    (_, s1, e1), (_, s2, e2) = chunk1, chunk2
    return (s1 <= s2 and e2 <= e1) or (s2 <= s1 and e1 <= e2)


def is_clashed(chunk1: tuple, chunk2: tuple, allow_nested: bool=True):
    if allow_nested:
        return is_overlapped(chunk1, chunk2) and not is_nested(chunk1, chunk2)
    else:
        return is_overlapped(chunk1, chunk2)


def filter_clashed_by_priority(chunks, allow_nested: bool=True):
    filtered_chunks = []
    for ck in chunks:
        if all(not is_clashed(ck, ex_ck, allow_nested=allow_nested) for ex_ck in filtered_chunks):
            filtered_chunks.append(ck)

    return filtered_chunks


class NERMetric(Metric):
    def __init__(self, matrix_segs, ent_thres, allow_nested=True):
        super(NERMetric, self).__init__()
        self.register_element('tp', 0, aggregate_method='sum')
        self.register_element('pre', 0, aggregate_method='sum')
        self.register_element('rec', 0, aggregate_method='sum')

        assert len(matrix_segs) == 1, "Only support pure entities."
        self.allow_nested = allow_nested
        self.ent_thres = ent_thres

    def update(self, ent_target, scores, word_len):
        ent_scores = scores.sigmoid()  # bsz x max_len x max_len x num_class
        ent_scores = (ent_scores + ent_scores.transpose(1, 2))/2
        span_pred = ent_scores.max(dim=-1)[0]

        span_ents = decode(span_pred, word_len, allow_nested=self.allow_nested, thres=self.ent_thres)
        for ents, span_ent, ent_pred in zip(ent_target, span_ents, ent_scores.cpu().numpy()):
            pred_ent = set()
            for s, e, l in span_ent:
                score = ent_pred[s, e]
                ent_type = score.argmax()
                if score[ent_type]>=self.ent_thres:
                    pred_ent.add((s, e, ent_type))
            ents = set(map(tuple, ents))
            self.tp += len(ents.intersection(pred_ent))
            self.pre += len(pred_ent)
            self.rec += len(ents)

    def get_metric(self) -> dict:
        f, rec, pre = _compute_f_rec_pre(self.tp, self.rec, self.pre)
        res = {'f': f, 'rec': rec, 'pre': pre}
        return res

from fastNLP.io import Pipe
from transformers import AutoTokenizer
import numpy as np
import sparse
from tqdm import tqdm
import json
from collections import Counter
from fastNLP import DataSet, Instance
from fastNLP.io import Loader, DataBundle, iob2


class UnifyPipe(Pipe):
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if 'roberta' in model_name:
            self.add_prefix_space = True
            self.cls = self.tokenizer.cls_token_id
            self.sep = self.tokenizer.sep_token_id
        elif 'deberta' in model_name:
            self.add_prefix_space = False
            self.cls = self.tokenizer.bos_token_id
            self.sep = self.tokenizer.eos_token_id
        elif 'bert' in model_name:
            self.add_prefix_space = False
            self.cls = self.tokenizer.cls_token_id
            self.sep = self.tokenizer.sep_token_id
        else:
            self.add_prefix_space = False
            self.cls = self.tokenizer.cls_token_id
            self.sep = self.tokenizer.sep_token_id
            #raise RuntimeError(f"Unsupported {model_name}")


class SpanNerPipe(UnifyPipe):
    def __init__(self, model_name, max_len=400):
        super(SpanNerPipe, self).__init__(model_name)
        self.matrix_segs = {}  
        self.max_len = max_len

    def process(self, data_bundle: DataBundle) -> DataBundle:
        print("process")
        word2bpes = {}
        labels = set()
        for ins in data_bundle.get_dataset('train'):
            raw_ents = ins['raw_ents']
            for s, e, t in raw_ents:
                labels.add(t)
        labels = list(sorted(labels))
        label2idx = {l:i for i,l in enumerate(labels)}
        def get_new_ins(bpes, spans, indexes):
            bpes.append(self.sep)
            cur_word_idx = indexes[-1]
            indexes.append(0)
        
            matrix = np.zeros((cur_word_idx, cur_word_idx, len(label2idx)), dtype=np.int8)
            ent_target = []
            for _ner in spans:
                s, e, t = _ner
                try:
                    matrix[s, e, t] = 1
                    matrix[e, s, t] = 1
                    ent_target.append((s, e, t))
                except:
                    print("pass")
                    pass
            matrix = sparse.COO.from_numpy(matrix)
            assert len(bpes)<=512, len(bpes)
            new_ins = Instance(input_ids=bpes, indexes=indexes, bpe_len=len(bpes),
                               word_len=cur_word_idx, matrix=matrix, ent_target=ent_target)
            return new_ins

        def process(ins):
            raw_words = ins['raw_words']  # List[str]
            raw_ents = ins['raw_ents']  # List[(s, e, t)]
            old_ent_str = Counter()
            has_ent_mask = np.zeros(len(raw_words), dtype=bool)
            for s, e, t in raw_ents:
                old_ent_str[''.join(raw_words[s:e+1])] += 1
                has_ent_mask[s:e+1] = 1
            punct_indexes = []
            for idx, word in enumerate(raw_words):
                # is_upper = True
                # if idx<len(raw_words):
                #     is_upper = raw_words[idx][0].isupper()
                if self.split_name in ('train', 'dev'):
                    
                    if len(word)<1:
                        continue
                    if word[-1] == '.' and has_ent_mask[idx] == 0:  # truncate too long sentence.
                        punct_indexes.append(idx+1)

            if len(punct_indexes) == 0 or punct_indexes[-1] != len(raw_words):
                punct_indexes.append(len(raw_words))

            raw_sents = []
            raw_entss = []
            last_end_idx = 0
            for p_i in punct_indexes:
                raw_sents.append(raw_words[last_end_idx:p_i])
                cur_ents = [(s-last_end_idx, e-last_end_idx, t) for s, e, t in raw_ents if last_end_idx<=s<=e<p_i]
                raw_entss.append(cur_ents)
                last_end_idx = p_i

            bpes = [self.cls]
            indexes = [0]
            spans = []
            ins_lst = []
            new_ent_str = Counter()
            for _raw_words, _raw_ents in zip(raw_sents, raw_entss):
                _indexes = []
                _bpes = []
                for s, e, t in _raw_ents:
                    new_ent_str[''.join(_raw_words[s:e+1])] += 1

                for idx, word in enumerate(_raw_words, start=0):
                    if word in word2bpes:
                        __bpes = word2bpes[word]
                    else:
                        __bpes = self.tokenizer.encode(' '+word if self.add_prefix_space else word,
                                                       add_special_tokens=False)
                        word2bpes[word] = __bpes
                    _indexes.extend([idx]*len(__bpes))
                    _bpes.extend(__bpes)
                next_word_idx = indexes[-1]+1
                if len(bpes) + len(_bpes) <= self.max_len:
                    bpes = bpes + _bpes
                    indexes += [i + next_word_idx for i in _indexes]
                    spans += [(s+next_word_idx-1, e+next_word_idx-1, label2idx.get(t), ) for s, e, t in _raw_ents]
                else:
                    new_ins = get_new_ins(bpes, spans, indexes)
                    ins_lst.append(new_ins)
                    indexes = [0] + [i + 1 for i in _indexes]
                    spans = [(s, e, label2idx.get(t), ) for s, e, t in _raw_ents]
                    bpes = [self.cls] + _bpes
            if bpes:
                ins_lst.append(get_new_ins(bpes, spans, indexes))

            #assert len(new_ent_str - old_ent_str) == 0 and len(old_ent_str-new_ent_str)==0
            return ins_lst

        for name in data_bundle.get_dataset_names():
            self.split_name = name
            ds = data_bundle.get_dataset(name)
            new_ds = DataSet()
            for ins in tqdm(ds, total=len(ds), desc=name, leave=False):
                # in case there exist some overlong sentences, but no sentence will be overlong if follow the provided pre-processing
                ins_lst = process(ins)
                for ins in ins_lst:
                    new_ds.append(ins)
            data_bundle.set_dataset(new_ds, name)

        setattr(data_bundle, 'label2idx', label2idx)
        data_bundle.set_pad('input_ids', self.tokenizer.pad_token_id)
        data_bundle.set_pad('matrix', -100)
        data_bundle.set_pad('ent_target', None)
        self.matrix_segs['ent'] = len(label2idx)
        return data_bundle

    def process_from_file(self, paths: str) -> DataBundle:
        dl = SpanLoader().load(paths)
        return self.process(dl)


class SpanLoader(Loader):
    def _load(self, path):
        ds = DataSet()
        with open(path, 'r',encoding='gbk') as f:
            for line in f:
                data = json.loads(line)
                entities = data['entity_mentions']
                tokens = data['tokens']
                raw_ents = []
                for ent in entities:
                    ent['start']=int(ent['start'])
                    ent['end']=int(ent['end'])
                    raw_ents.append((ent['start'], ent['end'], ent['entity_type']))
                _raw_ents = list(set(raw_ents))
                if len(_raw_ents) != len(raw_ents):
                    print("Detect duplicate entities...")
                ds.append(Instance(raw_words=tokens, raw_ents=raw_ents))
        return ds

import torch
from fastNLP import Padder


class Torch3DMatrixPadder(Padder):
    def __init__(self, num_class, pad_val=-11, batch_size=32, max_len=512):
        super(Torch3DMatrixPadder, self).__init__(pad_val=pad_val, dtype=int)
        self.buffer = torch.full((batch_size, max_len, max_len, num_class), fill_value=self.pad_val,
                                 dtype=torch.float32).clone()

    def __call__(self, field):
        max_len = max([len(f) for f in field])
        buffer = self.buffer[:len(field), :max_len, :max_len].clone()
        buffer.fill_(self.pad_val)
        for i, f in enumerate(field):
            buffer[i, :len(f), :len(f)] = torch.from_numpy(f)

        return buffer


from torch import nn
import torch


class LayerNorm(nn.Module):
    def __init__(self, shape=(1, 7, 1, 1), dim_index=1):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))
        self.dim_index = dim_index
        self.eps = 1e-6

    def forward(self, x):
        """

    
        """
        u = x.mean(dim=self.dim_index, keepdim=True)
        s = (x - u).pow(2).mean(dim=self.dim_index, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x



class MaskConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, groups=1):
        super(MaskConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding,bias=False, groups=groups)
        #self.conv2d = Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding,bias=False, groups=groups)

    def forward(self, x, mask):
        """

        :param x:
        :param mask:
        :return:
        """
        x = x.masked_fill(mask, 0)
        _x = self.conv2d(x)
        return _x
# Code for "Context-Gated Convolution"
# ECCV 2020
# Xudong Lin*, Lin Ma, Wei Liu, Shih-Fu Chang
# {xudong.lin, shih.fu.chang}@columbia.edu, forest.linma@gmail.com, wl2223@columbia.edu

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np  

class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        # for convolutional layers with a kernel size of 1, just use traditional convolution
        if kernel_size == 1:
            self.ind = True
        else:
            self.ind = False            
            self.oc = out_channels
            self.ks = kernel_size
            
            # the target spatial size of the pooling layer
            ws = kernel_size
            self.avg_pool = nn.AdaptiveAvgPool2d((ws,ws))
            
            # the dimension of the latent repsentation
            self.num_lat = int((kernel_size * kernel_size) / 2 + 1)
            
            # the context encoding module
            self.ce = nn.Linear(ws*ws, self.num_lat, False)            
            self.ce_bn = nn.BatchNorm1d(in_channels)
            self.ci_bn2 = nn.BatchNorm1d(in_channels)
            
            # activation function is relu
            self.act = nn.ReLU(inplace=True)
            
            
            # the number of groups in the channel interacting module
            if in_channels // 16:
                self.g = 16
            else:
                self.g = in_channels
            # the channel interacting module    
            self.ci = nn.Linear(self.g, out_channels // (in_channels // self.g), bias=False)
            
            self.ci_bn = nn.BatchNorm1d(out_channels)
            
            # the gate decoding module
            self.gd = nn.Linear(self.num_lat, kernel_size * kernel_size, False)
            self.gd2 = nn.Linear(self.num_lat, kernel_size * kernel_size, False)
            
            # used to prrepare the input feature map to patches
            self.unfold = nn.Unfold(kernel_size, dilation, padding, stride)
            
            # sigmoid function
            self.sig = nn.Sigmoid()
    def forward(self, x):
        # for convolutional layers with a kernel size of 1, just use traditional convolution
        if self.ind:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        else:
            b, c, h, w = x.size()
            weight = self.weight
            # allocate glbal information
            gl = self.avg_pool(x).view(b,c,-1)
            # context-encoding module
            out = self.ce(gl)
            # use different bn for the following two branches
            ce2 = out
            out = self.ce_bn(out)
            out = self.act(out)
            # gate decoding branch 1
            out = self.gd(out)
            # channel interacting module
            if self.g >3:
                # grouped linear
                oc = self.ci(self.act(self.ci_bn2(ce2).view(b, c//self.g, self.g, -1).transpose(2,3))).transpose(2,3).contiguous()
            else:
                # linear layer for resnet.conv1
                oc = self.ci(self.act(self.ci_bn2(ce2).transpose(2,1))).transpose(2,1).contiguous() 
            oc = oc.view(b,self.oc,-1) 
            oc = self.ci_bn(oc)
            oc = self.act(oc)
            # gate decoding branch 2
            oc = self.gd2(oc)   
            # produce gate
            out = self.sig(out.view(b, 1, c, self.ks, self.ks) + oc.view(b, self.oc, 1, self.ks, self.ks))
            # unfolding input feature map to patches
            x_un = self.unfold(x)
            b, _, l = x_un.size()
            # gating
            out = (out * weight.unsqueeze(0)).view(b, self.oc, -1)
            # currently only handle square input and output
            return torch.matmul(out,x_un).view(b, self.oc, int(np.sqrt(l)), int(np.sqrt(l)))   
            


class MaskCNN(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, depth=3):
        super(MaskCNN, self).__init__()

        layers = []
        for i in range(depth):
            layers.extend([
                MaskConv2d(input_channels, input_channels, kernel_size=kernel_size, padding=kernel_size//2),
                LayerNorm((1, input_channels, 1, 1), dim_index=1),
                nn.GELU()])
        layers.append(MaskConv2d(input_channels, output_channels, kernel_size=3, padding=3//2))
        self.cnns = nn.ModuleList(layers)

    def forward(self, x, mask):
        _x = x  
        for layer in self.cnns:
            if isinstance(layer, LayerNorm):
                x = x + _x
                x = layer(x)
                _x = x
            elif not isinstance(layer, nn.GELU):
                x = layer(x, mask)
            else:
                x = layer(x)
        return _x
from torch import nn
import torch


class MultiHeadBiaffine(nn.Module):
    def __init__(self, dim, out=None, n_head=4):
        super(MultiHeadBiaffine, self).__init__()
        assert dim%n_head==0
        in_head_dim = dim//n_head
        out = dim if out is None else out
        assert out%n_head == 0
        out_head_dim = out//n_head
        self.n_head = n_head
        self.W = nn.Parameter(nn.init.xavier_normal_(torch.randn(self.n_head, out_head_dim, in_head_dim, in_head_dim)))
        self.out_dim = out

    def forward(self, h, v):
        """

        :param h: bsz x max_len x dim
        :param v: bsz x max_len x dim
        :return: bsz x max_len x max_len x out_dim
        """
        bsz, max_len, dim = h.size()
        h = h.reshape(bsz, max_len, self.n_head, -1)
        v = v.reshape(bsz, max_len, self.n_head, -1)
        w = torch.einsum('blhx,hdxy,bkhy->bhdlk', h, self.W, v)
        w = w.reshape(bsz, self.out_dim, max_len, max_len)
        return w


parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=2e-5, type=float)
parser.add_argument('-b', '--batch_size', default=48, type=int)
parser.add_argument('-n', '--n_epochs', default=50, type=int)
parser.add_argument('--warmup', default=0.1, type=float)
parser.add_argument('-d', '--dataset_name', default='genia', type=str)
parser.add_argument('--model_name', default=None, type=str)
parser.add_argument('--cnn_depth', default=3, type=int)
parser.add_argument('--cnn_dim', default=200, type=int)
parser.add_argument('--logit_drop', default=0, type=float)
parser.add_argument('--biaffine_size', default=200, type=int)
parser.add_argument('--n_head', default=5, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--accumulation_steps', default=1, type=int)


args = parser.parse_args()
dataset_name = args.dataset_name
if args.model_name is None:
    if 'genia' in args.dataset_name:
        args.model_name = 'dmis-lab/biobert-v1.1'
    elif args.dataset_name in ('ace2004', 'ace2005'):
        #args.model_name = 'junnyu/bert_chinese_mc_base'
        args.model_name = 'roberta-base'
    else:
        args.model_name = 'roberta-base'

model_name = args.model_name
n_head = args.n_head
######hyper
non_ptm_lr_ratio = 100
schedule = 'linear'
weight_decay = 1e-2
size_embed_dim = 25
ent_thres = 0.5
kernel_size = 3
######hyper

def seed_torch(seed=0):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

seed_torch(seed=args.seed)
#os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'  
#torch.use_deterministic_algorithms(True)

fitlog.set_log_dir('logs/')
seed = fitlog.set_rng_seed(rng_seed=args.seed)
os.environ['FASTNLP_GLOBAL_SEED'] = str(seed)
fitlog.add_hyper(args)
fitlog.add_hyper_in_file(__file__)


@cache_results('caches/ner_caches.pkl', _refresh=False)
def get_data(dataset_name, model_name):
    
    if dataset_name == 'ace2004':
        paths = 'preprocess/outputs/ace2004'
    elif dataset_name == 'ace2005':
        paths = 'preprocess/outputs/ace2005'
    elif dataset_name == 'genia':
        paths = 'preprocess/outputs/genia'
    else:
        raise RuntimeError("Does not support.")
    pipe = SpanNerPipe(model_name=model_name)
    dl = pipe.process_from_file(paths)

    return dl, pipe.matrix_segs

dl, matrix_segs = get_data(dataset_name, model_name)
def densify(x):
    x = x.todense().astype(np.float32)
    return x

dl.apply_field(densify, field_name='matrix', new_field_name='matrix', progress_bar='Densify')

print(dl)
label2idx = getattr(dl, 'ner_vocab') if hasattr(dl, 'ner_vocab') else getattr(dl, 'label2idx')
print(f"{len(label2idx)} labels: {label2idx}, matrix_segs:{matrix_segs}")
dls = {}
for name, ds in dl.iter_datasets():
    ds.set_pad('matrix', pad_fn=Torch3DMatrixPadder(pad_val=ds.collator.input_fields['matrix']['pad_val'],
                                                    num_class=matrix_segs['ent'],
                                                    batch_size=args.batch_size))

    if name == 'train':
        _dl = prepare_torch_dataloader(ds, batch_size=args.batch_size, num_workers=0,
                                       batch_sampler=BucketedBatchSampler(ds, 'input_ids',
                                                                          batch_size=args.batch_size,
                                                                          num_batch_per_bucket=30),
                                       pin_memory=True, shuffle=True)
    else:
        _dl = prepare_torch_dataloader(ds, batch_size=args.batch_size, num_workers=0,
                                      sampler=SortedSampler(ds, 'input_ids'), pin_memory=True, shuffle=False)
        if name=="test":print(SortedSampler(ds, 'input_ids'))
    dls[name] = _dl

model = CNNNer(model_name, num_ner_tag=matrix_segs['ent'], cnn_dim=args.cnn_dim, biaffine_size=args.biaffine_size,
                 size_embed_dim=size_embed_dim, logit_drop=args.logit_drop,
                kernel_size=kernel_size, n_head=n_head, cnn_depth=args.cnn_depth)

# optimizer
parameters = []
ln_params = []
non_ln_params = []
non_pretrain_params = []
non_pretrain_ln_params = []

import collections
counter = collections.Counter()
for name, param in model.named_parameters():
    counter[name.split('.')[0]] += torch.numel(param)
print(counter)
print("Total param ", sum(counter.values()))
fitlog.add_to_line(json.dumps(counter, indent=2))
fitlog.add_other(value=sum(counter.values()), name='total_param')

for name, param in model.named_parameters():
    name = name.lower()
    if param.requires_grad is False:
        continue
    if 'pretrain_model' in name:
        if 'norm' in name or 'bias' in name:
            ln_params.append(param)
        else:
            non_ln_params.append(param)
    else:
        if 'norm' in name or 'bias' in name:
            non_pretrain_ln_params.append(param)
        else:
            non_pretrain_params.append(param)
optimizer = torch.optim.AdamW([{'params': non_ln_params, 'lr': args.lr, 'weight_decay': weight_decay},
                               {'params': ln_params, 'lr': args.lr, 'weight_decay': 0},
                               {'params': non_pretrain_ln_params, 'lr': args.lr*non_ptm_lr_ratio, 'weight_decay': 0},
                               {'params': non_pretrain_params, 'lr': args.lr*non_ptm_lr_ratio, 'weight_decay': weight_decay}])
# callbacks
callbacks = []
callbacks.append(FitlogCallback())
callbacks.append(TorchGradClipCallback(clip_value=5))
callbacks.append(TorchWarmupCallback(warmup=args.warmup, schedule=schedule))

evaluate_dls = {}
if 'dev' in dls:
    evaluate_dls = {'dev': dls.get('dev')}
if 'test' in dls:
    evaluate_dls['test'] = dls['test']

allow_nested = True
metrics = {'f': NERMetric(matrix_segs=matrix_segs, ent_thres=ent_thres, allow_nested=allow_nested)}

print(dls['test'])

trainer = Trainer(model=model,
                  driver='torch',
                  train_dataloader=dls.get('train'),
                  evaluate_dataloaders=evaluate_dls,
                  optimizers=optimizer,
                  callbacks=callbacks,
                  overfit_batches=0,
                  device=0,
                  n_epochs=args.n_epochs,
                  metrics=metrics,
                  monitor='f#f#dev',
                  evaluate_every=-1,
                  evaluate_use_dist_sampler=True,
                  accumulation_steps=args.accumulation_steps,
                 fp32=True,
                  progress_bar='rich')

trainer.run(num_train_batch_per_epoch=-1, num_eval_batch_per_dl=-1, num_eval_sanity_batch=1)

class NERMetric2(Metric):
    def __init__(self, matrix_segs, ent_thres, allow_nested=True):
        super(NERMetric2, self).__init__()
        self.register_element('tp', 0, aggregate_method='sum')
        self.register_element('pre', 0, aggregate_method='sum')
        self.register_element('rec', 0, aggregate_method='sum')

        assert len(matrix_segs) == 1, "Only support pure entities."
        self.allow_nested = allow_nested
        self.ent_thres = ent_thres

    def update(self, ent_target, scores, word_len):
        ent_scores = scores.sigmoid()  # bsz x max_len x max_len x num_class
        ent_scores = (ent_scores + ent_scores.transpose(1, 2))/2
        span_pred = ent_scores.max(dim=-1)[0]

        span_ents = decode(span_pred, word_len, allow_nested=self.allow_nested, thres=self.ent_thres)
        index=0
        for ents, span_ent, ent_pred in zip(ent_target, span_ents, ent_scores.cpu().numpy()):
            pred_ent = set()
            for s, e, l in span_ent:
                score = ent_pred[s, e]
                ent_type = score.argmax()
                if score[ent_type]>=self.ent_thres:
                    pred_ent.add((s, e, ent_type))
            ents = set(map(tuple, ents))
            
            index=index+1
            print(index)
            print("pred_ent")
            print(pred_ent)
            print("ents")
            print(ents)

            self.tp += len(ents.intersection(pred_ent))
            self.pre += len(pred_ent)
            self.rec += len(ents)

    def get_metric(self) -> dict:
        f, rec, pre = _compute_f_rec_pre(self.tp, self.rec, self.pre)
        res = {'f': f, 'rec': rec, 'pre': pre}
        return res
allow_nested = True
metrics2 = {'f': NERMetric2(matrix_segs=matrix_segs, ent_thres=ent_thres, allow_nested=allow_nested)}

_dl = prepare_torch_dataloader(ds, batch_size=args.batch_size, num_workers=0,sampler=SortedSampler(ds, 'input_ids'), pin_memory=True, shuffle=False)
dls[name] = _dl

PATH = "model/model_name.pth"
torch.save(model, PATH)
model=torch.load(PATH)

from fastNLP import Evaluator
evaluator=Evaluator(
model=model,
dataloaders=dls['test'],
driver='torch',
metrics=metrics2
)
result=evaluator.run()

fitlog.finish()

