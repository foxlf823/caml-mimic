"""
    Holds PyTorch models
"""
from gensim.models import KeyedVectors
import torch
import torch.nn as nn
import torch.nn.functional as F
# feili
# from torch.nn.init import xavier_uniform
from torch.nn.init import xavier_uniform_ as xavier_uniform
from torch.autograd import Variable

import numpy as np

from math import floor
import random
import sys
import time

from constants import *
from dataproc import extract_wvs

from pytorch_pretrained_bert import BertModel, BertConfig
import os
from embedding import load_pretrain_emb, norm2one
import re

def build_pretrain_embedding(embedding_path, word_alphabet, norm):

    embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)

    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.zeros([len(word_alphabet)+2, embedd_dim], dtype=np.float32)  # add UNK (last) and PAD (0)
    perfect_match = 0
    case_match = 0
    digits_replaced_with_zeros_found = 0
    lowercase_and_digits_replaced_with_zeros_found = 0
    not_match = 0
    for word, index in word_alphabet.items():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1

        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
            case_match += 1

        elif re.sub('\d', '0', word) in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[re.sub('\d', '0', word)])
            else:
                pretrain_emb[index,:] = embedd_dict[re.sub('\d', '0', word)]
            digits_replaced_with_zeros_found += 1

        elif re.sub('\d', '0', word.lower()) in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[re.sub('\d', '0', word.lower())])
            else:
                pretrain_emb[index,:] = embedd_dict[re.sub('\d', '0', word.lower())]
            lowercase_and_digits_replaced_with_zeros_found += 1

        else:
            if norm:
                pretrain_emb[index, :] = norm2one(np.random.uniform(-scale, scale, [1, embedd_dim]))
            else:
                pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1

    # initialize pad and unknown
    pretrain_emb[0, :] = np.zeros([1, embedd_dim], dtype=np.float32)
    if norm:
        pretrain_emb[-1, :] = norm2one(np.random.uniform(-scale, scale, [1, embedd_dim]))
    else:
        pretrain_emb[-1, :] = np.random.uniform(-scale, scale, [1, embedd_dim])


    print("pretrained word emb size {}".format(len(embedd_dict)))
    print("prefect match:%.2f%%, case_match:%.2f%%, dig_zero_match:%.2f%%, "
                 "case_dig_zero_match:%.2f%%, not_match:%.2f%%"
                 %(perfect_match*100.0/len(word_alphabet), case_match*100.0/len(word_alphabet), digits_replaced_with_zeros_found*100.0/len(word_alphabet),
                   lowercase_and_digits_replaced_with_zeros_found*100.0/len(word_alphabet), not_match*100.0/len(word_alphabet)))

    return pretrain_emb, embedd_dim

class BaseModel(nn.Module):

    def __init__(self, args, Y, embed_file, dicts, lmbda=0, dropout=0.5, gpu=-1, embed_size=100):
        super(BaseModel, self).__init__()
        #torch.manual_seed(1337)
        self.gpu = gpu
        self.Y = Y
        self.embed_size = embed_size
        self.embed_drop = nn.Dropout(p=dropout)
        self.lmbda = lmbda

        #make embedding layer
        if embed_file:
            print("loading pretrained embeddings from {}".format(embed_file))
            if args.use_ext_emb:
                pretrain_word_embedding, pretrain_emb_dim = build_pretrain_embedding(embed_file, dicts['w2ind'])
                W = torch.from_numpy(pretrain_word_embedding)
                self.embed_size = pretrain_emb_dim
            else:
                W = torch.Tensor(extract_wvs.load_embeddings(embed_file))

            self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.embed.weight.data = W.clone()
        else:
            #add 2 to include UNK and PAD
            vocab_size = len(dicts['ind2w'])
            self.embed = nn.Embedding(vocab_size+2, embed_size, padding_idx=0)

        self.use_pos = args.use_pos
        if self.use_pos:
            # salience
            #self.pos_embed = nn.Embedding(2, self.embed.embedding_dim, padding_idx=0)
            self.pos_embed = nn.Embedding(3, self.embed.embedding_dim, padding_idx=0)
            

    def _get_loss(self, yhat, target, diffs=None):
        #calculate the BCE
        loss = F.binary_cross_entropy_with_logits(yhat, target)

        #add description regularization loss if relevant
        if self.lmbda > 0 and diffs is not None:
            diff = torch.stack(diffs).mean()
            loss = loss + diff
        return loss

    def embed_descriptions(self, desc_data, gpu):
        #label description embedding via convolutional layer
        #number of labels is inconsistent across instances, so have to iterate over the batch
        b_batch = []
        for inst in desc_data:
            if len(inst) > 0:
                if gpu >= 0:
                    lt = torch.LongTensor(inst).cuda(gpu)
                else:
                    lt = Variable(torch.LongTensor(inst))
                d = self.desc_embedding(lt)
                d = d.transpose(1,2)
                d = self.label_conv(d)
                d = F.max_pool1d(F.tanh(d), kernel_size=d.size()[2])
                d = d.squeeze(2)
                b_inst = self.label_fc1(d)
                b_batch.append(b_inst)
            else:
                b_batch.append([])
        return b_batch

    def _compare_label_embeddings(self, target, b_batch, desc_data):
        #description regularization loss 
        #b is the embedding from description conv
        #iterate over batch because each instance has different # labels
        diffs = []
        for i,bi in enumerate(b_batch):
            ti = target[i]
            inds = torch.nonzero(ti.data).squeeze().cpu().numpy()

            zi = self.final.weight[inds,:]
            diff = (zi - bi).mul(zi - bi).mean()

            #multiply by number of labels to make sure overall mean is balanced with regard to number of labels
            diffs.append(self.lmbda*diff*bi.size()[0])
        return diffs


class Bert_BaseModel(nn.Module):

    def __init__(self, Y, embed_file, dicts, bert_dir, lmbda=0, dropout=0.5, gpu=-1, embed_size=100):
        super(Bert_BaseModel, self).__init__()
        #torch.manual_seed(1337)

        print("loading pretrained bert from {}".format(bert_dir))
        config_file = os.path.join(bert_dir, 'bert_config.json')
        self.bert_config = BertConfig.from_json_file(config_file)
        print("Model config {}".format(self.bert_config))
        self.bert_size = self.bert_config.hidden_size

        self.bert = BertModel.from_pretrained(bert_dir)
        self.bert_drop = nn.Dropout(p=dropout)


    def _get_loss(self, yhat, target, diffs=None):
        # calculate the BCE
        loss = F.binary_cross_entropy_with_logits(yhat, target)

        return loss

class BOWPool(BaseModel):
    """
        Logistic regression model over average or max-pooled word vector input
    """

    def __init__(self, Y, embed_file, lmbda, gpu, dicts, pool='max', embed_size=100, dropout=0.5, code_emb=None):
        super(BOWPool, self).__init__(Y, embed_file, dicts, lmbda, dropout=dropout, gpu=gpu, embed_size=embed_size)
        self.final = nn.Linear(embed_size, Y)
        if code_emb:
            self._code_emb_init(code_emb, dicts)
        else:
            xavier_uniform(self.final.weight)
        self.pool = pool

    def _code_emb_init(self, code_emb, dicts):
        code_embs = KeyedVectors.load_word2vec_format(code_emb)
        weights = np.zeros(self.final.weight.size())
        for i in range(self.Y):
            code = dicts['ind2c'][i]
            weights[i] = code_embs[code]
        self.final.weight.data = torch.Tensor(weights).clone()

    def forward(self, x, target, desc_data=None, get_attention=False):
        #get embeddings and apply dropout
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)
        if self.pool == 'max':
            import pdb; pdb.set_trace()
            x = F.max_pool1d(x)
        else:
            x = F.avg_pool1d(x)
        logits = F.sigmoid(self.final(x))
        loss = self._get_loss(logits, target, diffs)
        return yhat, loss, None

class ConvAttnPool(BaseModel):

    def __init__(self, args, Y, embed_file, kernel_size, num_filter_maps, lmbda, gpu, dicts, embed_size=100, dropout=0.5, code_emb=None):
        super(ConvAttnPool, self).__init__(args, Y, embed_file, dicts, lmbda, dropout=dropout, gpu=gpu, embed_size=embed_size)

        if self.use_pos:
            self.conv = nn.Conv1d(self.embed_size*2, num_filter_maps, kernel_size=kernel_size,
                                  padding=int(floor(kernel_size / 2)))
        else:
            #initialize conv layer as in 2.1
            self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size, padding=int(floor(kernel_size/2)))
        xavier_uniform(self.conv.weight)

        #context vectors for computing attention as in 2.2
        self.U = nn.Linear(num_filter_maps, Y)
        xavier_uniform(self.U.weight)

        #final layer: create a matrix to use for the L binary classifiers as in 2.3
        self.final = nn.Linear(num_filter_maps, Y)
        xavier_uniform(self.final.weight)

        #initialize with trained code embeddings if applicable
        if code_emb:
            self._code_emb_init(code_emb, dicts)
            #also set conv weights to do sum of inputs
            weights = torch.eye(self.embed_size).unsqueeze(2).expand(-1,-1,kernel_size)/kernel_size
            self.conv.weight.data = weights.clone()
            self.conv.bias.data.zero_()
        
        #conv for label descriptions as in 2.5
        #description module has its own embedding and convolution layers
        if lmbda > 0:
            W = self.embed.weight.data
            self.desc_embedding = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.desc_embedding.weight.data = W.clone()

            self.label_conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size, padding=int(floor(kernel_size/2)))
            xavier_uniform(self.label_conv.weight)

            self.label_fc1 = nn.Linear(num_filter_maps, num_filter_maps)
            xavier_uniform(self.label_fc1.weight)

    def _code_emb_init(self, code_emb, dicts):
        code_embs = KeyedVectors.load_word2vec_format(code_emb)
        weights = np.zeros(self.final.weight.size())
        for i in range(self.Y):
            code = dicts['ind2c'][i]
            weights[i] = code_embs[code]
        self.U.weight.data = torch.Tensor(weights).clone()
        self.final.weight.data = torch.Tensor(weights).clone()
        
    def forward(self, x, target, desc_data=None, get_attention=True):

        if self.use_pos:
            word, pos = x
            x = torch.cat([self.embed(word), self.pos_embed(pos)], dim=-1)
        else:
            x = self.embed(x)

        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        #apply convolution and nonlinearity (tanh)
        # feili
        # x = F.tanh(self.conv(x).transpose(1,2))
        x = torch.tanh(self.conv(x).transpose(1, 2))
        #apply attention
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1,2)), dim=2)
        #document representations are weighted sums using the attention. Can compute all at once as a matmul
        m = alpha.matmul(x)
        #final layer classification
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        
        if desc_data is not None:
            #run descriptions through description module
            b_batch = self.embed_descriptions(desc_data, self.gpu)
            #get l2 similarity loss
            diffs = self._compare_label_embeddings(target, b_batch, desc_data)
        else:
            diffs = None
            
        #final sigmoid to get predictions
        yhat = y
        loss = self._get_loss(yhat, target, diffs)
        return yhat, loss, alpha


class ConvAttnPool_ldep(BaseModel):

    def __init__(self, Y, embed_file, kernel_size, num_filter_maps, lmbda, gpu, dicts, embed_size=100, dropout=0.5,
                 code_emb=None):
        super(ConvAttnPool_ldep, self).__init__(Y, embed_file, dicts, lmbda, dropout=dropout, gpu=gpu, embed_size=embed_size)

        # initialize conv layer as in 2.1
        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size,
                              padding=int(floor(kernel_size / 2)))
        xavier_uniform(self.conv.weight)

        # context vectors for computing attention as in 2.2
        self.U = nn.Linear(num_filter_maps, Y)
        xavier_uniform(self.U.weight)

        # final layer: create a matrix to use for the L binary classifiers as in 2.3
        self.final = nn.Linear(num_filter_maps, Y)
        xavier_uniform(self.final.weight)

        self.ldep = nn.Linear(Y, Y, bias=False)
        xavier_uniform(self.ldep.weight)
        #self.ldep.weight.data.copy_(torch.eye(Y, Y))

        # initialize with trained code embeddings if applicable
        if code_emb:
            self._code_emb_init(code_emb, dicts)
            # also set conv weights to do sum of inputs
            weights = torch.eye(self.embed_size).unsqueeze(2).expand(-1, -1, kernel_size) / kernel_size
            self.conv.weight.data = weights.clone()
            self.conv.bias.data.zero_()

        # conv for label descriptions as in 2.5
        # description module has its own embedding and convolution layers
        if lmbda > 0:
            W = self.embed.weight.data
            self.desc_embedding = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.desc_embedding.weight.data = W.clone()

            self.label_conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size,
                                        padding=int(floor(kernel_size / 2)))
            xavier_uniform(self.label_conv.weight)

            self.label_fc1 = nn.Linear(num_filter_maps, num_filter_maps)
            xavier_uniform(self.label_fc1.weight)

    def _code_emb_init(self, code_emb, dicts):
        code_embs = KeyedVectors.load_word2vec_format(code_emb)
        weights = np.zeros(self.final.weight.size())
        for i in range(self.Y):
            code = dicts['ind2c'][i]
            weights[i] = code_embs[code]
        self.U.weight.data = torch.Tensor(weights).clone()
        self.final.weight.data = torch.Tensor(weights).clone()

    def forward(self, x, target, desc_data=None, get_attention=True):
        # get embeddings and apply dropout
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        # apply convolution and nonlinearity (tanh)
        # feili
        # x = F.tanh(self.conv(x).transpose(1,2))
        x = torch.tanh(self.conv(x).transpose(1, 2))
        # apply attention
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
        # document representations are weighted sums using the attention. Can compute all at once as a matmul
        m = alpha.matmul(x)
        # final layer classification
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        y = self.ldep(y)

        if desc_data is not None:
            # run descriptions through description module
            b_batch = self.embed_descriptions(desc_data, self.gpu)
            # get l2 similarity loss
            diffs = self._compare_label_embeddings(target, b_batch, desc_data)
        else:
            diffs = None

        # final sigmoid to get predictions
        yhat = y
        loss = self._get_loss(yhat, target, diffs)
        return yhat, loss, alpha


class Bert_ConvAttn(Bert_BaseModel):

    def __init__(self, Y, embed_file, kernel_size, num_filter_maps, lmbda, gpu, dicts, bert_dir, embed_size=100, dropout=0.5,
                 code_emb=None):
        super(Bert_ConvAttn, self).__init__(Y, embed_file, dicts, bert_dir, lmbda, dropout=dropout, gpu=gpu, embed_size=embed_size)

        # initialize conv layer as in 2.1
        self.conv = nn.Conv1d(self.bert_size, num_filter_maps, kernel_size=kernel_size,
                              padding=int(floor(kernel_size / 2)))
        xavier_uniform(self.conv.weight)

        # context vectors for computing attention as in 2.2
        self.U = nn.Linear(num_filter_maps, Y)
        xavier_uniform(self.U.weight)

        # final layer: create a matrix to use for the L binary classifiers as in 2.3
        self.final = nn.Linear(num_filter_maps, Y)
        xavier_uniform(self.final.weight)

        # initialize with trained code embeddings if applicable
        if code_emb:
            self._code_emb_init(code_emb, dicts)
            # also set conv weights to do sum of inputs
            weights = torch.eye(self.embed_size).unsqueeze(2).expand(-1, -1, kernel_size) / kernel_size
            self.conv.weight.data = weights.clone()
            self.conv.bias.data.zero_()

        # conv for label descriptions as in 2.5
        # description module has its own embedding and convolution layers
        if lmbda > 0:
            W = self.embed.weight.data
            self.desc_embedding = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.desc_embedding.weight.data = W.clone()

            self.label_conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size,
                                        padding=int(floor(kernel_size / 2)))
            xavier_uniform(self.label_conv.weight)

            self.label_fc1 = nn.Linear(num_filter_maps, num_filter_maps)
            xavier_uniform(self.label_fc1.weight)

    def _code_emb_init(self, code_emb, dicts):
        code_embs = KeyedVectors.load_word2vec_format(code_emb)
        weights = np.zeros(self.final.weight.size())
        for i in range(self.Y):
            code = dicts['ind2c'][i]
            weights[i] = code_embs[code]
        self.U.weight.data = torch.Tensor(weights).clone()
        self.final.weight.data = torch.Tensor(weights).clone()

    def forward(self, x, target, desc_data=None, get_attention=True):

        input_ids, attention_mask, token_type_ids, batch_size, chunk_num = x # input_ids (batch x chunk_num)
        _, doc_chunk_rep = self.bert(input_ids, token_type_ids, attention_mask,
                                                  output_all_encoded_layers=False) # doc_chunk_rep ((batch x chunk_num) bert_size)
        doc_chunk_rep = self.bert_drop(doc_chunk_rep)

        doc_rep = doc_chunk_rep.view(batch_size, chunk_num, -1) # (batch, chunk_num bert_size)

        x = doc_rep.transpose(1, 2)

        # apply convolution and nonlinearity (tanh)
        # feili
        # x = F.tanh(self.conv(x).transpose(1,2))
        x = torch.tanh(self.conv(x).transpose(1, 2))
        # apply attention
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
        # document representations are weighted sums using the attention. Can compute all at once as a matmul
        m = alpha.matmul(x)
        # final layer classification
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        if desc_data is not None:
            # run descriptions through description module
            b_batch = self.embed_descriptions(desc_data, self.gpu)
            # get l2 similarity loss
            diffs = self._compare_label_embeddings(target, b_batch, desc_data)
        else:
            diffs = None

        # final sigmoid to get predictions
        yhat = y
        loss = self._get_loss(yhat, target, diffs)
        return yhat, loss, alpha

class Bert_Pooling(Bert_BaseModel):

    def __init__(self, Y, embed_file, kernel_size, num_filter_maps, lmbda, gpu, dicts, bert_dir, embed_size=100, dropout=0.5,
                 code_emb=None):
        super(Bert_Pooling, self).__init__(Y, embed_file, dicts, bert_dir, lmbda, dropout=dropout, gpu=gpu, embed_size=embed_size)

        # self.conv = nn.Conv1d(self.bert_size, num_filter_maps, kernel_size=3, padding=1)
        # xavier_uniform(self.conv.weight)
        #
        # self.batchnorm = nn.BatchNorm1d(num_filter_maps)
        #

        # self.final = nn.Linear(num_filter_maps, Y)
        self.final_dropout = nn.Dropout(p=dropout)
        self.final = nn.Linear(self.bert_size, Y)
        xavier_uniform(self.final.weight)

    def forward(self, inputs_id, segments, masks, target):

        _, sentences_rep = self.bert(inputs_id, segments, masks, output_all_encoded_layers=False)
        # sentences_rep = self.bert_drop(sentences_rep)
        #
        # x = sentences_rep.unsqueeze(0).transpose(1, 2)
        #
        # x = F.relu(self.batchnorm(self.conv(x)))
        # x = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(-1)

        x = sentences_rep.unsqueeze(0).transpose(1, 2)
        x = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(-1)

        x = self.final_dropout(x)
        y = self.final(x)

        loss = self._get_loss(y, target)
        return y, loss

class Bert_Conv(Bert_BaseModel):

    def __init__(self, Y, embed_file, kernel_size, num_filter_maps, lmbda, gpu, dicts, bert_dir, embed_size=100, dropout=0.5,
                 code_emb=None):
        super(Bert_Conv, self).__init__(Y, embed_file, dicts, bert_dir, lmbda, dropout=dropout, gpu=gpu, embed_size=embed_size)

        self.conv = nn.Conv1d(self.bert_size, num_filter_maps, kernel_size=3, padding=1)
        xavier_uniform(self.conv.weight)

        # self.batchnorm = nn.BatchNorm1d(num_filter_maps)


        self.final = nn.Linear(num_filter_maps, Y)
        self.final_dropout = nn.Dropout(p=dropout)

        xavier_uniform(self.final.weight)

    def forward(self, inputs_id, segments, masks, target):

        _, sentences_rep = self.bert(inputs_id, segments, masks, output_all_encoded_layers=False)
        sentences_rep = self.bert_drop(sentences_rep)

        x = sentences_rep.unsqueeze(0).transpose(1, 2)

        x = self.conv(x)
        # x = self.batchnorm(x)
        # x = F.tanh(x)


        x = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(-1)


        x = self.final_dropout(x)
        y = self.final(x)

        loss = self._get_loss(y, target)
        return y, loss

from pytorch_pretrained_bert.modeling import BertLayer, BertEmbeddings
import copy
class Transformer1(nn.Module):

    def __init__(self, args, Y):
        super(Transformer1, self).__init__()

        assert args.sent_encoder_layer_start <= args.sent_encoder_layer_end
        assert 1<= args.sent_encoder_layer_start and args.sent_encoder_layer_start<=12
        assert 1<= args.sent_encoder_layer_end and args.sent_encoder_layer_end<=12

        print("loading pretrained bert from {}".format(args.bert_dir))
        config_file = os.path.join(args.bert_dir, 'bert_config.json')
        bert_config = BertConfig.from_json_file(config_file)
        print("Model config {}".format(bert_config))
        pretrained_bert = BertModel.from_pretrained(args.bert_dir)

        self.embeddings = copy.deepcopy(pretrained_bert.embeddings)

        self.sent_encoder = nn.ModuleList()
        i = args.sent_encoder_layer_start-1
        while i <= args.sent_encoder_layer_end-1:
            layer = pretrained_bert.encoder.layer[i]
            self.sent_encoder.append(copy.deepcopy(layer))
            i += 1

        del pretrained_bert

        # self.dim_reduction = nn.Sequential()
        # self.dim_reduction.add_module('linear', nn.Linear(bert_config.hidden_size, args.num_filter_maps))
        # self.dim_reduction.add_module('dropout', nn.Dropout(p=bert_config.hidden_dropout_prob))
        # self.dim_reduction.add_module('norm', nn.LayerNorm(args.num_filter_maps))
        # self.dim_reduction.add_module('act', nn.ReLU())

        # self.final_dropout = nn.Dropout(p=bert_config.hidden_dropout_prob)
        self.final = nn.Linear(args.num_filter_maps, Y)

    def forward(self, input_ids, token_type_ids, attention_mask, target, doc_lengths):
        batch_size, sent_num, sent_len = input_ids.size()
        input_ids = input_ids.view(batch_size*sent_num, sent_len)
        token_type_ids = token_type_ids.view(batch_size*sent_num, sent_len)
        attention_mask = attention_mask.view(batch_size*sent_num, sent_len)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)

        hidden_states = embedding_output
        for layer_module in self.sent_encoder:
            hidden_states = layer_module(hidden_states, extended_attention_mask)

        sent_rep = hidden_states[:, 0]
        sent_rep = sent_rep.view(batch_size, sent_num, -1)
        x = sent_rep.transpose(1, 2)
        x = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(-1)

        # x = self.dim_reduction(x)

        # x = self.final_dropout(x)
        y = self.final(x)

        loss = F.binary_cross_entropy_with_logits(y, target)

        return y, loss

class DotAttentionLayer(nn.Module):
    def __init__(self, hidden_size, gpu):
        super(DotAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, 1, bias=False)
        self.gpu = gpu

    def forward(self, inputs, lengths):
        """
        input: (unpacked_padded_output: batch_size x seq_len x hidden_size, lengths: batch_size)
        """
        batch_size, max_len, _ = inputs.size()
        flat_input = inputs.contiguous().view(-1, self.hidden_size)
        logits = self.W(flat_input).view(batch_size, max_len)
        alphas = F.softmax(logits, dim=1)

        # computing mask
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        if self.gpu >= 0 and torch.cuda.is_available():
            idxes = idxes.cuda(self.gpu)
        mask = (idxes<lengths.unsqueeze(1)).float()

        alphas = alphas * mask
        # renormalize
        alphas = alphas / torch.sum(alphas, 1).view(-1, 1)
        output = torch.bmm(alphas.unsqueeze(1), inputs).squeeze(1)
        return output

class Transformer2(nn.Module):

    def __init__(self, args, Y):
        super(Transformer2, self).__init__()

        assert args.sent_encoder_layer_start <= args.sent_encoder_layer_end
        assert 1<= args.sent_encoder_layer_start and args.sent_encoder_layer_start<=12
        assert 1<= args.sent_encoder_layer_end and args.sent_encoder_layer_end<=12

        print("loading pretrained bert from {}".format(args.bert_dir))
        config_file = os.path.join(args.bert_dir, 'bert_config.json')
        bert_config = BertConfig.from_json_file(config_file)
        print("Model config {}".format(bert_config))
        pretrained_bert = BertModel.from_pretrained(args.bert_dir)

        self.embeddings = copy.deepcopy(pretrained_bert.embeddings)

        self.sent_encoder = nn.ModuleList()
        i = args.sent_encoder_layer_start-1
        while i <= args.sent_encoder_layer_end-1:
            layer = pretrained_bert.encoder.layer[i]
            self.sent_encoder.append(copy.deepcopy(layer))
            i += 1

        del pretrained_bert

        # dot-attention
        self.dot_attn = DotAttentionLayer(bert_config.hidden_size, args.gpu)

        self.final = nn.Linear(args.num_filter_maps, Y)

    def forward(self, input_ids, token_type_ids, attention_mask, target, doc_lengths):
        batch_size, sent_num, sent_len = input_ids.size()
        input_ids = input_ids.view(batch_size*sent_num, sent_len)
        token_type_ids = token_type_ids.view(batch_size*sent_num, sent_len)
        attention_mask = attention_mask.view(batch_size*sent_num, sent_len)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)

        hidden_states = embedding_output
        for layer_module in self.sent_encoder:
            hidden_states = layer_module(hidden_states, extended_attention_mask)

        sent_rep = hidden_states[:, 0]

        sent_rep = sent_rep.view(batch_size, sent_num, -1)

        x = self.dot_attn(sent_rep, doc_lengths)

        y = self.final(x)

        loss = F.binary_cross_entropy_with_logits(y, target)

        return y, loss

import transformer
class Transformer3(nn.Module):

    def __init__(self, args, Y):
        super(Transformer3, self).__init__()

        assert args.sent_encoder_layer_start <= args.sent_encoder_layer_end
        assert 1<= args.sent_encoder_layer_start and args.sent_encoder_layer_start<=12
        assert 1<= args.sent_encoder_layer_end and args.sent_encoder_layer_end<=12

        print("loading pretrained bert from {}".format(args.bert_dir))
        config_file = os.path.join(args.bert_dir, 'bert_config.json')
        bert_config = BertConfig.from_json_file(config_file)
        print("Model config {}".format(bert_config))
        pretrained_bert = BertModel.from_pretrained(args.bert_dir)

        self.embeddings = copy.deepcopy(pretrained_bert.embeddings)

        self.sent_encoder = nn.ModuleList()
        i = args.sent_encoder_layer_start-1
        while i <= args.sent_encoder_layer_end-1:
            layer = pretrained_bert.encoder.layer[i]
            self.sent_encoder.append(copy.deepcopy(layer))
            i += 1

        del pretrained_bert

        # sentence-level self-attention
        self.position_enc = nn.Embedding.from_pretrained(
            transformer.Models.get_sinusoid_encoding_table(args.max_sent_num+1, bert_config.hidden_size, padding_idx=0),
            freeze=True)

        doc_encoder_head = 12
        assert bert_config.hidden_size % doc_encoder_head == 0
        self.doc_encoder = nn.ModuleList([
            transformer.Layers.EncoderLayer(bert_config.hidden_size, 4*bert_config.hidden_size, doc_encoder_head,
                                            bert_config.hidden_size//doc_encoder_head, bert_config.hidden_size//doc_encoder_head
                                            , dropout=bert_config.hidden_dropout_prob)
            for _ in range(args.doc_encoder_layer)])


        # dot-attention
        self.dot_attn = DotAttentionLayer(bert_config.hidden_size, args.gpu)

        self.final = nn.Linear(args.num_filter_maps, Y)

    def forward(self, input_ids, token_type_ids, attention_mask, target, doc_lengths, pad_sent, pad_sent_position):
        batch_size, sent_num, sent_len = input_ids.size()
        input_ids = input_ids.view(batch_size*sent_num, sent_len)
        token_type_ids = token_type_ids.view(batch_size*sent_num, sent_len)
        attention_mask = attention_mask.view(batch_size*sent_num, sent_len)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)

        hidden_states = embedding_output
        for layer_module in self.sent_encoder:
            hidden_states = layer_module(hidden_states, extended_attention_mask)

        sent_rep = hidden_states[:, 0]

        sent_rep = sent_rep.view(batch_size, sent_num, -1)

        slf_attn_mask = transformer.Models.get_attn_key_pad_mask(seq_k=pad_sent, seq_q=pad_sent)
        non_pad_mask = transformer.Models.get_non_pad_mask(pad_sent)

        enc_output = sent_rep + self.position_enc(pad_sent_position)

        for enc_layer in self.doc_encoder:
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        x = self.dot_attn(enc_output, doc_lengths)

        y = self.final(x)

        loss = F.binary_cross_entropy_with_logits(y, target)

        return y, loss

class Transformer4(nn.Module):

    def __init__(self, args, Y):
        super(Transformer4, self).__init__()

        assert args.sent_encoder_layer_start <= args.sent_encoder_layer_end
        assert 1<= args.sent_encoder_layer_start and args.sent_encoder_layer_start<=12
        assert 1<= args.sent_encoder_layer_end and args.sent_encoder_layer_end<=12

        print("loading pretrained bert from {}".format(args.bert_dir))
        config_file = os.path.join(args.bert_dir, 'bert_config.json')
        bert_config = BertConfig.from_json_file(config_file)
        print("Model config {}".format(bert_config))
        pretrained_bert = BertModel.from_pretrained(args.bert_dir)

        self.embeddings = copy.deepcopy(pretrained_bert.embeddings)

        self.sent_encoder = nn.ModuleList()
        i = args.sent_encoder_layer_start-1
        while i <= args.sent_encoder_layer_end-1:
            layer = pretrained_bert.encoder.layer[i]
            self.sent_encoder.append(copy.deepcopy(layer))
            i += 1

        del pretrained_bert

        # sentence-level self-attention
        self.position_enc = nn.Embedding.from_pretrained(
            transformer.Models.get_sinusoid_encoding_table(args.max_sent_num+1, bert_config.hidden_size, padding_idx=0),
            freeze=True)

        doc_encoder_head = 12
        assert bert_config.hidden_size % doc_encoder_head == 0
        self.doc_encoder = nn.ModuleList([
            transformer.Layers.EncoderLayer(bert_config.hidden_size, 4*bert_config.hidden_size, doc_encoder_head,
                                            bert_config.hidden_size//doc_encoder_head, bert_config.hidden_size//doc_encoder_head
                                            , dropout=bert_config.hidden_dropout_prob)
            for _ in range(args.doc_encoder_layer)])

        self.final = nn.Linear(args.num_filter_maps, Y)

    def forward(self, input_ids, token_type_ids, attention_mask, target, doc_lengths, pad_sent, pad_sent_position):
        batch_size, sent_num, sent_len = input_ids.size()
        input_ids = input_ids.view(batch_size*sent_num, sent_len)
        token_type_ids = token_type_ids.view(batch_size*sent_num, sent_len)
        attention_mask = attention_mask.view(batch_size*sent_num, sent_len)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)

        hidden_states = embedding_output
        for layer_module in self.sent_encoder:
            hidden_states = layer_module(hidden_states, extended_attention_mask)

        sent_rep = hidden_states[:, 0]

        sent_rep = sent_rep.view(batch_size, sent_num, -1)

        slf_attn_mask = transformer.Models.get_attn_key_pad_mask(seq_k=pad_sent, seq_q=pad_sent)
        non_pad_mask = transformer.Models.get_non_pad_mask(pad_sent)

        enc_output = sent_rep + self.position_enc(pad_sent_position)

        for enc_layer in self.doc_encoder:
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        x = enc_output.transpose(1, 2)
        x = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(-1)

        y = self.final(x)

        loss = F.binary_cross_entropy_with_logits(y, target)

        return y, loss

from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertLayerNorm
class Bert_seq_cls(nn.Module):

    def __init__(self, args, Y):
        super(Bert_seq_cls, self).__init__()

        print("loading pretrained bert from {}".format(args.bert_dir))
        config_file = os.path.join(args.bert_dir, 'bert_config.json')
        self.config = BertConfig.from_json_file(config_file)
        print("Model config {}".format(self.config))
        self.bert = BertModel.from_pretrained(args.bert_dir)

        self.dim_reduction = nn.Linear(self.config.hidden_size, args.num_filter_maps)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(args.num_filter_maps, Y)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, target):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        x = self.dim_reduction(pooled_output)
        x = self.dropout(x)
        y = self.classifier(x)

        loss = F.binary_cross_entropy_with_logits(y, target)
        return y, loss

    def init_bert_weights(self, module):

        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

# class CNN(nn.Module):
#     def __init__(self, args, Y):
#         super(CNN, self).__init__()
#
#         W = torch.Tensor(extract_wvs.load_embeddings(args.embed_file))
#         self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
#         self.embed.weight.data = W.clone()
#
#         self.embed_drop = nn.Dropout(p=args.dropout)
#
#         filter_size = int(args.filter_size)
#
#         self.conv = nn.Conv1d(self.embed.embedding_dim, args.num_filter_maps, kernel_size=filter_size,
#                               padding=int(floor(filter_size / 2)))
#         xavier_uniform(self.conv.weight)
#
#         # context vectors for computing attention as in 2.2
#         self.U = nn.Linear(args.num_filter_maps, Y)
#         xavier_uniform(self.U.weight)
#
#         self.use_tfidf = args.use_tfidf
#         if self.use_tfidf:
#             self.tfidf_hidden = nn.Sequential(nn.Linear(W.size()[0], args.num_filter_maps),
#                                         nn.Dropout(p=args.dropout),
#                                         nn.LayerNorm(args.num_filter_maps),
#                                         nn.ReLU())
#
#
#         # final layer: create a matrix to use for the L binary classifiers as in 2.3
#         if self.use_tfidf :
#             self.final = nn.Linear(2*args.num_filter_maps, Y)
#         else:
#             self.final = nn.Linear(args.num_filter_maps, Y)
#         xavier_uniform(self.final.weight)
#
#         self.Y = Y
#
#     def forward(self, x, target, tfidf):
#
#         x = self.embed(x)
#
#         x = self.embed_drop(x)
#         x = x.transpose(1, 2)
#
#         x = torch.tanh(self.conv(x).transpose(1, 2))
#         # apply attention
#         alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
#         # document representations are weighted sums using the attention. Can compute all at once as a matmul
#         m = alpha.matmul(x)
#
#         if self.use_tfidf:
#             h_tfidf = self.tfidf_hidden(tfidf).unsqueeze(1).expand(-1, self.Y, -1)
#             m = torch.cat((m, h_tfidf), dim=2)
#
#         # final layer classification
#         y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
#
#         loss = F.binary_cross_entropy_with_logits(y, target)
#
#         return y, loss
#
#     def freeze_net(self):
#         for p in self.embed.parameters():
#             p.requires_grad = False

class TFIDF(nn.Module):
    def __init__(self, args, Y):
        super(TFIDF, self).__init__()

        W = torch.Tensor(extract_wvs.load_embeddings(args.embed_file))

        self.hidden = nn.Sequential(nn.Linear(W.size()[0], args.num_filter_maps),
                                    nn.Dropout(p=args.dropout),
                                    nn.LayerNorm(args.num_filter_maps),
                                    nn.ReLU())

        # final layer: create a matrix to use for the L binary classifiers as in 2.3
        self.final = nn.Linear(args.num_filter_maps, Y)
        xavier_uniform(self.final.weight)

    def forward(self, x, target, tfidf):
        h = self.hidden(tfidf)
        y = self.final(h)

        loss = F.binary_cross_entropy_with_logits(y, target)

        return y, loss

    def freeze_net(self):
        pass

from elmoformanylangs import Embedder

class CNN(nn.Module):
    def __init__(self, args, Y, dicts):
        super(CNN, self).__init__()

        self.gpu = args.gpu

        W = torch.Tensor(extract_wvs.load_embeddings(args.embed_file))
        self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
        self.embed.weight.data = W.clone()
        self.feature_size = self.embed.embedding_dim

        self.use_position = args.use_position
        if self.use_position:
            self.embed_position = nn.Embedding(MAX_LENGTH+1, 10, padding_idx=0)
            self.feature_size += 10

        self.elmo = args.elmo
        if self.elmo is not None:
            self.elmo = Embedder(args.elmo)
            self.feature_size += self.elmo.config['encoder']['projection_dim']*2

        self.embed_drop = nn.Dropout(p=args.dropout)

        filter_size = int(args.filter_size)


        self.conv = nn.Conv1d(self.feature_size, args.num_filter_maps, kernel_size=filter_size,
                                  padding=int(floor(filter_size / 2)))
        xavier_uniform(self.conv.weight)

        # context vectors for computing attention as in 2.2
        self.U = nn.Linear(args.num_filter_maps, Y)
        xavier_uniform(self.U.weight)


        self.final = nn.Linear(args.num_filter_maps, Y)
        xavier_uniform(self.final.weight)

        self.loss_function = nn.BCEWithLogitsLoss()

    def forward(self, x, target, position, text_inputs):
        features = [self.embed(x)]

        if self.use_position:
            f = self.embed_position(position)
            features.append(f)

        if self.elmo is not None:
            with torch.no_grad():
                aa = self.elmo.sents2elmo(text_inputs)
                aa = np.array(aa)
                elmo_rep = torch.from_numpy(aa) # batch, seq_len, elmo_hidden*2
                if self.gpu >= 0 and torch.cuda.is_available():
                    elmo_rep = elmo_rep.cuda(self.gpu)
                features.append(elmo_rep)

        x = torch.cat(features, dim=2)


        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        x = torch.tanh(self.conv(x).transpose(1, 2))
        # apply attention
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
        # document representations are weighted sums using the attention. Can compute all at once as a matmul
        m = alpha.matmul(x)

        # final layer classification
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        loss = self.loss_function(y, target) # F.binary_cross_entropy_with_logits(y, target)

        return y, loss

    def freeze_net(self):
        for p in self.embed.parameters():
            p.requires_grad = False

    def set_weighted_loss(self, label_weight, Y):
        weight = torch.ones(Y)
        for label_idx in range(Y):
            if label_idx in label_weight:
                weight[label_idx] = label_weight[label_idx]
        if self.gpu >= 0 and torch.cuda.is_available():
            weight = weight.cuda(self.gpu)
        self.loss_function = nn.BCEWithLogitsLoss(pos_weight=weight)

        return

class MultiCNN(nn.Module):
    def __init__(self, args, Y, dicts):
        super(MultiCNN, self).__init__()

        self.gpu = args.gpu

        W = torch.Tensor(extract_wvs.load_embeddings(args.embed_file))
        self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
        self.embed.weight.data = W.clone()
        self.feature_size = self.embed.embedding_dim

        self.use_position = args.use_position
        if self.use_position:
            self.embed_position = nn.Embedding(MAX_LENGTH+1, 10, padding_idx=0)
            self.feature_size += 10

        self.elmo = args.elmo
        if self.elmo is not None:
            self.elmo = Embedder(args.elmo)
            self.feature_size += self.elmo.config['encoder']['projection_dim']*2

        self.embed_drop = nn.Dropout(p=args.dropout)

        if args.filter_size.find(',') == -1:
            self.filter_num = 1
            filter_size = int(args.filter_size)
            self.conv = nn.Conv1d(self.feature_size, args.num_filter_maps, kernel_size=filter_size,
                                  padding=int(floor(filter_size / 2)))
            xavier_uniform(self.conv.weight)
        else:
            filter_sizes = args.filter_size.split(',')
            self.filter_num = len(filter_sizes)
            self.conv = nn.ModuleList()
            for filter_size in filter_sizes:
                filter_size = int(filter_size)
                tmp = nn.Conv1d(self.feature_size, args.num_filter_maps, kernel_size=filter_size,
                                      padding=int(floor(filter_size / 2)))
                xavier_uniform(tmp.weight)
                self.conv.add_module('conv-{}'.format(filter_size), tmp)



        # context vectors for computing attention as in 2.2
        self.U = nn.Linear(self.filter_num*args.num_filter_maps, Y)
        xavier_uniform(self.U.weight)


        self.final = nn.Linear(self.filter_num*args.num_filter_maps, Y)
        xavier_uniform(self.final.weight)

        self.loss_function = nn.BCEWithLogitsLoss()

    def forward(self, x, target, position, text_inputs):
        features = [self.embed(x)]

        if self.use_position:
            f = self.embed_position(position)
            features.append(f)

        if self.elmo is not None:
            with torch.no_grad():
                aa = self.elmo.sents2elmo(text_inputs)
                aa = np.array(aa)
                elmo_rep = torch.from_numpy(aa) # batch, seq_len, elmo_hidden*2
                if self.gpu >= 0 and torch.cuda.is_available():
                    elmo_rep = elmo_rep.cuda(self.gpu)
                features.append(elmo_rep)

        x = torch.cat(features, dim=2)


        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        if self.filter_num == 1:
            x = torch.tanh(self.conv(x).transpose(1, 2))
        else:
            conv_result = []
            for tmp in self.conv:
                conv_result.append(torch.tanh(tmp(x).transpose(1, 2)))
            x = torch.cat(conv_result, dim=2)

        # apply attention
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
        # document representations are weighted sums using the attention. Can compute all at once as a matmul
        m = alpha.matmul(x)

        # final layer classification
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        loss = self.loss_function(y, target) # F.binary_cross_entropy_with_logits(y, target)

        return y, loss

    def freeze_net(self):
        for p in self.embed.parameters():
            p.requires_grad = False

    def set_weighted_loss(self, label_weight, Y):
        weight = torch.ones(Y)
        for label_idx in range(Y):
            if label_idx in label_weight:
                weight[label_idx] = label_weight[label_idx]
        if self.gpu >= 0 and torch.cuda.is_available():
            weight = weight.cuda(self.gpu)
        self.loss_function = nn.BCEWithLogitsLoss(pos_weight=weight)

        return


class ResCNN(nn.Module):

    def __init__(self, args, Y, dicts):
        super(ResCNN, self).__init__()

        self.gpu = args.gpu

        W = torch.Tensor(extract_wvs.load_embeddings(args.embed_file))
        self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
        self.embed.weight.data = W.clone()
        self.feature_size = self.embed.embedding_dim

        self.use_position = args.use_position
        if self.use_position:
            self.embed_position = nn.Embedding(MAX_LENGTH+1, 10, padding_idx=0)
            self.feature_size += 10

        self.elmo = args.elmo
        if self.elmo is not None:
            self.elmo = Embedder(args.elmo)
            self.feature_size += self.elmo.config['encoder']['projection_dim']*2

        self.embed_drop = nn.Dropout(p=args.dropout)

        conv_dict = {1: [self.feature_size, args.num_filter_maps],
                     2: [self.feature_size, 100, args.num_filter_maps],
                     3: [self.feature_size, 150, 100, args.num_filter_maps],
                     4: [self.feature_size, 200, 150, 100, args.num_filter_maps]
                     }

        self.conv = nn.ModuleList()
        conv_dimension = conv_dict[args.conv_layer]
        for idx in range(args.conv_layer):
            tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], int(args.filter_size), 1, True, args.dropout)
            self.conv.add_module('conv-{}'.format(idx), tmp)


        # context vectors for computing attention as in 2.2
        self.U = nn.Linear(args.num_filter_maps, Y)
        xavier_uniform(self.U.weight)

        # final layer: create a matrix to use for the L binary classifiers as in 2.3
        self.final = nn.Linear(args.num_filter_maps, Y)
        xavier_uniform(self.final.weight)

        self.loss_function = nn.BCEWithLogitsLoss()


    def forward(self, x, target, position, text_inputs):

        features = [self.embed(x)]

        if self.use_position:
            f = self.embed_position(position)
            features.append(f)

        if self.elmo is not None:
            with torch.no_grad():
                aa = self.elmo.sents2elmo(text_inputs)
                aa = np.array(aa)
                elmo_rep = torch.from_numpy(aa) # batch, seq_len, elmo_hidden*2
                if self.gpu >= 0 and torch.cuda.is_available():
                    elmo_rep = elmo_rep.cuda(self.gpu)
                features.append(elmo_rep)

        x = torch.cat(features, dim=2)

        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        for conv in self.conv:
            x = conv(x)
        x = x.transpose(1, 2)
        # apply attention
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
        # document representations are weighted sums using the attention. Can compute all at once as a matmul
        m = alpha.matmul(x)
        # final layer classification
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        loss = self.loss_function(y, target) # F.binary_cross_entropy_with_logits(y, target)

        return y, loss

    def freeze_net(self):
        for p in self.embed.parameters():
            p.requires_grad = False

    def set_weighted_loss(self, label_weight, Y):
        weight = torch.ones(Y)
        for label_idx in range(Y):
            if label_idx in label_weight:
                weight[label_idx] = label_weight[label_idx]
        if self.gpu >= 0 and torch.cuda.is_available():
            weight = weight.cuda(self.gpu)
        self.loss_function = nn.BCEWithLogitsLoss(pos_weight=weight)

        return



class MultiResCNN(nn.Module):

    def __init__(self, args, Y, dicts):
        super(MultiResCNN, self).__init__()

        self.gpu = args.gpu

        W = torch.Tensor(extract_wvs.load_embeddings(args.embed_file))
        self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
        self.embed.weight.data = W.clone()
        self.feature_size = self.embed.embedding_dim

        self.use_position = args.use_position
        if self.use_position:
            self.embed_position = nn.Embedding(MAX_LENGTH+1, 10, padding_idx=0)
            self.feature_size += 10

        self.elmo = args.elmo
        if self.elmo is not None:
            self.elmo = Embedder(args.elmo)
            self.feature_size += self.elmo.config['encoder']['projection_dim']*2

        self.embed_drop = nn.Dropout(p=args.dropout)

        self.conv = nn.ModuleList()
        filter_sizes = args.filter_size.split(',')
        conv_dict = {1: [self.feature_size, args.num_filter_maps],
                     2: [self.feature_size, 100, args.num_filter_maps],
                     3: [self.feature_size, 150, 100, args.num_filter_maps],
                     4: [self.feature_size, 200, 150, 100, args.num_filter_maps]
                     }
        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(self.feature_size, self.feature_size, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            xavier_uniform(tmp.weight)
            one_channel.add_module('baseconv', tmp)

            conv_dimension = conv_dict[args.conv_layer]
            for idx in range(args.conv_layer):
                tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True,
                                    args.dropout)
                one_channel.add_module('resconv-{}'.format(idx), tmp)

            self.conv.add_module('channel-{}'.format(filter_size), one_channel)

        # context vectors for computing attention as in 2.2
        self.U = nn.Linear(self.filter_num*args.num_filter_maps, Y)
        xavier_uniform(self.U.weight)

        # final layer: create a matrix to use for the L binary classifiers as in 2.3
        self.final = nn.Linear(self.filter_num*args.num_filter_maps, Y)
        xavier_uniform(self.final.weight)

        self.loss_function = nn.BCEWithLogitsLoss()


    def forward(self, x, target, position, text_inputs):

        features = [self.embed(x)]

        if self.use_position:
            f = self.embed_position(position)
            features.append(f)

        if self.elmo is not None:
            with torch.no_grad():
                aa = self.elmo.sents2elmo(text_inputs)
                aa = np.array(aa)
                elmo_rep = torch.from_numpy(aa) # batch, seq_len, elmo_hidden*2
                if self.gpu >= 0 and torch.cuda.is_available():
                    elmo_rep = elmo_rep.cuda(self.gpu)
                features.append(elmo_rep)

        x = torch.cat(features, dim=2)

        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)

        # apply attention
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
        # document representations are weighted sums using the attention. Can compute all at once as a matmul
        m = alpha.matmul(x)
        # final layer classification
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        loss = self.loss_function(y, target) # F.binary_cross_entropy_with_logits(y, target)

        return y, loss

    def freeze_net(self):
        for p in self.embed.parameters():
            p.requires_grad = False

    def set_weighted_loss(self, label_weight, Y):
        weight = torch.ones(Y)
        for label_idx in range(Y):
            if label_idx in label_weight:
                weight[label_idx] = label_weight[label_idx]
        if self.gpu >= 0 and torch.cuda.is_available():
            weight = weight.cuda(self.gpu)
        self.loss_function = nn.BCEWithLogitsLoss(pos_weight=weight)

        return



class VanillaConv(BaseModel):

    def __init__(self, Y, embed_file, kernel_size, num_filter_maps, gpu=-1, dicts=None, embed_size=100, dropout=0.5):
        super(VanillaConv, self).__init__(Y, embed_file, dicts, dropout=dropout, embed_size=embed_size) 
        #initialize conv layer as in 2.1
        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size)
        xavier_uniform(self.conv.weight)

        #linear output
        self.fc = nn.Linear(num_filter_maps, Y)
        xavier_uniform(self.fc.weight)

    def forward(self, x, target, desc_data=None, get_attention=False):
        #embed
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        #conv/max-pooling
        c = self.conv(x)
        if get_attention:
            #get argmax vector too
            # feili
            # x, argmax = F.max_pool1d(F.tanh(c), kernel_size=c.size()[2], return_indices=True)
            x, argmax = F.max_pool1d(torch.tanh(c), kernel_size=c.size()[2], return_indices=True)
            attn = self.construct_attention(argmax, c.size()[2])
        else:
            # feili
            # x = F.max_pool1d(F.tanh(c), kernel_size=c.size()[2])
            x = F.max_pool1d(torch.tanh(c), kernel_size=c.size()[2])
            attn = None
        x = x.squeeze(dim=2)

        #linear output
        x = self.fc(x)

        #final sigmoid to get predictions
        yhat = x
        loss = self._get_loss(yhat, target)
        return yhat, loss, attn

    def construct_attention(self, argmax, num_windows):
        attn_batches = []
        for argmax_i in argmax:
            attns = []
            for i in range(num_windows):
                #generate mask to select indices of conv features where max was i
                mask = (argmax_i == i).repeat(1,self.Y).t()
                #apply mask to every label's weight vector and take the sum to get the 'attention' score
                weights = self.fc.weight[mask].view(-1,self.Y)
                if len(weights.size()) > 0:
                    window_attns = weights.sum(dim=0)
                    attns.append(window_attns)
                else:
                    #this window was never a max
                    attns.append(Variable(torch.zeros(self.Y)).cuda())
            #combine
            attn = torch.stack(attns)
            attn_batches.append(attn)
        attn_full = torch.stack(attn_batches)
        #put it in the right form for passing to interpret
        attn_full = attn_full.transpose(1,2)
        return attn_full


class VanillaRNN(BaseModel):
    """
        General RNN - can be LSTM or GRU, uni/bi-directional
    """

    def __init__(self, Y, embed_file, dicts, rnn_dim, cell_type, num_layers, gpu, embed_size=100, bidirectional=False):
        super(VanillaRNN, self).__init__(Y, embed_file, dicts, embed_size=embed_size, gpu=gpu)
        self.gpu = gpu
        self.rnn_dim = rnn_dim
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        #recurrent unit
        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(self.embed_size, floor(self.rnn_dim/self.num_directions), self.num_layers, bidirectional=bidirectional)
        else:
            self.rnn = nn.GRU(self.embed_size, floor(self.rnn_dim/self.num_directions), self.num_layers, bidirectional=bidirectional)
        #linear output
        self.final = nn.Linear(self.rnn_dim, Y)

        #arbitrary initialization
        self.batch_size = 16
        self.hidden = self.init_hidden()

    def forward(self, x, target, desc_data=None, get_attention=False):
        #clear hidden state, reset batch size at the start of each batch
        self.refresh(x.size()[0])

        #embed
        embeds = self.embed(x).transpose(0,1)
        #apply RNN
        out, self.hidden = self.rnn(embeds, self.hidden)

        #get final hidden state in the appropriate way
        last_hidden = self.hidden[0] if self.cell_type == 'lstm' else self.hidden
        last_hidden = last_hidden[-1] if self.num_directions == 1 else last_hidden[-2:].transpose(0,1).contiguous().view(self.batch_size, -1)
        #apply linear layer and sigmoid to get predictions
        yhat = self.final(last_hidden)
        loss = self._get_loss(yhat, target)
        return yhat, loss, None

    def init_hidden(self):
        if self.gpu >= 0:
            # h_0 = Variable(torch.cuda.FloatTensor(self.num_directions*self.num_layers, self.batch_size,
            #                                       floor(self.rnn_dim/self.num_directions)).zero_())
            # if self.cell_type == 'lstm':
            #     c_0 = Variable(torch.cuda.FloatTensor(self.num_directions*self.num_layers, self.batch_size,
            #                                           floor(self.rnn_dim/self.num_directions)).zero_())
            #     return (h_0, c_0)
            # else:
            #     return h_0

            h_0 = torch.FloatTensor(self.num_directions * self.num_layers, self.batch_size,
                                                  floor(self.rnn_dim / self.num_directions)).zero_().cuda(self.gpu)
            if self.cell_type == 'lstm':
                c_0 = torch.FloatTensor(self.num_directions * self.num_layers, self.batch_size,
                                                      floor(self.rnn_dim / self.num_directions)).zero_().cuda(self.gpu)
                return (h_0, c_0)
            else:
                return h_0

        else:
            h_0 = Variable(torch.zeros(self.num_directions*self.num_layers, self.batch_size, floor(self.rnn_dim/self.num_directions)))
            if self.cell_type == 'lstm':
                c_0 = Variable(torch.zeros(self.num_directions*self.num_layers, self.batch_size, floor(self.rnn_dim/self.num_directions)))
                return (h_0, c_0)
            else:
                return h_0

    def refresh(self, batch_size):
        self.batch_size = batch_size
        self.hidden = self.init_hidden()



class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, use_res, dropout):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.Conv1d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel)
        )

        self.use_res = use_res
        if self.use_res:
            self.shortcut = nn.Sequential()
            if stride != 1 or inchannel != outchannel:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(outchannel)
                )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.left(x)
        if self.use_res:
            out += self.shortcut(x)
        # out = F.relu(out)
        out = F.tanh(out)
        out = self.dropout(out)
        return out



class MultiConvAttnPool(BaseModel):

    def __init__(self, args, Y, embed_file, kernel_size, num_filter_maps, lmbda, gpu, dicts, conv_layer, use_res, embed_size=100, dropout=0.5,
                 code_emb=None):
        super(MultiConvAttnPool, self).__init__(args, Y, embed_file, dicts, lmbda, dropout=dropout, gpu=gpu, embed_size=embed_size)

        # initialize conv layer as in 2.1
        # self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size,
        #                       padding=int(floor(kernel_size / 2)))
        # xavier_uniform(self.conv.weight)

        # conv_layer=1: 64
        # conv_layer=2: 128 64
        # conv_layer=3: 256 128 64

        if self.use_pos:
            conv_dict = {1: [self.embed_size*2, 64], 2: [self.embed_size*2, 128, 64], 3: [self.embed_size*2, 256, 128, 64]}
        else:
            conv_dict = {1: [self.embed_size, 64], 2: [self.embed_size, 128, 64], 3: [self.embed_size, 256, 128, 64]}


        self.convs = nn.ModuleList()
        conv_dimension = conv_dict[conv_layer]
        for idx in range(conv_layer):
            self.convs.append(ResidualBlock(conv_dimension[idx], conv_dimension[idx+1], int(args.filter_size), 1, use_res, dropout))


        # context vectors for computing attention as in 2.2
        self.U = nn.Linear(num_filter_maps, Y)
        xavier_uniform(self.U.weight)

        # final layer: create a matrix to use for the L binary classifiers as in 2.3
        self.final = nn.Linear(num_filter_maps, Y)
        xavier_uniform(self.final.weight)

        # initialize with trained code embeddings if applicable
        if code_emb:
            self._code_emb_init(code_emb, dicts)
            # also set conv weights to do sum of inputs
            weights = torch.eye(self.embed_size).unsqueeze(2).expand(-1, -1, kernel_size) / kernel_size
            self.conv.weight.data = weights.clone()
            self.conv.bias.data.zero_()

        # conv for label descriptions as in 2.5
        # description module has its own embedding and convolution layers
        if lmbda > 0:
            W = self.embed.weight.data
            self.desc_embedding = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.desc_embedding.weight.data = W.clone()

            self.label_conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size,
                                        padding=int(floor(kernel_size / 2)))
            xavier_uniform(self.label_conv.weight)

            self.label_fc1 = nn.Linear(num_filter_maps, num_filter_maps)
            xavier_uniform(self.label_fc1.weight)

    def _code_emb_init(self, code_emb, dicts):
        code_embs = KeyedVectors.load_word2vec_format(code_emb)
        weights = np.zeros(self.final.weight.size())
        for i in range(self.Y):
            code = dicts['ind2c'][i]
            weights[i] = code_embs[code]
        self.U.weight.data = torch.Tensor(weights).clone()
        self.final.weight.data = torch.Tensor(weights).clone()

    def forward(self, x, target, desc_data=None, get_attention=True):

        if self.use_pos:
            word, pos = x
            x = torch.cat([self.embed(word), self.pos_embed(pos)], dim=-1)
        else:
            x = self.embed(x)

        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        # apply convolution and nonlinearity (tanh)
        # feili
        # x = F.tanh(self.conv(x).transpose(1,2))
        # x = torch.tanh(self.conv(x).transpose(1, 2))
        for conv in self.convs:
            x = conv(x)
        x = x.transpose(1, 2)
        # apply attention
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
        # document representations are weighted sums using the attention. Can compute all at once as a matmul
        m = alpha.matmul(x)
        # final layer classification
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        if desc_data is not None:
            # run descriptions through description module
            b_batch = self.embed_descriptions(desc_data, self.gpu)
            # get l2 similarity loss
            diffs = self._compare_label_embeddings(target, b_batch, desc_data)
        else:
            diffs = None

        # final sigmoid to get predictions
        yhat = y
        loss = self._get_loss(yhat, target, diffs)
        return yhat, loss, alpha


class ConvAttnPool_lco(BaseModel):

    def __init__(self, args, Y, embed_file, kernel_size, num_filter_maps, lmbda, gpu, dicts, embed_size=100,
                 dropout=0.5, code_emb=None):
        super(ConvAttnPool_lco, self).__init__(args, Y, embed_file, dicts, lmbda, dropout=dropout, gpu=gpu,
                                           embed_size=embed_size)

        if self.use_pos:
            self.conv = nn.Conv1d(self.embed_size * 2, num_filter_maps, kernel_size=kernel_size,
                                  padding=int(floor(kernel_size / 2)))
        else:
            # initialize conv layer as in 2.1
            self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size,
                                  padding=int(floor(kernel_size / 2)))
        xavier_uniform(self.conv.weight)

        # context vectors for computing attention as in 2.2
        self.U = nn.Linear(num_filter_maps, Y)
        xavier_uniform(self.U.weight)

        # final layer: create a matrix to use for the L binary classifiers as in 2.3
        self.final = nn.Linear(num_filter_maps, Y, bias=False)
        xavier_uniform(self.final.weight)

        # label co-occurence
        self.W_O = nn.Linear(num_filter_maps, Y, bias=False)
        xavier_uniform(self.W_O.weight)
        self.transform = nn.Linear(num_filter_maps, num_filter_maps, bias=False)
        xavier_uniform(self.transform.weight)
        self.label_matrix = torch.from_numpy(dicts['label_matrix'])
        if gpu >= 0 and torch.cuda.is_available():
            self.label_matrix = self.label_matrix.cuda(gpu)

        self.lco_weight = 0.5

        # initialize with trained code embeddings if applicable
        if code_emb:
            self._code_emb_init(code_emb, dicts)
            # also set conv weights to do sum of inputs
            weights = torch.eye(self.embed_size).unsqueeze(2).expand(-1, -1, kernel_size) / kernel_size
            self.conv.weight.data = weights.clone()
            self.conv.bias.data.zero_()

        # conv for label descriptions as in 2.5
        # description module has its own embedding and convolution layers
        if lmbda > 0:
            W = self.embed.weight.data
            self.desc_embedding = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.desc_embedding.weight.data = W.clone()

            self.label_conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size,
                                        padding=int(floor(kernel_size / 2)))
            xavier_uniform(self.label_conv.weight)

            self.label_fc1 = nn.Linear(num_filter_maps, num_filter_maps)
            xavier_uniform(self.label_fc1.weight)

    def _code_emb_init(self, code_emb, dicts):
        code_embs = KeyedVectors.load_word2vec_format(code_emb)
        weights = np.zeros(self.final.weight.size())
        for i in range(self.Y):
            code = dicts['ind2c'][i]
            weights[i] = code_embs[code]
        self.U.weight.data = torch.Tensor(weights).clone()
        self.final.weight.data = torch.Tensor(weights).clone()

    def forward(self, x, target, desc_data=None, get_attention=True):

        if self.use_pos:
            word, pos = x
            x = torch.cat([self.embed(word), self.pos_embed(pos)], dim=-1)
        else:
            x = self.embed(x)

        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        # apply convolution and nonlinearity (tanh)
        # feili
        # x = F.tanh(self.conv(x).transpose(1,2))
        x = torch.tanh(self.conv(x).transpose(1, 2))
        # apply attention
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
        # document representations are weighted sums using the attention. Can compute all at once as a matmul
        m = alpha.matmul(x)
        # final layer classification
        # y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        y_1 = self.final.weight.mul(m).sum(dim=2)

        label_vectors = self.transform(self.label_matrix.mm(self.W_O.weight) / self.label_matrix.sum(1, keepdim=True))
        y_2 = label_vectors.mul(m).sum(dim=2)

        y = self.lco_weight * y_2 + (1-self.lco_weight) * y_1

        if desc_data is not None:
            # run descriptions through description module
            b_batch = self.embed_descriptions(desc_data, self.gpu)
            # get l2 similarity loss
            diffs = self._compare_label_embeddings(target, b_batch, desc_data)
        else:
            diffs = None

        # final sigmoid to get predictions
        yhat = y
        loss = self._get_loss(yhat, target, diffs)
        return yhat, loss, alpha
