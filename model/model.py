# -*- coding: utf-8 -*- 

'''
Author: ByronVon
Email: sir.housir@gmail.com
Version: 
Date: 2020-12-11 17:15:27
LastEditTime: 2020-12-30 14:38:28
'''

import torch
from transformers import BertModel, BertTokenizer


class Tokenizer(object):
    """自定义的tokenizer类"""
    def __init__(self, path) -> None:
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(path)


class SentenceBertModel(torch.nn.Module):
    """
    Sentence-Bert模型，这里采用[CLS] token做代表句子向量，注意有部分ONNX不支持所有的op，当op不存在时则需要自己定义
    e.g：ONNX不支持model中的BCELoss()，而loss部分在推理时不需要，因此可以略去这部分构建一个纯推理的model再转
    TODO: 尝试不同的pooling策略
    """
    def __init__(self, model_path=None):
        super(SentenceBertModel, self).__init__()
        
        self.bert = BertModel.from_pretrained(model_path)
        self.dense = torch.nn.Linear(self.bert.config.hidden_size*3, 1)
        self.dropout = torch.nn.Dropout(0.1)
        self.sigmoid = torch.nn.Sigmoid()
        self.loss = torch.nn.BCELoss()

        torch.nn.init.xavier_uniform_(self.dense.weight) ## 初始化分类器层

    def forward(self, word_seq_tensor_0=None, word_seq_tensor_1=None, labels=None):
        ## 句子1/2编码
        cls_hidden_state_0 = self.bert(input_ids=word_seq_tensor_0)[1]
        cls_hidden_state_1 = self.bert(input_ids=word_seq_tensor_1)[1]
        
        outputs_0 = self.dropout(cls_hidden_state_0)
        outputs_1 = self.dropout(cls_hidden_state_1)
        ## 使用三元组作为输出dense层的输入特征
        outputs_x = torch.cat((outputs_0, outputs_1, torch.abs(outputs_0-outputs_1)), dim=1)
        
        logits = self.sigmoid(self.dense(outputs_x)) ## [batch,num_tags]
        
        outputs = (logits,)

        if labels is not None:
            loss = self.loss(logits, labels)
            outputs = outputs + (loss,)
        else:
            outputs = outputs + (None,)
        return outputs


class SentenceBertEncoder(torch.nn.Module):
    """SentenceBert的句子编码器"""
    def __init__(self, model_path=None):
        super(SentenceBertEncoder, self).__init__()
        
        self.bert = BertModel.from_pretrained(model_path)

    def forward(self, word_seq_tensor=None):
        cls_hidden_state = self.bert(input_ids=word_seq_tensor)[1]
        return cls_hidden_state



