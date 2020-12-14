# -*- coding: utf-8 -*- 

'''
Author: ByronVon
Email: wangy@craiditx.com
Version: 
Date: 2020-12-11 17:13:27
LastEditTime: 2020-12-14 17:06:52
'''

import torch
import psutil
import shutil
import onnxruntime
import numpy as np
from .model import SentenceBertModel, SentenceBertEncoder, Tokenizer


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def conv(
    pretrain_model_path=None,
    finetune_model_path=None,
    onnx_model_path=None,
    ):
    """torch转onnx格式"""
    ## 先加载原分类模型，再保存pretrain部分，加载pretrain部分进行转换             
    model = SentenceBertModel(pretrain_model_path)
    model.load_state_dict(torch.load(finetune_model_path, map_location=device))
    model.bert.save_pretrained('tmp')
    model = SentenceBertEncoder('tmp')
    model.to(device)
    model.eval()
    ## 删除缓存文件
    shutil.rmtree('tmp') 
    ## 构建一个dummy_inputs
    tokenizer = Tokenizer(pretrain_model_path).tokenizer
    tmp = tokenizer.encode_plus(
        text='人生苦短，我用python',
        truncation=True,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        return_attention_mask=True,
    )
    inputs = {'word_seq_tensor': torch.tensor([tmp['input_ids']], dtype=torch.long).to(device)}
    ## torch-->onnx
    with torch.no_grad():
        torch.onnx.export(
            model,
            args=tuple(inputs.values()),
            f=onnx_model_path,
            opset_version=11,
            do_constant_folding=True,
            input_names=['word_seq_tensor'],
            output_names=['sentence_vector']
        )
    print(f'ONNX Model Exported at {onnx_model_path}')


def infer(
    pretrain_model_path,
    onnx_model_path,
    sentences,
    ):
    """使用onnx加速推理过程"""
    ## 构建计算图
    sess_option = onnxruntime.SessionOptions()
    sess_option.intra_op_num_threads = psutil.cpu_count(logical=True)
    session = onnxruntime.InferenceSession(onnx_model_path)

    ## 初始化tokenizer
    tokenizer = Tokenizer(pretrain_model_path).tokenizer
    tmp = tokenizer.encode_plus(
        text=sentences,
        truncation=True,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        return_attention_mask=True,
    )
    ## TODO: 改成批量输入

    inputs = {'word_seq_tensor': np.array([tmp['input_ids']])}

    res = session.run(None, inputs)

    return res