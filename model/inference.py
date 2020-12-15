# -*- coding: utf-8 -*- 

'''
Author: ByronVon
Email: wangy@craiditx.com
Version: 
Date: 2020-12-11 17:13:27
LastEditTime: 2020-12-15 18:47:12
'''

from numpy.lib.function_base import gradient
import onnx
import torch
import psutil
import shutil
import onnxruntime
import numpy as np
from .model import SentenceBertModel, SentenceBertEncoder, Tokenizer


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_inputs(pretrain_model_path, sentences):
    """将句子/句子组变成向量"""
    if isinstance(sentences, str):
        sentences = [sentences]
    elif isinstance(sentences, list):
        sentences = sentences
    else:
        raise Exception('Wrong Type For Sentences!')
    
    tokenizer = Tokenizer(pretrain_model_path).tokenizer
    
    tmp = []
    for sent in sentences:
        _ = tokenizer.encode_plus(
            text=sent,
            truncation=True,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            return_attention_mask=True,
            )
        tmp.append(_['input_ids'])
    return tmp


def conv(
    pretrain_model_path=None,
    finetune_model_path=None,
    onnx_model_path=None,
    batch_inference=False, ## 支持批量计算
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
    # tokenizer = Tokenizer(pretrain_model_path).tokenizer
    # tmp = tokenizer.encode_plus(
    #     text='人生苦短，我用python',
    #     truncation=True,
    #     add_special_tokens=True,
    #     max_length=128,
    #     padding='max_length',
    #     return_attention_mask=True,
    # )
    tmp = get_inputs(pretrain_model_path, '人生苦短，我用Python!')
    inputs = {'word_seq_tensor': torch.tensor(tmp, dtype=torch.long).to(device)}
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
    ## 支持批量推理时
    if batch_inference:
        mp = onnx.load_model(onnx_model_path)
        mp.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None' ## 将dummy_inputs中的第一个维度设置为"None"
        onnx.save(mp, onnx_model_path)
    
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

    # ## 初始化tokenizer
    # tokenizer = Tokenizer(pretrain_model_path).tokenizer
    
    # tmp = tokenizer.encode_plus(
    #     text=sentences,
    #     truncation=True,
    #     add_special_tokens=True,
    #     max_length=128,
    #     padding='max_length',
    #     return_attention_mask=True,
    # )
    tmp = get_inputs(pretrain_model_path, sentences)
    ## TODO: 改成批量输入

    inputs = {'word_seq_tensor': np.array(tmp)}

    res = session.run(None, inputs)

    return res