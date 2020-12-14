# -*- coding: utf-8 -*- 

'''
Author: ByronVon
Email: wangy@craiditx.com
Version: 
Date: 2020-12-11 17:22:23
LastEditTime: 2020-12-14 12:29:04
'''

import sys
import argparse

import torch


def run_train(args):
    """
    use_gpu,
    pretrained_model_path,
    encode_path,
    weight_path, 
    # 下面这些参数已经调好了，测试的时候可以直接用
    # train_data, 
    # valid_data, 
    # epochs, 
    # batch_size, 
    # shuffle=True, 
    # fn=collate_fn

    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--use_gpu', type=bool, help='use gpu or not', default=True)
    parser.add_argument('--pretrained_model_path', type=str, help='pretrained model path', default='/Users/wonbyron/Desktop/work/NLP-Demo/module/sent_match/chinese_roberta_wwm_ext_pytorch')
    parser.add_argument('--encode_path', type=str, help='save finetuned bert encoder', default='encode')
    parser.add_argument('--weight_path', type=str, help='weight path', default='tmp.bin')
    # parser.add_argument('--train_data_path', type=str, help='train_data_path', default=None)
    # parser.add_argument('--valid_data_path', type=str, help='valid_data_path', default=None)
    # parser.add_argument('--epochs', type=int, help='epochs for finetune', default=10)
    # parser.add_argument('--batch_size', type=int, help='batch_size for finetune', default=16)

    args = parser.parse_args() ## 解析配置参数 ## args.epochs 就可以调用结果

    from sent_match import finetune

    finetune.train(
        use_gpu=args.use_gpu, 
        pretrained_model_path=args.pretrained_model_path,
        encode_path=args.encode_path,
        weight_path=args.weight_path,
        )
    

def run_test(args):
    """
    use_gpu,
    weight_path, 
    ## 参数已经设置好了
    # test_data, 
    # batch_size, 
    # shuffle=False, 
    # fn=collate_fn
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--use_gpu', type=bool, help='use gpu or not', default=False)
    parser.add_argument('--pretrained_model_path', type=str, help='pretrained model path', default='/Users/wonbyron/Desktop/work/NLP-Demo/module/sent_match/chinese_roberta_wwm_ext_pytorch')
    parser.add_argument('--weight_path', type=str, help='finetuned model path', default='/Users/wonbyron/Desktop/work/NLP-Demo/tmp/smodel/smodel.bin')
    # parser.add_argument('--test_data_path', type=str, help='test_data_path', default=None)   
    # parser.add_argument('--batch_size', type=int, help='batch_size for finetune', default=16)

    args = parser.parse_args()

    from sent_match import finetune

    print(
        finetune.test(
        use_gpu=args.use_gpu,
        pretrained_model_path=args.pretrained_model_path,
        weight_path=args.weight_path,)
    )
    

def run_convert(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--use_gpu', type=bool, help='use gpu or not', default=False)
    parser.add_argument('--pretrained_model_path', type=str, help='pretrained model path', default='/Users/wonbyron/Desktop/work/NLP-Demo/tmp/smodel_vector') ## 
    parser.add_argument('--weight_path', type=str, help='finetuned weight path', default=None)
    parser.add_argument('--onnx_model_path', type=str, help='onnx export model path', default='sent_encoder.onnx')

    from sent_match.inference import conv
    args = parser.parse_args()
    
    conv(
        use_gpu=args.use_gpu, 
        pretrained_model_path=args.pretrained_model_path,
        weight_path=args.weight_path,
        onnx_model_path=args.onnx_model_path)


def run_inference(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--onnx_model_path', type=str, help='onnx_model_path', default='/Users/wonbyron/Desktop/work/NLP-Demo/module/sent_encoder.onnx')
    parser.add_argument('--sentences', type=str, help='sentences', default='我爱我家')

    args = parser.parse_args()

    import torch
    from sent_match.inference import infer
    
    sent_a = torch.tensor(infer(args.onnx_model_path, '你是问e租宝理财安全码吗？')[0])
    sent_b = torch.tensor(infer(args.onnx_model_path, '你是问e租包理财安全吗')[0])
    sent_c = torch.tensor(infer(args.onnx_model_path, 'e租包理财安全吗？')[0])

    print(torch.cosine_similarity(sent_a, sent_b))
    print(torch.cosine_similarity(sent_a, sent_c))
    print(torch.cosine_similarity(sent_b, sent_c))

if __name__ == "__main__":

    # run_train(sys.argv[1:]) ## 测试train, 默认不开启GPU
    # run_test(sys.argv[1:]) ## 测试训练
    # run_convert(sys.argv[1:]) ## 测试模型转换
    run_inference(sys.argv[1:]) ## 测试infer