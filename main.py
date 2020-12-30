# -*- coding: utf-8 -*- 

'''
Author: ByronVon
Email: sir.housir@gmail.com
Version: 
Date: 2020-12-11 17:22:23
LastEditTime: 2020-12-30 14:37:34
'''

import sys
import argparse


def run_train(args):
    """
    目前支持的参数
    pretrain_model_path, 
    finetune_model_path, 
    train_data=BasicConfig.train_data_path, 
    valid_data=BasicConfig.valid_data_path, 
    epochs=BasicConfig.epochs, 
    batch_size=BasicConfig.batch_size, 
    lr = BasicConfig.learning_rate,
    shuffle=True
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrain_model_path', type=str, help='pretrain model path', default='./chinese_roberta_wwm_ext_pytorch/')
    parser.add_argument('--finetune_model_path', type=str, help='finetune model path', default='tmp.bin')

    args = parser.parse_args() ## 解析配置参数 ## args.epochs 就可以调用结果

    from model import finetune

    finetune.train(
        pretrain_model_path=args.pretrain_model_path,
        finetune_model_path=args.finetune_model_path,
        )
    """
    2020-12-14 17:03:05,378--root--finetune.py--DEBUG--model initializerd
    10%|████████████                                                                                                            | 1/10 [01:45<15:53, 105.96s/it]
    """
    

def run_test(args):
    """
    pretrain_model_path, 
    finetune_model_path, 
    test_data=BasicConfig.test_data_path, 
    batch_size=BasicConfig.batch_size, 
    shuffle=False, 
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrain_model_path', type=str, help='pretrain model path', default='./chinese_roberta_wwm_ext_pytorch/')
    parser.add_argument('--finetune_model_path', type=str, help='finetune model path', default='model.bin')

    args = parser.parse_args()

    from model import finetune

    print(
        finetune.test(
        pretrain_model_path=args.pretrain_model_path,
        finetune_model_path=args.finetune_model_path,
        )
    )
    """
    2020-12-14 17:00:57,014--root--finetune.py--DEBUG--model initialized
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:24<00:00,  3.49s/it]
    0.89
    """

def run_convert(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrain_model_path', type=str, help='pretrain model path', default='./chinese_roberta_wwm_ext_pytorch') ## 
    parser.add_argument('--finetune_model_path', type=str, help='finetune weight path', default='./model.bin')
    parser.add_argument('--onnx_model_path', type=str, help='onnx export model path', default='./tmp.onnx')
    parser.add_argument('--batch_inference', type=bool, help='enable batch inference', default=True)

    args = parser.parse_args()
    
    from model.inference import conv

    conv(
        pretrain_model_path=args.pretrain_model_path,
        finetune_model_path=args.finetune_model_path,
        onnx_model_path=args.onnx_model_path,
        batch_inference=args.batch_inference,
        )


def run_inference(args):
    # parser = argparse.ArgumentParser()

    # parser.add_argument('--onnx_model_path', type=str, help='onnx model path', default='./tmp.onnx')
    # parser.add_argument('--sentences', type=str, help='sentences', default='我爱我家')

    # args = parser.parse_args()

    import torch
    from model.inference import infer
    
    sent_a = torch.tensor(infer('./chinese_roberta_wwm_ext_pytorch', './tmp.onnx', '你是问e租宝理财安全码吗？')[0])
    sent_b = torch.tensor(infer('./chinese_roberta_wwm_ext_pytorch', './tmp.onnx', '你是问e租包理财安全吗')[0])
    sent_c = torch.tensor(infer('./chinese_roberta_wwm_ext_pytorch', './tmp.onnx', 'e租包理财安全吗？')[0])

    sent_d = torch.tensor(infer('./chinese_roberta_wwm_ext_pytorch', './tmp.onnx', ['你是问e租宝理财安全码吗？', '你是问e租包理财安全吗', 'e租包理财安全吗？']))

    print(torch.cosine_similarity(sent_a, sent_b))
    print(torch.cosine_similarity(sent_a, sent_c))
    print(torch.cosine_similarity(sent_b, sent_c))

    print(sent_d)

    """
    tensor([0.8946])
    tensor([0.8852])
    tensor([0.9786])
    """

if __name__ == "__main__":

    # run_train(sys.argv[1:]) ## 测试train, 默认不开启GPU
    # run_test(sys.argv[1:]) ## 测试训练
    # run_convert(sys.argv[1:]) ## 测试模型转换
    run_inference(sys.argv[1:]) ## 测试infer

    # from sent_match.model import SentenceBertModel
    # from sent_match.finetune import collate_fn

    # device = torch.device('cpu')
    # model = SentenceBertModel(model_path='/Users/wonbyron/bert/chinese_roberta_wwm_ext_pytorch/')
    # model.load_state_dict(torch.load('/Users/wonbyron/Desktop/work/NLP-Demo/tmp/smodel/smodel.bin', map_location=device))
    # model.to(device)
    # model.eval()

    # input_ids_0, input_ids_1, labels = collate_fn([[('哪种减肥药最快最有效', '减肥效果最好'),0]])

    # print(input_ids_0)
    # print(input_ids_1)
    # with torch.no_grad():
    #     pred, loss = model(
    #         torch.tensor(input_ids_0,dtype=torch.long).to(device),
    #         torch.tensor(input_ids_1,dtype=torch.long).to(device)
    #     )
    # # print(model.eval())
    # print(pred, loss)
