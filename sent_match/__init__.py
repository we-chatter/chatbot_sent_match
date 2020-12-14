# -*- coding: utf-8 -*- 

'''
Author: ByronVon
Email: wangy@craiditx.com
Version: 
Date: 2020-12-11 17:15:50
LastEditTime: 2020-12-14 11:11:40
'''
## TODO 将配置项写到config中
import os

dir = os.path.dirname(os.path.abspath(__file__))


pretrained_model_path = os.path.join(dir, 'chinese_roberta_wwm_ext_pytorch')
train_data_path = os.path.join(dir, 'dataset/train.txt')
valid_data_path = os.path.join(dir, 'dataset/dev.txt')
test_data_path = os.path.join(dir, 'dataset/test_.txt')