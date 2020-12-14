# 说明  

使用sentence-bert在LCQMC上微调reberta-wwm-ext，主要用于句子相似度匹配  
使用onnx框架加速推理

## 微调后的参数文件 
 
链接: https://pan.baidu.com/s/14uvJNTAgiTzd70xm9-SD5g  密码: s1bt  

## 句子转向量  

```python
run_inference(sys.argv[1:])  

"""
tensor([0.8946])  
tensor([0.8852])  
tensor([0.9786])  
""" 
```

## 模型格式转换

将在当前目录下生成onnx格式的模型文件  

```python
run_convert(sys.argv[1:])  
```

## 微调模型

```python
run_train(sys.argv[1:])  

"""  
2020-12-14 17:03:05,378--root--finetune.py--DEBUG--model initializerd  
10%| ████████████                                 | 1/10 [01:45<15:53, 105.96s/it]  
"""  
```

## 评估模型

```python
run_test(sys.argv[1:])  

"""  
2020-12-14 17:00:57,014--root--finetune.py--DEBUG--model initialized  
100%|█████████████████████████████████████████████| 7/7 [00:24<00:00,  3.49s/it]  
0.89  
"""  
```  
