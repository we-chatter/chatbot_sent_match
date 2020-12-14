# -*- coding: utf-8 -*- 

'''
Author: ByronVon
Email: wangy@craiditx.com
Version: 
Date: 2020-12-11 17:16:26
LastEditTime: 2020-12-14 11:59:00
'''

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup, AdamW

from .utils import Log
from . import train_data_path, valid_data_path, test_data_path
from .model import SentenceBertModel, SentenceBertEncoder, tokenizer

logger = Log().Logger


def load_data(filename, sp='\t'):
    """读取原始文本"""
    texts, labels = [], []
    with open(filename, encoding='utf-8') as f:
        for l in f.readlines():
            cont = l.strip().split(sep=sp)
            if len(cont) == 3: ## 长度为3表示（sent_a,sent_b,label）这个三元组
                text0, text1, label = cont[0], cont[1], cont[2]
            else:
                text0, text1, label = cont[0], cont[1], -1
            texts.append((text0, text1))
            labels.append(int(label))
    
    return texts, labels
    

class DatasetIterater(Dataset):
    """pytorch的数据生成器，每次取一条数据"""
    def __init__(self, text, label):
        super(DatasetIterater, self).__init__()
        self.text = text
        self.label = label

    def __getitem__(self, item):
        return self.text[item], self.label[item]

    def __len__(self):
        return len(self.text)


def collate_fn(batch_data):
    """将每条数据处理成指定格式，每次会读取一批数据"""
    batch_token_ids_0, batch_token_ids_1, batch_labels = [], [], []

    for data, label in batch_data:
        encodes_0 = tokenizer.encode_plus(
            text=data[0],
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_attention_mask=False, ## 训练时没有用到mask,就不返回了
            )
        encodes_1 = tokenizer.encode_plus(
            text=data[0],
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_attention_mask=False, ## 训练时没有用到mask,就不返回了
            )
        batch_token_ids_0.append(encodes_0['input_ids'])
        batch_token_ids_1.append(encodes_1['input_ids'])
        batch_labels.append([label])

    return batch_token_ids_0, batch_token_ids_1, batch_labels


def dataloader(data_path, batch_size, shuffle=True, fn=collate_fn):
    """自定义的数据生成器"""
    data = load_data(data_path)
    data = DatasetIterater(data[0], data[1])
    gens = DataLoader(dataset=data,batch_size=batch_size,shuffle=shuffle,collate_fn=fn)
    return gens


def train(
    use_gpu, 
    pretrained_model_path, 
    encode_path, 
    weight_path, 
    epochs=10, 
    batch_size=16, 
    train_data=train_data_path, 
    valid_data=valid_data_path, 
    shuffle=True, 
    fn=collate_fn,):
    """训练模型部分函数"""
    ## 初始化模型
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    model = SentenceBertModel(pretrained_model_path)
    model.to(device)
    logger.debug(f'model initializerd')
    ## 加载数据集
    train_generator = dataloader(train_data, batch_size=batch_size, shuffle=shuffle, fn=fn)
    valid_generator = dataloader(valid_data, batch_size=batch_size, shuffle=shuffle, fn=fn)

    ## 初始化优化器
    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=len(train_generator)*epochs)

    ## 训练部分
    best_valid_acc, iters = 0.0, 0
    for _ in range(epochs):
        index = 0
        train_loss, valid_loss = 0.0, 0.0
        train_true, train_pred, valid_true, valid_pred = [], [], [], []
        
        model.train()
        for train in train_generator:
            optimizer.zero_grad() ## 每个epoch执行完毕后，梯度清零
            iters += 1
            index += 1
            
            input_ids_0 = torch.tensor(train[0],dtype=torch.long).to(device)
            input_ids_1 = torch.tensor(train[1],dtype=torch.long).to(device)
            label_ids = torch.tensor(train[2],dtype=torch.float).to(device)
            
            pred, loss = model(input_ids_0,input_ids_1,label_ids)
            
            train_loss += loss
            
            train_true.append(label_ids.cpu().float())
            train_pred.append(pred.cpu().float())
            ## 评估函数部分
            if index%200 == 0: ## 由于数据量比较大，每200轮输出下训练集结果
                model.eval()
                for valid in valid_generator:
                    input_ids_0_ = torch.tensor(valid[0],dtype=torch.long).to(device)
                    input_ids_1_ = torch.tensor(valid[1],dtype=torch.long).to(device)
                    label_ids_ = torch.tensor(valid[2],dtype=torch.float).to(device)
                    with torch.no_grad():
                        pred_, loss_ = model(input_ids_0_,input_ids_1_,label_ids_)
                        valid_loss += loss_
                        valid_true.append(label_ids_.cpu().float())
                        valid_pred.append(pred_.cpu().float())
                ## 将预测结果转换成array数组形式，(1,len(data)*batch_size）
                train_true_ = torch.cat(train_true).detach().numpy()
                train_pred_ = torch.cat(train_pred).detach().numpy() > 0.5
                valid_true_ = torch.cat(valid_true).detach().numpy()
                valid_pred_ = torch.cat(valid_pred).detach().numpy() > 0.5
                # print(train_true_.astype(int), train_pred_.astype(int))
                train_acc = evaluate(train_true_.astype(int), train_pred_.astype(int))
                valid_acc = evaluate(valid_true_.astype(int), valid_pred_.astype(int))

                ## 保存每轮验证集上的最优参数
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    # 仅保存和加载模型参数(推荐使用) ## 保存全部模型 包括最后的dense层
                    torch.save(model.state_dict(), weight_path)
                    if encode_path is not None:
                        model.bert.save_pretrained(encode_path) ## 保留bert部分模型的参数
                logger.debug(f'iteration: {iters}\ntrain_acc:{train_acc}  train_loss:{train_loss/100}\nvalid_acc:{valid_acc}  valid_loss:{valid_loss/len(valid_generator)}\n')
                # print(f'epoch: {epoch}\ntrain_acc:{train_acc}  train_loss:{loss}\nvalid_acc:{valid_acc}  valid_loss:{loss_}\n')
                train_loss = 0 ## 这里将train_loss清0，实际显示的是每500轮的train_loss均值和val_loss均值
                valid_loss = 0
                
            loss.backward()
            optimizer.step()
            scheduler.step()


def test(
    use_gpu,
    pretrained_model_path, 
    weight_path, 
    batch_size=16, 
    test_data=test_data_path, 
    shuffle=False, 
    fn=collate_fn):
    """测试函数"""
    ## 加载模型
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    model = SentenceBertModel(pretrained_model_path)
    # model = SentenceBertModel()
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    logger.debug('model initialized')

    test_generator = dataloader(test_data, batch_size=batch_size, shuffle=shuffle, fn=fn)

    test_true, test_pred = [], []
    for test in test_generator:
        input_ids = torch.tensor(test[0],dtype=torch.long).to(device)
        types_ids = torch.tensor(test[1],dtype=torch.long).to(device)
        label_ids = torch.tensor(test[2],dtype=torch.float).to(device)

        with torch.no_grad():
            pred, _ = model(input_ids, types_ids, label_ids)
            test_true.append(label_ids.cpu().float())
            test_pred.append(pred.cpu().float())
            
    test_true_ = torch.cat(test_true).detach().numpy()
    test_pred_ = torch.cat(test_pred).detach().numpy() > 0.5
    test_acc = evaluate(test_true_.astype(int), test_pred_.astype(int))

    return test_true_, test_pred_, test_acc


def encoder(
    use_gpu, 
    pretrained_model_path, 
    sentences):
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    model = SentenceBertEncoder(pretrained_model_path)
    model.to(device)
    model.eval()
    logger.debug('model initialized')

    if isinstance(sentences, list):
        ## 使用None和'-1'构成一个三元组以复用collate_fn函数，进行批量数据编码，
        sentences = list(zip(sentences, [None]*len(sentences), [-1]*len(sentences)))
    elif isinstance(sentences, None, str):
        sentences = [[sentences], [None], [-1]]
    token_ids = collate_fn(sentences)[0] ## 只拿第一组
    token_ids = torch.tensor(token_ids, dtype=torch.long).to(device)
    
    rest = []
    with torch.no_grad():
        for token in token_ids:
            _ = model(token).cpu().detach().numpy()
            rest.append(_)

    return rest


def evaluate(true_labels, pred_labels): ## 输入应当为array数组
    """评估函数"""
    total, right = 0, 0
    for true, pred in zip(true_labels, pred_labels):
        total = len(true_labels)
        right += (pred == true).sum()
    return right/total