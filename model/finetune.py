# -*- coding: utf-8 -*- 

'''
Author: ByronVon
Email: wangy@craiditx.com
Version: 
Date: 2020-12-11 17:16:26
LastEditTime: 2020-12-14 17:00:27
'''

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup, AdamW

from . import config
from .model import SentenceBertModel, Tokenizer


tokenizer = Tokenizer(config.pretrain_model_path).tokenizer
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_data(filename):
    """读取原始文本"""
    texts, labels = [], []
    with open(filename, encoding='utf-8') as f:
        for l in f.readlines():
            cont = l.strip().split('\t')
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
            text=data[1],
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
    pretrain_model_path, 
    finetune_model_path, 
    train_data=config.train_data_path, 
    valid_data=config.valid_data_path, 
    epochs=config.epochs, 
    batch_size=config.batch_size, 
    lr = config.learning_rate,
    shuffle=True, 
    ):
    """训练模型部分函数"""
    ## 初始化模型
    model = SentenceBertModel(model_path=pretrain_model_path)
    model.to(device)
    config.logger.debug(f'model initializerd')
    ## 加载数据集
    train_generator = dataloader(train_data, batch_size=batch_size, shuffle=shuffle, fn=collate_fn)
    valid_generator = dataloader(valid_data, batch_size=batch_size, shuffle=shuffle, fn=collate_fn)

    ## 初始化优化器
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=len(train_generator)*epochs)

    ## 训练部分
    best_valid_acc, iters = 0.0, 0
    for _ in tqdm(range(epochs)):
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
            
            pred, loss = model(input_ids_0, input_ids_1, label_ids)
            
            train_loss += loss
            
            train_true.append(label_ids.cpu().float())
            train_pred.append(pred.cpu().float())
            ## 评估函数部分
            if index%200==0 or index>len(train_generator)*batch_size: ## 由于数据量比较大，每200轮输出下训练集结果
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
                    torch.save(model.state_dict(), finetune_model_path)

                config.logger.debug(f'iteration: {iters}\ntrain_acc:{train_acc}  train_loss:{train_loss/100}\nvalid_acc:{valid_acc}  valid_loss:{valid_loss/len(valid_generator)}\n')
                # print(f'epoch: {epoch}\ntrain_acc:{train_acc}  train_loss:{loss}\nvalid_acc:{valid_acc}  valid_loss:{loss_}\n')
                train_loss = 0 ## 这里将train_loss清0，实际显示的是每500轮的train_loss均值和val_loss均值
                valid_loss = 0
                
            loss.backward()
            optimizer.step()
            scheduler.step()


def test(
    pretrain_model_path, 
    finetune_model_path, 
    test_data=config.test_data_path, 
    batch_size=config.batch_size, 
    shuffle=False, 
    ):
    """测试函数"""
    ## 加载模型
    model = SentenceBertModel(model_path=pretrain_model_path)
    model.load_state_dict(torch.load(finetune_model_path, map_location=device))
    model.to(device)
    model.eval()
    
    config.logger.debug('model initialized')

    test_generator = dataloader(test_data, batch_size=batch_size, shuffle=shuffle, fn=collate_fn)

    test_true, test_pred = [], []
    for test in tqdm(test_generator):
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

    return test_acc


def evaluate(true_labels, pred_labels): ## 输入应当为array数组
    """评估函数"""
    total, right = 0, 0
    for true, pred in zip(true_labels, pred_labels):
        total = len(true_labels)
        right += (pred == true).sum()
    return right/total
