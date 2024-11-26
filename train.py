#coding=utf-8
import torch
import os
from transformers import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from transformers import logging

logging.set_verbosity_warning()

# 超参数
episode = 5
batch_size = 2
max_len = 32
learning_rate = 2e-5
weight_decay = 1e-2
num_labels = 2
dropout_prop = 0.3
tbdir = 'Logs'

# 数据集
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=512):
        self.data = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # mx 训练集处理
                # if line[0] == '\n':
                #     continue
                # text, label = line.strip().split('==')
                # self.data.append(text)
                # self.labels.append([1, 0] if int(label) == 0 else [0, 1])
                
                # rc、ad 训练集处理
                if line[0] == '\n':
                    continue
                self.labels.append([1, 0] if int(line[0]) == 0 else [0, 1])
                self.data.append((line[2:]).strip())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length = True,
            return_token_type_ids=True,
            truncation=True,
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float32)
        }


# 首先确认我们的模型运行在 GPU 上
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 获取 Bert 的 Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-chinese')
# tokenizer = BertTokenizer('bert-chinese\\vocab.txt')
# 获取 Bert 的模型结构
config = BertConfig.from_pretrained("bert-chinese", num_labels=num_labels, hidden_dropout_prob=dropout_prop)
model = BertForSequenceClassification.from_pretrained("bert-chinese", config=config).to(device)
print('模型结构:', model)
# 创建 Dataset 和 DataLoader
train_dataset = TextDataset('dataset\\ad\\train.txt', tokenizer, max_len=max_len)
test_dataset = TextDataset('dataset\\ad\\test.txt', tokenizer, max_len=max_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# 创建损失函数和迭代器
# 设置 bias 和 LayerNorm.weight 不使用 weight_decay
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()
# 创建 Writer
if tbdir not in os.listdir('.'):
    os.mkdir(tbdir)
else:
    if len(os.listdir(tbdir)) > 0:
        for file in os.listdir(tbdir):
            os.remove(os.path.join(tbdir, file))
writer = SummaryWriter(tbdir)

if __name__ == "__main__":
    for epoch in range(episode):
        model.train()
        optimizer.zero_grad()
        
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}') as tbar:
            for step, batch in enumerate(tbar, start=0):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = (batch['label'].reshape(-1, 2)).to(device)

                outputs = model(input_ids, attention_mask, token_type_ids)
                # print('\nOutput:', outputs.logits, ', Labels:', labels)
                loss = criterion(outputs.logits, labels)

                loss.backward()
                optimizer.step()
                
                tbar.set_postfix(loss=loss.item())
                writer.add_scalar("train_loss", loss.item(), epoch * len(train_loader) + step + 1)

    model.eval()
    # 存储真实标签和预测标签
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        with tqdm(test_loader, desc='Test') as tbar:
            for batch in tbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask, token_type_ids)

                all_labels.extend(torch.max(labels, 1)[1].cpu().numpy())
                all_predictions.extend(torch.max(outputs.logits, 1)[1].cpu().numpy())
    
    # 计算 F1 分数
    f1score = f1_score(all_labels, all_predictions)
    print(f"\nF1-Score: {f1score}")

    # 计算准确率
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"准确率: {accuracy * 100:.2f}%")
    
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    print(conf_matrix)
    
    if f1score == 0:
        raise RuntimeError('训练失败, 不进行测试')
    
    while True:
        text = input("输入要判断的内容: ")
        if text == '' or text == 'q':
            print('退出最终人工测试')
            break
        
        inputs = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length = True,
            return_token_type_ids=True,
            truncation=True,
        )

        # 将编码后的输入转换为张量，并移动到 GPU 上
        input_ids = torch.tensor(inputs['input_ids']).unsqueeze(0).to(device)
        attention_mask = torch.tensor(inputs['attention_mask']).unsqueeze(0).to(device)
        token_type_ids = torch.tensor(inputs['token_type_ids']).unsqueeze(0).to(device)

        # 将输入传递给模型，并获取预测结果
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, token_type_ids)
            _, predicted = torch.max(outputs.logits, 1)

        # 打印预测结果
        if predicted.item() == 0:
            print("预测结果: 正常")
        else:
            print("预测结果: 不正常")