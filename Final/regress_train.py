import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from utils import get_current_time
import os
import time

from transformers import (
  BertTokenizerFast,
  AutoModel,
)

model_root = './models/'
model_type_root = './models/text_regress'
logs_root = './logs/'
data_root = './data/' #自己建這個資料夾，把jsonl資料都丟進去

if not os.path.isdir(model_root):
    os.makedirs(model_root)
if not os.path.isdir(model_type_root):
    os.makedirs(model_type_root)
if not os.path.isdir(logs_root):
    os.makedirs(logs_root)

feature_size = 768
batch_size = 16
lr = 0.001
num_epochs = 40
mile_stones = [15, 30]
device = "cuda:0"

tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
text_model = AutoModel.from_pretrained('ckiplab/bert-base-chinese-ws').to(device)
#text_model.load_state_dict(torch.load('./models/text_regress/text_model_eph10.pth'))

#for param in text_model.parameters():
#    param.requires_grad = False

import torch.nn.functional as F

class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(feature_size, 512)
        #self.bn1 = torch.nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        #self.bn2 = torch.nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = self.bn1(x)
        x = F.relu(self.fc2(x))
        #x = self.bn2(x)
        x = self.fc3(x)
        return x

views = []
titles = []
ratio = 0.7
for file_name in os.listdir(data_root):
    temp_views = []
    temp_titles = []
    temp_views2 = []
    data_path = os.path.join(data_root,file_name)
    with open(data_path, 'r', encoding='utf-8') as json_file:
        json_list = list(json_file)
   
    for json_str in json_list:
        result = json.loads(json_str)
        #if result['image']!=None:
        temp_views.append(float(result['views']))
        temp_titles.append(result['title'])
    threshold = sorted(temp_views)[int(len(temp_views)*ratio)]
    
    for idx, view in enumerate(temp_views):
        if view < threshold:
            temp_views2.append(view)
            titles.append(temp_titles[idx])
    min_v = min(temp_views2)
    max_v = max(temp_views2)

    for view in temp_views2:
        views.append((view-min_v)/(max_v-min_v))

class TextDataset(Dataset,):
    def __init__(self, titles, views):
        self.titles = titles
        self.views = torch.tensor(views)
        print(len(self.titles), len(self.views))
    def __len__(self):
        return len(self.views)
    def __getitem__(self, idx):
        encoded_input = tokenizer(self.titles[idx], max_length = 512, padding = 'max_length',return_tensors='pt')
        return encoded_input, self.views[idx]

text_dataset = TextDataset(titles,views)
generator=torch.Generator().manual_seed(1)
train_text_dataset, valid_text_dataset = torch.utils.data.random_split(
        text_dataset, [0.8, 0.2], generator=generator)

dataloader = {}
dataloader['Train'] = DataLoader(train_text_dataset, batch_size=batch_size, shuffle=True)
dataloader['Valid'] = DataLoader(valid_text_dataset, batch_size=batch_size, shuffle=False)

regress_model = RegressionModel().to(device)
#regress_model.load_state_dict(torch.load('./models/text_regress/regress_model_eph10.pth'))
optimizer = torch.optim.SGD(regress_model.parameters(),lr=lr, momentum=0.9,weight_decay=5e-4)
criterian = torch.nn.MSELoss()

current_time = get_current_time()

with open('./logs/'+current_time+'.txt', 'w') as log_file:
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        logs = ''
        if epoch in mile_stones:
            optimizer.param_groups[0]['lr'] *= 0.1

        for phase in ['Train', 'Valid']:
            running_loss = 0.0
            n_data = 0
            start = time.time()
            for i, (encoded_inputs, views) in enumerate(dataloader[phase]):
                pre = time.time()
                input_ids = torch.squeeze(encoded_inputs['input_ids']).to(device)
                attention_mask = torch.squeeze(encoded_inputs['attention_mask']).to(device)
                views = views.to(device)
                n_data+=len(views)
                if phase == 'Train':
                    optimizer.zero_grad()
                    text_embeddings = text_model(input_ids, attention_mask)['pooler_output'].to(device)
                    outputs = torch.squeeze(regress_model(text_embeddings))
                    loss = torch.sqrt(criterian(outputs, views))
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        text_embeddings = text_model(input_ids, attention_mask)['pooler_output'].to(device)
                        outputs = torch.squeeze(regress_model(text_embeddings))
                        loss = torch.sqrt(criterian(outputs, views))

                running_loss += loss.item() * text_embeddings.size(0)
                diff = time.time()-pre
                lasting = time.time() - start
                total_time = diff*(len(dataloader[phase]))-lasting

                log = phase+f': {i+1}/{len(dataloader[phase])}, RMSE = {running_loss/n_data:.4f}    [{int(lasting/60):02d}:{int(lasting%60):02d}<{int(total_time/60):02d}:{int(total_time%60):02d}]'
                print(log, end = '     \r')
            logs += log+'\n'
            print('')
        log_file.write(f'Epoch {epoch+1}: \n' + logs[:-1] + '\n')
        if epoch%5==0:
            torch.save(regress_model.state_dict(), f'./models/text_regress/regress_model_eph{epoch}.pth')
            torch.save(text_model.state_dict(), f'./models/text_regress/text_model_eph{epoch}.pth')