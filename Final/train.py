import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import json
from utils import get_current_time
import os
import torchvision
from PIL import Image
import requests

from transformers import (
  BertTokenizerFast,
  AutoModel,
)

model_root = './models/'
logs_root = './logs/'
data_root = './data/' #自己建這個資料夾，把jsonl資料都丟進去

if not os.path.isdir(model_root):
    os.makedirs(model_root)
if not os.path.isdir(logs_root):
    os.makedirs(logs_root)

feature_size = 768
batch_size = 32
lr = 0.001
num_epochs = 100
mile_stones = [50, 75]
device = "cuda:0"

tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
text_model = AutoModel.from_pretrained('ckiplab/bert-base-chinese-ws').to(device)

img_model = torchvision.models.resnet18(weights='DEFAULT', )
img_model.fc = nn.Sequential(*list(img_model.fc.children())[:-3])
#for param in img_model.parameters():
#    param.requires_grad = False
img_model = img_model.to(device)

for param in text_model.parameters():
    param.requires_grad = False

import torch.nn.functional as F

class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(feature_size + 512, 512)
        #self.bn1 = torch.nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        #self.bn2 = torch.nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = self.bn1(x)
        x = F.relu(self.fc2(x))
        #x = self.bn2(x)
        x = self.fc3(x)
        return x

class ImageDataset(Dataset):
    def __init__(self, image_urls):
        self.image_urls = image_urls
    def __len__(self):
        return len(self.image_urls)
    def __getitem__(self, idx):
        img = Image.open(requests.get(self.image_urls[idx], stream=True).raw)
        ratio = 224 / min(img.size)
        transform = transforms.Compose([
            transforms.Resize(size=(int(img.size[0]*ratio),int(img.size[1]*ratio))),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        return transform(img)

titles = []
targets = []
images = []
nodes = ['8 days', '2 weeks', '3 weeks', '4 weeks', '1 month']
for file_name in os.listdir(data_root):
    data_path = os.path.join(data_root,file_name)
    with open(data_path, 'r', encoding='utf-8') as json_file:
        json_list = list(json_file)
    f = True
    i = 0
    views = []
    temp_titles = []
    temp_images = []
    for json_str in json_list:
        node = nodes[i]
        result = json.loads(json_str)
        if result['image']!=None:
            if result['upload_time'] == node and f:
                i+=1
                if node == '1 month':
                    break
                for title in temp_titles:
                    titles.append(title)
                midv = sorted(views)[int(len(views)/2)]
                for view in views:
                    if view>midv:
                        targets.append(1)
                    else:
                        targets.append(0)

                for image in  temp_images:
                    images.append(image)

                f = False
                views = []
                temp_titles = []
                temp_images = []
            if result['upload_time'] != node:
                f = True
            views.append(float(result['views']))
            temp_images.append(result['image'])
            temp_titles.append(result['title'])
print(len(images))
class TextDataset(Dataset,):
    def __init__(self, titles, targets, image_urls):
        self.titles = titles
        self.targets = torch.tensor(targets)
        self.image_urls = image_urls

        print(len(titles), len(targets))
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        img = Image.open(requests.get(self.image_urls[idx], stream=True).raw)
        ratio = 224 / min(img.size)
        trans = transforms.Compose([
            transforms.Resize(size=(int(img.size[0]*ratio),int(img.size[1]*ratio))),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        encoded_input = tokenizer(self.titles[idx], max_length = 512, padding = 'max_length',return_tensors='pt')
        return encoded_input, trans(img), self.targets[idx]

text_dataset = TextDataset(titles, targets, images)
generator=torch.Generator().manual_seed(1)
train_text_dataset, valid_text_dataset = torch.utils.data.random_split(
        text_dataset, [0.8, 0.2], generator=generator)

dataloader = {}
dataloader['Train'] = DataLoader(train_text_dataset, batch_size=batch_size, shuffle=True)
dataloader['Valid'] = DataLoader(valid_text_dataset, batch_size=batch_size, shuffle=False)

regress_model = RegressionModel().to(device)
optimizer = torch.optim.SGD(regress_model.parameters(),lr=lr, momentum=0.9,weight_decay=5e-4)
#criterian = torch.nn.MSELoss()
criterian = torch.nn.CrossEntropyLoss()

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
            running_acc = 0.0
            n_data = 0
            #p = tqdm(total=len(dataloader[phase]), position=0, leave=True)
            for i, (encoded_inputs, imgs, targets) in enumerate(dataloader[phase]):
                #p.update(1)
                input_ids = torch.squeeze(encoded_inputs['input_ids']).to(device)
                attention_mask = torch.squeeze(encoded_inputs['attention_mask']).to(device)
                imgs = imgs.to(device)
                targets = targets.type(torch.LongTensor).to(device)
                n_data+=len(targets)
                if phase == 'Train':
                    optimizer.zero_grad()
                    text_embeddings = text_model(input_ids, attention_mask)['pooler_output'].to(device)
                    img_embeddings = img_model(imgs)
                    regress_input = torch.cat((img_embeddings, text_embeddings), axis = 1)
                    outputs = regress_model(regress_input)
                    preds = torch.argmax(outputs, axis=1)
                    loss = criterian(outputs, targets)

                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        text_embeddings = text_model(input_ids, attention_mask)['pooler_output'].to(device)
                        img_embeddings = img_model(imgs)
                        regress_input = torch.cat((img_embeddings, text_embeddings), axis = 1)
                        outputs = regress_model(regress_input)
                        preds = torch.argmax(outputs, axis=1)

                        loss = criterian(outputs, targets)
                running_acc += sum(targets==preds)
                running_loss += loss.item() * text_embeddings.size(0)
                log = phase+f': {i+1}/{len(dataloader[phase])}, Loss = {running_loss/n_data:.4f}, Acc = {running_acc/n_data:.4f}'
                print(log, end = '\r')
            #p.close()
            logs += log+'\n'
            print('')
        log_file.write(f'Epoch {epoch+1}: \n' + logs[:-1] + '\n')
        if epoch%5==0:
            torch.save(regress_model.state_dict(), f'./models/text_and_image/regress_model_eph{epoch}.pth')
            torch.save(text_model.state_dict(), f'./models/text_and_image/text_model_eph{epoch}.pth')