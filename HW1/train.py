train_path = './train_data_nocom/'
models_path = './resnet18/'
result_path = './results/'

from torch.utils.data.dataset import Dataset
import cv2
import os
from tqdm import tqdm   
import torch
import time
import matplotlib.pyplot as plt
import sys
import copy
from torchvision.transforms import transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
size = 360
num_images = 10000

def get_random_crop(image, crop_height, crop_width):

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    if max_x>0:
        x = np.random.randint(0, max_x)
    else:
        x = 0
    if max_y>0:
        y = np.random.randint(0, max_y)
    else:
        y = 0
    crop = image[y: y + crop_height, x: x + crop_width]
    return crop


def LoadImages(path):
    label_list = os.listdir(path)
    labels = []
    images = []
    n0 = 0
    n1 = 0

    for label in label_list:
        count = 0
        file_list = os.listdir(train_path+'/'+label)
        progress = tqdm(total=num_images, position=0, leave=True, file=sys.stdout)
        for f in file_list:
            if count>=num_images-1:
                progress.update(1)
                break

            img = cv2.imread(train_path+'/'+label+'/'+f)
            if img is not None:
                progress.update(1)
                count+=1
                H = img.shape[0]
                W = img.shape[1]

                if H==81 and W==161:
                  continue
                if int(label)==0:
                  n0+=1
                else:
                  n1+=1
                #print(img.shape)
                if int(H)>int(W):
                  shorter_side = int(W)
                else:
                  shorter_side = int(H)
                #if shorter_side<size:
                scale = 1.1*size/shorter_side
                width = int(W * scale)
                height = int(H * scale)

                img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
                img = get_random_crop(img, size, size)
                img = torch.FloatTensor(img).permute(2, 0, 1)

                images.append(img)
                labels.append(int(label))
                progress.refresh()
        progress.close()
    return images, labels, n0, n1


class MyDataset(Dataset):
    def __init__(self, images, labels=None,):
        self.images = images
        self.labels = labels
        #self.transform = transforms.RandomCrop((640,640))
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.images[idx], torch.tensor(int(self.labels[idx]))

images, labels, n0, n1 = LoadImages(train_path)
print(n0, n1)
dataset = MyDataset(images = images, labels = labels)


BATCH_SIZE = 5
EPOCHS = 30
#transform=transforms.RandomCrop((640,640))

data_loader={'train':0, 'valid':0}
dataset_sizes={'train':0, 'valid':0}

total_data_size = len(dataset)
train_data_size = int(total_data_size * 0.9)
valid_data_size = total_data_size - train_data_size

trainset, validset = torch.utils.data.random_split(dataset,[train_data_size,valid_data_size])

data_loader['train'] = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,shuffle=True)
dataset_sizes['train'] = len(trainset)
data_loader['valid'] = torch.utils.data.DataLoader(validset, batch_size=BATCH_SIZE,shuffle=True)
dataset_sizes['valid'] = len(validset)

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=True)
#model = torchvision.models.resnet18()
#model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='ResNet50_Weights.DEFAULT')
model = model.train()

learning_rate = 0.1

#model = torch.load('./drive/MyDrive/CloudComputing/HW1/example/models/temp_50_50_0.001/epoch9_model.pth')
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
#optimizer_ft = optimizer = torch.optim.Adam(model_ft.parameters(), lr=learning_rate, weight_decay=1e-5)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=15, gamma=0.1)


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs, result_path, model_path):
    since = time.time()
    losses = {'train':[], 'valid':[]}
    accuracy = {'train':[], 'valid':[]}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
 
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            print(phase)
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            progress = tqdm(total=len(data_loader[phase]), position=0, leave=True, file=sys.stdout)
            for inputs, labels in dataloaders[phase]:
                progress.update(1)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                progress.refresh()
            progress.close()
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            losses[phase].append(epoch_loss)
            accuracy[phase].append(float(epoch_acc))
            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model, model_path+'epoch%d_model_vcrop.pth'%epoch)
    #    print()

    #------------------------------------------ plot the results
    loss_fig, loss_ax = plt.subplots() 
    plt.xticks(range(1,num_epochs,5))
    loss_ax.plot(losses['train'], label ='Training Loss')
    loss_ax.plot(losses['valid'], label ='Validation Loss')
    loss_ax.set_xlabel('epochs')
    loss_ax.legend()
    loss_fig.suptitle("Loss", fontweight ="bold")
    loss_fig.show()
    loss_fig.savefig(result_path+'loss.png')

    acc_fig, acc_ax = plt.subplots()
    plt.xticks(range(1,num_epochs,5))
    acc_ax.plot(accuracy['train'], label ='Training Accuracy')
    acc_ax.plot(accuracy['valid'], label ='Validation Accuracy')
    acc_ax.set_xlabel('epochs')
    acc_ax.legend()
    acc_fig.suptitle("Accuracy", fontweight ="bold")
    acc_fig.show()
    acc_fig.savefig(result_path+'acc.png')

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model




try:
  model_path = models_path + "epo_{}_bs_{}_lr_{}/".format(EPOCHS,BATCH_SIZE,learning_rate)
  os.makedirs(model_path)
  print(model_path+"created!")
except Exception as e:
  print(e)
result_path = result_path + "epo_{}_bs_{}_lr_{}".format(EPOCHS,BATCH_SIZE,learning_rate)

model = train_model(model, data_loader, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=EPOCHS, result_path=result_path, model_path = model_path)