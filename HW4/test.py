
from PIL import Image
import os
import csv
from vgg import vgg
import torch
from torchvision import transforms
from tqdm import tqdm

test_root = './data/test'
model_path = './74_ckpt.pth'
#model_path = './best_model.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.paths = []
        for name in os.listdir(test_root):
            if name[-4:] == '.jpg':
                self.paths.append(os.path.join(test_root, name))
        self.trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, item):
        img_path = self.paths[item]
        img = Image.open(img_path).convert('RGB')
        return self.trans(img)

if __name__ == '__main__':
    model = vgg()
    device = torch.device('cuda')
    model.to(device)
    model.load_state_dict(torch.load(model_path), device)
    datasets = MyDataset()
    progress = tqdm(total=len(datasets))
    predictions = []
    for inputs in datasets:
        #progress.update(1)
        inputs = inputs[None,:].to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            print(torch.sum(outputs).item())
            predictions.append(torch.sum(outputs).item())

    progress.close()

    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'Count'])
        for i,pred in enumerate(predictions):
            writer.writerow([i+1, pred])