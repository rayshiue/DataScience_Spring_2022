import os
from vgg import vgg
import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.utils.model_zoo as model_zoo
import torchvision.transforms.functional as F
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from PIL import Image
import random

origin_data_root = './data/train'
data_root = './data/processed'
crop_size = 512
sigma = 8.0
background_ratio = 1.0
downsample_ratio = 8
lr = 1e-5
weight_decay = 1e-4
total_epoch = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def cal_innner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right-inner_left, 0.0) * np.maximum(inner_down-inner_up, 0.0)
    return inner_area

class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.paths = []
        for name in os.listdir(data_root):
            if name[-4:] == '.jpg':
                self.paths.append(os.path.join(data_root, name))
        self.trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, item):
        img_path = self.paths[item]
        gnd_path = img_path.replace('jpg', 'npy')
        img = Image.open(img_path).convert('RGB')
        keypoints = np.load(gnd_path)
        return self.train_transform(img, keypoints)
    
    def train_transform(self, img, keypoints):
        w, h = img.size
        st_size = min(w, h)
        i = random.randint(0, h- crop_size)
        j = random.randint(0, w - crop_size)
        img = F.crop(img, i, j, crop_size, crop_size)

        nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0)

        points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
        points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
        bbox = np.concatenate((points_left_up, points_right_down), axis=1)
        inner_area = cal_innner_area(j, i, j+crop_size, i+crop_size, bbox)
        origin_area = nearest_dis * nearest_dis
        ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
        mask = (ratio >= 0.3)

        target = ratio[mask]
        keypoints = keypoints[mask]
        keypoints = keypoints[:, :2] - [j, i]
    
        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                keypoints[:, 0] = crop_size - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(target.copy()).float(), st_size

def bay_loss(prob, target_list, pre_density):
    target = torch.zeros((1,), dtype=torch.float32, device=device)
    if prob is None:
        pre_count = torch.sum(pre_density[0])
    else:
        target = target_list[0]
        pre_count = torch.sum(pre_density[0].view((1, -1)) * prob, dim=1)
    loss = torch.sum(torch.abs(target - pre_count))
    return loss

dmap_coor = (torch.arange(0, crop_size, step = downsample_ratio, dtype=torch.float32, device=device) + downsample_ratio / 2).unsqueeze_(0)
softmax = torch.nn.Softmax(dim=0)
def post_prob(points, st_size):
    prob = None
    if len(points) > 0:
        x = points[:, 0].unsqueeze_(1)
        y = points[:, 1].unsqueeze_(1)
        x_dis = (-2 * torch.matmul(x, dmap_coor) + x * x + dmap_coor * dmap_coor).unsqueeze_(1)
        y_dis = (-2 * torch.matmul(y, dmap_coor) + y * y + dmap_coor * dmap_coor).unsqueeze_(2)
        dis = y_dis + x_dis
        dis = dis.view((dis.size(0), -1))
        if len(dis) > 0:
            min_dis = torch.min(dis, dim=0, keepdim=True)[0]
            bg_dis = (st_size * background_ratio) ** 2 / (min_dis + 1e-5)
            dis = -torch.cat([dis, bg_dis], 0) / (2.0 * sigma ** 2)
            prob = softmax(dis)
    return prob
    
def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, targets, st_sizes

def preprocess():
    try:
        os.makedirs(data_root)
    except Exception as e:
        print(e)
    print('Preprocessing the data ...')
    data_name_list = os.listdir(origin_data_root)
    progress = tqdm(total = int(len(data_name_list)/2))
    for name in data_name_list:
        if name[-4:] != '.jpg':
            continue
        else:
            img_path = os.path.join(origin_data_root,name)
        progress.update(1)
        gnd_path = img_path.replace('.jpg','.txt')
        points = [] 
        with open (gnd_path, 'r') as f:
            while True:
                point = f.readline()
                if not point:
                    break
                point = point.split(' ')
                points.append([float(point[0]), float(point[1])])
        points = np.array(points)
        if len(points)==0:
            points = np.zeros((1,2))

        img = Image.open(img_path)
        w, h = img.size
        idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= w) * (points[:, 1] >= 0) * (points[:, 1] <= h)
        points = points[idx_mask]
        
        shorter_side = min(w,h)
        if shorter_side < crop_size:
            ratio = crop_size/shorter_side
            img = img.resize((round(w*ratio),round(h*ratio)),Image.Resampling.LANCZOS)
            points *= ratio

        x, y = points[:,0], points[:,1]
        n_points = points.shape[0]
        square = x*x + y*y
        dis = np.sqrt(square[:, None] - 2*np.matmul(points, points.T) + square[None, :])
        if n_points == 0:
            points = np.zeros((1,3))
        else:
            if n_points > 3:
                closest_dis = np.sort(dis)[:,1:4]
            else:
                closest_dis = np.sort(dis)[:,1:n_points]
            if len(closest_dis)==1:
                closest_dis = [[0]]
            avg_dis = np.mean(closest_dis,axis=1)
            points = np.concatenate((points, avg_dis[:,None]), axis=1)
        save_img_path = os.path.join(data_root, name)
        img.save(save_img_path, quality=95)
        np.save(save_img_path.replace('.jpg', '.npy'), points)
    progress.close()

if __name__ == '__main__':
    #preprocess()
    torch.backends.cudnn.benchmark = True
    datasets = MyDataset()
    dataloaders = DataLoader(datasets, collate_fn=train_collate, batch_size=1, shuffle=True, pin_memory=True)
    model = vgg()
    model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'), strict=False)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(total_epoch):
        print('Epoch: {}/{}'.format(epoch,total_epoch))
        epoch_loss = 0
        epoch_mae = 0
        total_num_input = 0 

        progress = tqdm(total = len(dataloaders))
        for inputs, points, targets, st_sizes in dataloaders:
            progress.update(1)
            inputs = inputs.to(device)
            st_sizes = st_sizes.to(device)
            gnd_count = np.array([len(points[0])], dtype=np.float32)
            points = points[0].to(device)
            targets = targets[0].to(device)

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                prob = post_prob(points, st_sizes)
                loss = bay_loss(prob, targets, outputs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                N = inputs.size(0)
                pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                total_num_input += N
                epoch_loss += loss.item() * N
                epoch_mae += np.mean(abs(pre_count - gnd_count)) * N
        progress.close()
        print('Loss: {:.3f} MAE: {:.3f}'.format(epoch_loss/total_num_input, epoch_mae/total_num_input))
    torch.save(model.state_dict(), './model.pth')