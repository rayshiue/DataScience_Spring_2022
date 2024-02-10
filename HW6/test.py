import numpy as np
import torch
import csv
from model import GCN

test_mask = np.load('./dataset/test_mask.npy')
test_graph = torch.load('./dataset/test_sub-graph_tensor_noLabel.pt')

x = test_graph['feature'].float()
edge_idx = test_graph['edge_index']

hidden_channels = 128
out_channels = 128
model = GCN(10, hidden_channels, out_channels)
model.load_state_dict(torch.load('./best.pth'))
pred = model(x, edge_idx)


with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['node idx','node anomaly score'])
    for idx, p in enumerate(pred):
        if test_mask[idx]:
            writer.writerow([idx, float(p)])