import torch
import torch.nn.functional as F
import numpy as np
from model import GCN

model = GCN(10, 128, 128).to('cuda')

train_mask = np.load('./dataset/train_mask.npy')
train_graph = torch.load('./dataset/train_sub-graph_tensor.pt')

x = train_graph['feature'].float().to('cuda')
edge_idx = train_graph['edge_index'].to('cuda')
y = train_graph['label'].float().to('cuda')
print(sum(y))
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=5e-4)
criterion = torch.nn.BCELoss()

num_epochs = 10000
model.train()
for i in range(num_epochs):
    pred = torch.squeeze(model(x, edge_idx)[train_mask])
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if i % 100 == 0:
        print(f'Epoch {i}: Loss = {float(loss)}, Sum = {float(sum(pred))}')
    if i%3000==0 and i!=0:
        optimizer.param_groups[-1]['lr']*=0.1

torch.save(model.state_dict(), './best.pth')
    