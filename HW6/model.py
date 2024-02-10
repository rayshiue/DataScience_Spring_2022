import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.L1 = torch.nn.Linear(out_channels + 10, 1024)
        self.L2 = torch.nn.Linear(1024, 1)

    def forward(self, x, edge_index):
        x = torch.concatenate((x, F.relu(self.conv2(F.relu(self.conv1(x, edge_index)), edge_index))), dim=1)
        x = F.relu(self.L1(x))
        x = self.L2(x)
        return torch.nn.Sigmoid()(x)