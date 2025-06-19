import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, mask_channels, dropout_p = 0.0, threshold = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        self.mask_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, mask_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(mask_channels, 1),
        )

        self.dropout_p = dropout_p
        self.threshold = threshold
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        for m in self.mask_head:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout_p, training=self.training)
        z = self.conv2(h, edge_index)

        logits = F.log_softmax(z, dim=1)
        prob = self.mask_head(h).squeeze(-1)
        prob = torch.sigmoid(prob)

        mask = (prob > self.threshold).float()
        mask = prob + (mask - prob.detach())
        return logits, prob, mask

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x