import torch
from torch import nn
import torch.nn.functional as F


class SinkhornNet(nn.Module):
    def __init__(self, N, n_iters, tau):
        super(SinkhornNet, self).__init__()
        self.n_iters = n_iters
        self.tau = tau

        self.W1_txt = nn.Linear(300, 128)
        self.W1_vis = nn.Linear(2048, 512)
        self.W2_vis = nn.Linear(512, 128)
        self.W_fc_pos = nn.Linear(260, 256)
        self.W_fc = nn.Linear(256, N)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.W1_txt.weight)
        nn.init.constant_(self.W1_txt.bias, 0)
        nn.init.xavier_normal_(self.W1_vis.weight)
        nn.init.constant_(self.W1_vis.bias, 0)
        nn.init.xavier_normal_(self.W2_vis.weight)
        nn.init.constant_(self.W2_vis.bias, 0)
        nn.init.xavier_normal_(self.W_fc_pos.weight)
        nn.init.constant_(self.W_fc_pos.bias, 0)
        nn.init.xavier_normal_(self.W_fc.weight)
        nn.init.constant_(self.W_fc.bias, 0)

    def sinkhorn(self, x):
        x = torch.exp(x / self.tau)

        for _ in range(self.n_iters):
           x = x / (10e-8 + torch.sum(x, -2, keepdim=True))
           x = x / (10e-8 + torch.sum(x, -1, keepdim=True))

        return x

    def forward(self, seq):
        x_txt = seq[:, :, :300]
        x_vis = seq[:, :, 300:2348]
        x_pos = seq[:, :, 2348:]
        x_txt = F.relu(self.W1_txt(x_txt))
        x_vis = F.relu(self.W1_vis(x_vis))
        x_vis = F.relu(self.W2_vis(x_vis))
        x = torch.cat((x_txt, x_vis, x_pos), dim=-1)
        x = F.relu(self.W_fc_pos(x))

        x = torch.tanh(self.W_fc(x))

        return self.sinkhorn(x)
