import torch
import torch.nn as nn

# Code location is: root/models/common.py -> 280 lines
class LFSa(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, cin, dimension=1, lk=8):
        super().__init__()
        self.act = nn.Softmax(dim=-1)
        self.d = dimension
        # head_dim = cin // self.head
        self.c = nn.Conv2d(in_channels=cin, out_channels=cin, kernel_size=1, stride=1, bias=True)
        # self.to_qkv = nn.Conv2d(in_channels=cin, out_channels=cin * 3, kernel_size=1, stride=1, bias=True)
        self.lc = nn.Conv2d(in_channels=cin, out_channels=cin, kernel_size=lk, stride=1, padding=lk//2, groups=cin, bias=True)
        self.sc = nn.Conv2d(in_channels=cin, out_channels=cin, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        
        x = torch.cat(x, self.d)
        q, k, v = self.c(x), self.c(x), self.c(x)
        q_r, k_r, v_r = q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3)
        _, _, h, w = x.shape

        x_row = self.act(torch.matmul(q, k_r) * w ** -0.5)
        x_col = self.act(torch.matmul(q_r, k) * h ** -0.5)
        x_row = self.lc(self.sc(torch.matmul(x_row, v)))
        x_col = self.lc(self.sc(torch.matmul(x_col, v_r)))
        
        return x + x_row + x_col.transpose(2,3)