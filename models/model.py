import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import BatchNorm

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[0:1, :].repeat((self.kernel_size - 1) // 2, 1)
        end = x[-1:, :].repeat((self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=0)
        x = self.avg(x.transpose(0, 1))
        x = x.transpose(0, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class GraphSAGEResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate: float = 0.2):
        super(GraphSAGEResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = SAGEConv(in_channels, out_channels)
        self.norm1 = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(drop_rate)
        self.conv2 = SAGEConv(out_channels, out_channels)
        self.norm2 = BatchNorm(out_channels)
        self.drop2 = nn.Dropout(drop_rate)
        if in_channels != out_channels:
            self.conv3 = SAGEConv(in_channels, out_channels)
            self.norm3 = BatchNorm(out_channels)
        self.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index):
        # conv1
        h = self.conv1(x, edge_index)
        h = self.norm1(h)
        h = self.relu(h)
        h = self.drop1(h)
        # conv2
        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        h = self.drop2(h)
        # shortcut
        if self.in_channels != self.out_channels:
            x = self.conv3(x, edge_index)
            x = self.norm3(x)
        # Residual 연결
        h += x
        h = self.relu(h)
        return h

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.norm1.reset_parameters()
        self.conv2.reset_parameters()
        self.norm2.reset_parameters()
        if self.in_channels != self.out_channels:
            self.conv3.reset_parameters()
            self.norm3.reset_parameters()
  

class GraphSAGEEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, drop_rate: float = 0.2):
        super(GraphSAGEEncoder, self).__init__()
        self.layer1 = GraphSAGEResBlock(in_channels, hidden_channels, drop_rate=drop_rate)
        self.layer2 = GraphSAGEResBlock(hidden_channels, hidden_channels, drop_rate=drop_rate)
        self.drop = nn.Dropout(drop_rate)
        self.gcn_mean = SAGEConv(hidden_channels, out_channels)
        self.gcn_logstd = SAGEConv(hidden_channels, out_channels)
  
    def forward(self, x, edge_index):
        # GraphSAGE 기반 인코딩
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        x = self.drop(x)
        mean = self.gcn_mean(x, edge_index)
        log_std = self.gcn_logstd(x, edge_index)
        return mean, log_std


class GraphSAGEDecoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, drop_rate: float = 0.2):
        super(GraphSAGEDecoder, self).__init__()
        self.conv = SAGEConv(in_channels, hidden_channels)
        self.drop = nn.Dropout(drop_rate)
        self.layer1 = GraphSAGEResBlock(hidden_channels, hidden_channels, drop_rate=drop_rate)
        self.layer2 = GraphSAGEResBlock(hidden_channels, out_channels, drop_rate=drop_rate)
        self.activation = nn.Sigmoid()  # 활성화 함수 추가

    def forward(self, x, edge_index):
        # GraphSAGE 기반 인코딩
        x = self.conv(x, edge_index)
        x = self.drop(x)
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        x = self.activation(x)  # 활성화 함수 적용
        return x

class GraphSAGEVAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEVAE, self).__init__()
        self.encoder = GraphSAGEEncoder(in_channels, hidden_channels, out_channels)
        self.decoder = GraphSAGEDecoder(out_channels, hidden_channels, in_channels)

    def forward(self, x: torch.Tensor, edge_index):
        mean, log_std = self.encoder(x, edge_index)
        z = self.reparameterize(mean, log_std)
        recon_x = self.decoder(z, edge_index)
        return self.compute_loss(x, recon_x, mean, log_std) # loss, recon_loss, kl_loss
    
    def reconstruct(self, x: torch.Tensor, edge_index):
        mean, log_std = self.encoder(x, edge_index)
        z = self.reparameterize(mean, log_std)
        recon_x = self.decoder(z, edge_index)
        return recon_x  

    def reparameterize(self, mean, log_std):
        # reparameterization trick
        log_std = torch.clamp(log_std, min=-10, max=10)  # 수치적 안정성을 위해 클램핑
        eps = torch.randn_like(mean)
        return mean + eps * torch.exp(log_std)

    def compute_loss(self, x: torch.Tensor, recon_x: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor, beta: float = 0.2):
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + log_std - mean.pow(2) - log_std.exp())
        loss = recon_loss + beta * kl_loss
        return loss, recon_loss, kl_loss


class WaterGraphNet(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels:int, drop_rate: float = 0.2):
        super(WaterGraphNet, self).__init__()
        self.decompsition = series_decomp(kernel_size=25)
        self.sage1 = GraphSAGEResBlock(in_channels, out_channels, drop_rate=drop_rate)
        self.sage2 = GraphSAGEResBlock(in_channels, out_channels, drop_rate=drop_rate)
    
    def forward(self, x: torch.Tensor, edge_index):
        seasonal_init, trend_init = self.decompsition(x.transpose(0, 1))
        seasonal_output = self.sage1(seasonal_init.transpose(0, 1), edge_index)
        trend_output = self.sage2(trend_init.transpose(0, 1), edge_index)
        x = seasonal_output + trend_output
        return x.squeeze(-1)








