import torch.nn as nn
import torch

# 最終用於評估效能的loss funtion之一，另一個是MSE loss
class xcorr_loss(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim

    # 若output或target全部為0，則會輸出nan，設條件排除，特別是CNN會發生這種情況。由於我的訊號都是-n到+n，
    # 有時候CNN會predict全部都是0造成分母計算出問題，之前解法是對label data做shift
    def forward(self, output, target):
        mean_o = output.mean(axis=self.dim)
        mean_t = target.mean(axis=self.dim)
        mean_ot = torch.mul(output, target).mean(axis=self.dim)
        mean_square_o = output.pow(2).mean(axis=self.dim)
        mean_square_t = target.pow(2).mean(axis=self.dim)
        den = torch.mul(torch.sqrt(mean_square_o - mean_o.pow(2)+1e-4), torch.sqrt(mean_square_t - mean_t.pow(2)+1e-4))
        xcorr = 1-torch.div(mean_ot-torch.mul(mean_o, mean_t), den)
        
        return xcorr.mean()

# 為增加模型訓練的效能(如correlation 0.9和0.95在數字上可能差距不大，但仍有改善的必要)
class log_xcorr_loss(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim

    # 若output或target全部為0，則會輸出nan，設條件排除，特別是CNN會發生這種情況。由於我的訊號都是-n到+n，
    # 有時候CNN會predict全部都是0造成分母計算出問題，之前解法是對label data做shift
    def forward(self, output, target):
        mean_o = output.mean(axis=self.dim)
        mean_t = target.mean(axis=self.dim)
        mean_ot = torch.mul(output, target).mean(axis=self.dim)
        mean_square_o = output.pow(2).mean(axis=self.dim)
        mean_square_t = target.pow(2).mean(axis=self.dim)
        den = torch.mul(torch.sqrt(mean_square_o - mean_o.pow(2)+1e-4), torch.sqrt(mean_square_t - mean_t.pow(2)+1e-4))
        xcorr = 1-torch.div(mean_ot-torch.mul(mean_o, mean_t), den)
        
        return torch.log(xcorr.mean()+1e-6)

# 同為改善訓練效能
class gradient_loss(nn.Module):
    def __init__(self, dim=0):
        super(gradient_loss, self).__init__()
        self.dim = dim

    def forward(self, output, target):
        d_o = torch.gradient(output, dim=self.dim)[0]
        d_t = torch.gradient(target, dim=self.dim)[0]
        loss_g = ((d_o - d_t)**2).mean()
        return loss_g*100

# 同為改善訓練效能
class nrmse_loss(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim
        
    def forward(self, output, target):
        signal_wise_mse = torch.sqrt(torch.mean((output - target)**2, dim=self.dim))
        target_amp = torch.sqrt(torch.mean(target**2, dim=self.dim))
        rrmse = torch.mean(torch.div(signal_wise_mse, target_amp))
        return rrmse