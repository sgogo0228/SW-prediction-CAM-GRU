import numpy as np
import torch.nn as nn
import scipy.signal as sig

# 將raw ultrasound data轉為IQ data
def IQ_demod(input, fc, fs, dim=0):
    sig_len, time_step = input.shape
    t = np.arange(0, sig_len/fs, 1/fs)
    t = np.repeat(t[:, None], time_step, axis=1)
    I = 2*input*np.cos(2*np.pi*fc*t)
    Q = 2*input*np.sin(2*np.pi*fc*t)
    b, a = sig.butter(5, fc*2/fs)
    I = sig.filtfilt(b, a, I, axis=dim)
    Q = sig.filtfilt(b, a, Q, axis=dim)
    return I, Q

# 初始化模型權重
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, mean=0, std=0.2)
        m.bias.data.fill_(0.1)
