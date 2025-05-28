import torch.nn as nn
import torch
from torch.autograd import Variable

# get position embedding (PE) table based on sin&cos function
# reference: 'Attention is all you need'
def get_sin_cos_PE(emb_len=1000, emb_dim=20):
    pos_emb = torch.zeros(emb_len, emb_dim, dtype=torch.float)
    vel = torch.arange(emb_dim)*0.5
    t_interval = torch.tensor(1/2/1e9) * (1000 / emb_len)
    t = torch.cat((torch.arange(emb_len / 2 - 1, -1, -1), torch.arange(emb_len / 2)))*t_interval
    freq = 50
    wave_len = 1/freq
    torch_pi = torch.acos(torch.zeros(1)).item()
    for i in range(emb_dim):
        pos_emb[:, i] = torch.cos(vel[i]*t/wave_len*2*torch_pi)
    
    return pos_emb

class Attention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super().__init__()
        
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        
        # self.linear_in = nn.Linear(enc_hidden_size, dec_hidden_size)     #主要應該還是為了匹配encoder&decoder兩邊hidden output的長度
        self.linear_out = nn.Linear(enc_hidden_size+dec_hidden_size, dec_hidden_size)
        self.linear_query = nn.Linear(enc_hidden_size, dec_hidden_size)
    
    def forward(self, output, encoder_output, time_lag):
        # output: [batch_size, seq_len_y-1, dec_hidden_size]  这个output 是decoder的每个时间步输出的隐藏状态
        # encoder_output: [batch_size, seq_len_x, 2*enc_hidden_size]
        
        output1 = self.linear_query(output)
        context_in = encoder_output.transpose(1, 2)   # [batch_size, dec_hidden_size, seq_len_x
        
        attn = torch.bmm(output1, context_in)  # [batch_size, seq_len_y-1, seq_len_x]
        
        # 这个东西就是求得当前时间步的输出output和所有输入相似性关系的一个得分score , 下面就是通过softmax把这个得分转成权重
        attn = nn.functional.softmax(attn, dim=2)    # 此时第二维度的数字全都变成了0-1之间的数， 越大表示当前的输出output与哪个相关程度越大
        context = torch.bmm(attn, encoder_output)   # [batch_size, seq_len_y-1, enc_hidden_size]

        output = torch.cat((context, output), dim=2)  # [batch_size, seq_len_y-1, 2*enc_hidden_size+dec_hidden_size]
        output = torch.tanh(self.linear_out(output))     # [batch_size*seq_len_y-1, dec_hidden_size]
        
        return output, attn
    
class CAM_GRU(nn.Module):
#建立GRU class
    def __init__(self,encode_input_channel, encode_output_channel, encode_layer_num, decode_input_channel, decode_output_channel, decode_layer_num):
        super().__init__()
        self.encode_input_channel=encode_input_channel
        self.encode_output_channel=encode_output_channel
        self.encode_layer_num=encode_layer_num
        self.decode_input_channel=decode_input_channel
        self.decode_output_channel=decode_output_channel
        self.decode_layer_num=decode_layer_num
        self.attn = Attention(encode_output_channel, decode_output_channel)
        #初始化encoder
        self.gru1 = nn.GRU(self.encode_input_channel, 500, self.encode_layer_num, batch_first=True, dropout=0.2, bidirectional=True).cuda()
        
        #初始化decoder
        self.gru2=nn.GRU(self.decode_input_channel, self.decode_output_channel, self.decode_layer_num, batch_first=True, dropout=0.2).cuda()
        self.key = nn.Linear(encode_output_channel, decode_output_channel)
        self.position_emb = nn.Embedding(1000, 10)
        self.linear_emb = nn.Linear(20, 184)
        self.linear_out = nn.Linear(184*3, 184*2)
        self.linear_out2 = nn.Linear(184*2, 184)
        self.linear_out3 = nn.Linear(184, 184)
        

    def forward(self,input, encoder_input, xcorr_max_idx, output_time_steps):
        output = torch.zeros((input.size(0), output_time_steps, self.decode_output_channel)).cuda()
        h0 = torch.zeros((2*self.encode_layer_num, input.size(0), self.encode_output_channel)).cuda()
        # encoder_output: (bs, seq, bi*encode_hid_size)
        #  hn: (bi*encode_layer, bs, encode_hid_size)
        encoder_output, hn = self.gru1(input, h0)
        
        #雙向結構會讓output有兩組，之前都是用這段將兩組加起來
        latency_vector=encoder_output[:, -1:, 0:self.encode_output_channel]+encoder_output[:, -1:, self.encode_output_channel:]
        hn = hn[self.encode_layer_num:, :, :]
        encoder_output = encoder_output[:, :, 0:self.encode_output_channel]+encoder_output[:, :, self.encode_output_channel:]
        encoder_output = self.key(encoder_output)
        
        # #雙向結構會讓output有兩組，嘗試過將gru encoder output設為一半長度並串接，但gru_2000_2layer_bs10_epoch200_groupwidth1000_2說明並沒有比加起來好
        # latency_vector=encoder_output[:, -1:, :]    #將encoder hidden output設為一半，這樣雙向的rnn output就會剛好符合endocder input的需求，不需要把他們加起來
        # hn = torch.cat((hn[[0, 2], :, :], hn[[1, 3], :, :]), dim=2)
        # hn.permute((1, 0, 2)).reshape((-1, 2, 1000)).permute((1, 0, 2))
        
        output[:, 0:1, :], hn = self.gru2(torch.cat((latency_vector, encoder_input[:, 0:1, :]), dim=2), hn)
        for t in range(1, output_time_steps):
            temp, hn = self.gru2(torch.cat((output[:, t-1:t, :], encoder_input[:, t:t+1, :]), dim=2), hn)
            output[:, t:t+1, :], _ = self.attn(temp, encoder_output.contiguous(), t)
        
        output = output.permute((0, 2, 1))
        pos_info = get_sin_cos_PE().cuda()
        pos_info = self.linear_emb(pos_info)[None, :, :].repeat((output.size(0), 1, 1))
        
        xcorr_max_idx = xcorr_max_idx[:, None, :].repeat((1, self.decode_output_channel, 1))
        output = torch.cat((output, xcorr_max_idx, pos_info), dim=2)
        output = output.reshape((-1, output_time_steps*3))
        output = self.linear_out3(self.linear_out2(self.linear_out(output)))
        output = output.reshape((-1, self.decode_output_channel, output_time_steps)).permute((0, 2, 1))
        
        return output,hn