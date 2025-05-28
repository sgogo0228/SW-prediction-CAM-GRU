import sys, os, time, argparse
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
from torchsummary import summary
import utils.losses as mylf
import utils.io_utils as io_utils
import model.cam_gru as custom_module
import data.dataset as my_dataset

num_samples=1
max_num_epochs=100

def load_data(data_dir="./"):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([1000, 920]),
        # transforms.Normalize(-5, 0.5),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([1000, 920]),
        # transforms.Normalize(-5, 0.5),
    ])

    train_dir = os.path.join(data_dir, 'training')
    train_datasets = my_dataset.Displacement_dataset_IQ(root=train_dir, transform=train_transform)

    val_dir = os.path.join(data_dir, 'validation')
    val_datasets = my_dataset.Displacement_dataset_IQ(root=val_dir, transform=val_transform)

    return train_datasets, val_datasets

def train_cifar(checkpoint_dir=None, data_dir=None, encode_layer_num=2, decode_layer_num=2, learning_rate=1e-4, batch_size=30, time_steps=920, output_time_steps=184, val_epoch_interval=5, storage_epoch=25, max_epoch=200):
    data_depth = 1000
    data_group_width = 1000
    groups = np.int32(data_depth/data_group_width)
    is_corr_fun_calc = True
    
    # 根據是否有corr input, IQ input調整倍率
    encode_input_channel = data_group_width*2
    encode_output_channel = data_group_width
    
    # 根據是否有corr input調整倍率
    decode_input_channel = data_group_width*2
    decode_output_channel = data_group_width

    net_name = 'CAM_GRU'
    params = {'encode_input_channel':encode_input_channel,
              'encode_output_channel':encode_output_channel,
              'encode_layer_num':encode_layer_num,
              'decode_input_channel':decode_input_channel,
              'decode_output_channel':decode_output_channel,
              'decode_layer_num':decode_layer_num,
              'max_epoch':max_epoch,
              'learning_rate':learning_rate,
              'batch_size':batch_size,
              'time_steps':time_steps,
              'output_time_steps':output_time_steps,
              'val_epoch_interval':val_epoch_interval,
              'data_depth':data_depth,
              'data_group_width':data_group_width,
              'net_name':net_name}

    file_copy = 1

    output_dir = os.path.join("outputs", "models")
    os.makedirs(output_dir, exist_ok=True)
    file_name_prefix = os.path.join(output_dir, f"{net_name}_{encode_input_channel}_{encode_layer_num}layer_bs{batch_size}_epoch{max_epoch}_groupwidth{data_group_width}")
    
    while os.path.exists(rf'{file_name_prefix}_{file_copy}.pt')==True :
        file_copy += 1
    file_name = rf'{file_name_prefix}_{file_copy}'
    

    trainset, valset = load_data(data_dir=data_dir)
    train_dataloader = torch.utils.data.DataLoader(trainset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0)
    
    val_dataloader = torch.utils.data.DataLoader(valset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0)
    
    model_class = getattr(custom_module, net_name)
    model = model_class(
        encode_input_channel, encode_output_channel, encode_layer_num,
        decode_input_channel, decode_output_channel, decode_layer_num
    )

    if torch.cuda.is_available():
        model.cuda()

    #建構LSTM物件
    optimizer=optim.Adam(model.parameters(),lr=learning_rate)
    #宣告反向傳播機制
    loss_g = mylf.gradient_loss(dim=1)
    loss_nrmse = mylf.nrmse_loss(dim=1)
    loss_xcorr = mylf.xcorr_loss(dim=1)
    loss_log_xcorr = mylf.log_xcorr_loss(dim=1)
    loss_val = nn.MSELoss()
    train_loss = np.zeros((max_epoch))
    train_mse = np.zeros((max_epoch))
    train_xcorr = np.zeros((max_epoch))
    train_log_xcorr = np.zeros((max_epoch))
    train_nrmse = np.zeros((max_epoch))
    val_loss = np.zeros((max_epoch))
    val_mse = np.zeros((max_epoch))
    val_xcorr = np.zeros((max_epoch))
    val_log_xcorr = np.zeros((max_epoch))
    val_nrmse = np.zeros((max_epoch))
    training_time = np.zeros((max_epoch))
    val_time = np.zeros((max_epoch))
    best_loss = 50000
    best_net = model
    best_model_epoch = 0
    
    #宣告損失函數
    for epoch in tqdm(range(max_epoch)):#每一輪
        t = time.time()
        for inputs, labels, I_inputs, Q_inputs in train_dataloader: #數據集內的每個數據和標籤
            inputs = inputs.reshape(-1, data_group_width, time_steps).permute((0, 2, 1)).cuda()
            I_inputs = I_inputs.reshape(-1, data_group_width, time_steps).permute((0, 2, 1)).cuda()
            Q_inputs = Q_inputs.reshape(-1, data_group_width, time_steps).permute((0, 2, 1)).cuda()
            labels = labels.reshape(-1, data_depth, 5, output_time_steps).mean(dim=2).permute((0, 2, 1)).cuda()
            
            IQ = torch.sqrt(I_inputs**2+Q_inputs**2)
            local_bz = inputs.size()[0]
            
            if is_corr_fun_calc:
                xcorr_func = torch.zeros(local_bz, time_steps, data_group_width)
                # 需更改model的in_feature_dim
                # 計算cross function作為輸入以增強效能
                for bz in range(0, local_bz):
                    temp1 = inputs[bz:bz+1, :, :]
                    temp2 = inputs[bz:bz+1, 0:1, :].repeat((1, time_steps, 1)).permute((1, 0, 2))
                    xcorr_func[bz, :, :] = nn.functional.conv1d(temp1, temp2, groups=time_steps, padding='same')[0, :, :]
                    xcorr_func[bz, :, :] = (xcorr_func[bz, :, :]-xcorr_func[bz, :, :].min())/(xcorr_func[bz, :, :].max()-xcorr_func[bz, :, :].min())
            else:
                xcorr_func = torch.zeros(local_bz, time_steps, 0)
                
            inputs = torch.cat((inputs, IQ), 2)
            encoder_info = xcorr_func.cuda()
            
            xcorr_max_idx = torch.max(xcorr_func, dim=2).indices.float()
            xcorr_max_idx = xcorr_max_idx.reshape(local_bz, -1, 184).mean(dim=1)-499
            xcorr_max_idx = xcorr_max_idx.cuda()
            
            temp,hn=model(inputs, encoder_info, xcorr_max_idx, output_time_steps) #正向傳播
            output = temp
            model.zero_grad() #清除lstm的上個數據的偏微分暫存值，否則會一直累加
            
            xcorr_loss = loss_xcorr(output,labels) #計算loss
            log_xcorr_loss = loss_log_xcorr(output, labels)
            mse_loss = loss_val(output,labels)
            nrmse_loss = loss_nrmse(output,labels)

            loss = 0.4*mse_loss + 0.6*xcorr_loss
            
            loss.backward() #從loss計算反向傳播
            optimizer.step() #更新所有權種和偏差
            loss = 0.4*mse_loss + 0.6*xcorr_loss
            
            train_loss[epoch] += loss.item() / len(train_dataloader)
            train_mse[epoch] += mse_loss.item() / len(train_dataloader)
            train_xcorr[epoch] += xcorr_loss.item() / len(train_dataloader)
            train_log_xcorr[epoch] += log_xcorr_loss.item() / len(train_dataloader)
            train_nrmse[epoch] += nrmse_loss.item() / len(train_dataloader)
        
        print(f'epoch: {epoch+1}, Train loss: {train_loss[epoch]}, train mse: {train_mse[epoch]}',
              f'train xcorr: {train_xcorr[epoch]}, train log xcorr: {train_log_xcorr[epoch]}, train nrmse: {train_nrmse[epoch]}')
        training_time[epoch] = time.time()-t

        # validation
        if ((epoch == 0) or (epoch % val_epoch_interval == val_epoch_interval-1)):
            t = time.time()
            model.eval()
            with torch.no_grad():
                
                for inputs, labels, I_inputs, Q_inputs in val_dataloader:
                    inputs = inputs.reshape(-1, data_group_width, time_steps).permute((0, 2, 1)).cuda()
                    I_inputs = I_inputs.reshape(-1, data_group_width, time_steps).permute((0, 2, 1)).cuda()
                    Q_inputs = Q_inputs.reshape(-1, data_group_width, time_steps).permute((0, 2, 1)).cuda()
                    labels = labels.reshape(-1, data_depth, 5, output_time_steps).mean(dim=2).permute((0, 2, 1)).cuda()
                    
                    IQ = torch.sqrt(I_inputs**2+Q_inputs**2)
                    local_bz = inputs.size()[0]
                    
                    if is_corr_fun_calc:
                        xcorr_func = torch.zeros(local_bz, time_steps, data_group_width)
                        # 需更改model的in_feature_dim
                        # 計算cross function作為輸入以增強效能
                        for bz in range(0, local_bz):
                            temp1 = inputs[bz:bz+1, :, :]
                            temp2 = inputs[bz:bz+1, 0:1, :].repeat((1, time_steps, 1)).permute((1, 0, 2))
                            xcorr_func[bz, :, :] = nn.functional.conv1d(temp1, temp2, groups=time_steps, padding='same')[0, :, :]
                            xcorr_func[bz, :, :] = (xcorr_func[bz, :, :]-xcorr_func[bz, :, :].min())/(xcorr_func[bz, :, :].max()-xcorr_func[bz, :, :].min())
                    else:
                        xcorr_func = torch.zeros(local_bz, time_steps, 0)
                        
                    inputs = torch.cat((inputs, IQ), 2)
                    encoder_info = xcorr_func.cuda()
                    
                    xcorr_max_idx = torch.max(xcorr_func, dim=2).indices.float()
                    xcorr_max_idx = xcorr_max_idx.reshape(local_bz, -1, 184).mean(dim=1)-499
                    xcorr_max_idx = xcorr_max_idx.cuda()
                    
                    temp,hn=model(inputs, encoder_info, xcorr_max_idx, output_time_steps) #正向傳播
                    output = temp
                    model.zero_grad() #清除lstm的上個數據的偏微分暫存值，否則會一直累加
                    
                    xcorr_loss = loss_xcorr(output,labels) #計算loss
                    log_xcorr_loss = loss_log_xcorr(output, labels)
                    mse_loss = loss_val(output,labels)
                    nrmse_loss = loss_nrmse(output,labels)
                    loss = 0.4*mse_loss + 0.6*xcorr_loss
                    val_loss[epoch] += loss.item() / len(val_dataloader)
                    val_mse[epoch] += mse_loss.item() / len(val_dataloader)
                    val_xcorr[epoch] += xcorr_loss.item() / len(val_dataloader)
                    val_log_xcorr[epoch] += log_xcorr_loss.item() / len(val_dataloader)
                    val_nrmse[epoch] += nrmse_loss.item() / len(val_dataloader)

                if val_xcorr[epoch] < best_loss:
                    best_model_epoch = epoch
                    best_loss = val_xcorr[epoch]
                    best_net = model

            print(f'Validation Loss:{val_loss[epoch]}, Validation mse:{val_mse[epoch]}', 
                  f'Validation xcorr:{val_xcorr[epoch]}, train log xcorr: {train_log_xcorr[epoch]}, validation nrmse: {val_nrmse[epoch]}')
            model.train()
            val_time[epoch] = time.time()-t
        
        if epoch%storage_epoch == storage_epoch-1:
            io_utils.output_data({'train_loss': train_loss, 'train_mse': train_mse, 'train_xcorr': train_xcorr, 'train_nrmse': train_nrmse,
            'val_loss': val_loss, 'val_mse': val_mse, 'val_xcorr': val_xcorr, 'val_nrmse': val_nrmse,
            'training_time': training_time, 'val_time': val_time, 'best_model_epoch':best_model_epoch, 'params':params}, file_name=file_name)
            torch.save(best_net.state_dict(), f'{file_name}.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./samples')
    parser.add_argument('--encode_layer_num', type=int, default=2)
    parser.add_argument('--decode_layer_num', type=int, default=2)
    parser.add_argument('--learning_rate', type=int, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--time_steps', type=int, default=920)
    parser.add_argument('--output_time_steps', type=int, default=184)
    parser.add_argument('--val_epoch_interval', type=int, default=5)
    parser.add_argument('--storage_epoch', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()

    train_cifar(data_dir=args.data_dir, encode_layer_num=args.encode_layer_num, decode_layer_num=args.decode_layer_num, 
                learning_rate=args.learning_rate, batch_size=args.batch_size, time_steps=args.time_steps, 
                output_time_steps=args.output_time_steps, val_epoch_interval=args.val_epoch_interval, 
                storage_epoch=args.storage_epoch, max_epoch=args.epochs)


    

