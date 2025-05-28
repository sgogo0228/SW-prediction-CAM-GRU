import numpy as np
import scipy.io as matlabio
import matplotlib.pyplot as plt

# 讀取raw ultrasound data
def headFile_NI(filename):
    name = filename.strip()
    with open(name + '.dat', 'rb') as f:
        rf = np.fromfile(f, dtype=np.int8)
    # data_dir = 'E:/temp'
    # mat = matlabio.loadmat(data_dir + '/30MHz_dist0to10_depth9to14k_amp600_cycle5100_1_SW0')
    # mat = mat['temp'][0][0][0]
    ScanSpeed=rf[3]*16129+rf[4]*127+rf[5]
    Aline=rf[6]*16129+rf[7]*127+rf[8]
    DataLength=rf[9]*16129+rf[10]*127+rf[11]
    SamplingRate=rf[12]*16129+rf[13]*127+rf[14]
    Delay=rf[15]*16129+rf[16]*127+rf[17]
    Vpp=rf[18]*16129+rf[19]*127+rf[20]
    XInterval=rf[21]*16129+rf[22]*127+rf[23]
    YInterval=rf[24]*16129+rf[25]*127+rf[26]
    MoveTimes=rf[27]*16129+rf[28]*127+rf[29]
    Doppler=rf[30]*16129+rf[31]*127+rf[32]
    rf = rf[33:]/255*Vpp
    rf = rf.reshape(Aline, DataLength, int(rf.size/DataLength/Aline))
    rf = np.transpose(rf, (1, 0, 2))
    out = {'ScanSpeed':ScanSpeed,
    'Aline': Aline,
    'DataLength': DataLength,
    'SamplingRate': SamplingRate,
    'Delay': Delay,
    'Vpp': Vpp,
    'XInterval': XInterval,
    'YInterval': YInterval,
    'MoveTimes': MoveTimes,
    'Doppler': Doppler,
    'rf': rf}

    return out

# 將訓練結果輸出成.mat檔(如loss、time等)，並在matlab進一步輸出結果圖
def output_data(data, file_name='./temp'):
    matlabio.savemat(f'{file_name}.mat', data)
    return

# 輸出training progress
def show_training_result(output, label, ymin, ymax, cc, mse, yinterval=0.2):
    cc = cc
    mse = mse
    font_style = {'size': 24, 
        'family': 'Times New Roman', 
        'weight': 'bold'}
    label_style = {'labelsize': 30, 
        'labelweight': 'bold', 
        'linewidth': 3}
    data_style = {'linewidth': 3}
    legend_style = {'labelspacing': 0.3, 
        'edgecolor':'none', 
        'facecolor':'none', 
        'fontsize':20, 
        'loc': 'upper right'}
    plt.rc('font', **font_style)
    plt.rc('axes', **label_style)
    plt.rc('lines', **data_style)
    plt.rc('legend', **legend_style)

    plt.figure(figsize=(8,3))
    # x_axis = 0.1*np.arange(0, output.shape[0])
    plt.plot(0.1*np.arange(0, output.shape[0]), output)
    plt.plot(0.1*np.arange(0, label.shape[0]), label, 'r')
    plt.legend(['predicted', 'labeled'])

    plt.xlabel('Time (ms)')
    plt.ylabel('Phase (rad)')
    temp = plt.yticks(np.arange(ymin, ymax, yinterval))
    temp = plt.ylim([ymin, ymax])
    plt.title(f'CC: {np.round(cc, 3)}, MSE: {np.round(mse, 3)}', fontweight='bold')
