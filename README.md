# 超音波剪切波訊號之智慧預測 | Smart Prediction of Ultrasound Shear Wave
本專案開發一套剪切波影像預測流程，採用名為 CAM-GRU（Correlation and Attention Modulated GRU）的 GRU 架構模型。該模型整合IQ訊號處理、互相關增強、注意力機制與positional embedding。相較傳統方法，達到 0.8 的相關係數，並使預測時間降低77%。  
A shear wave image prediction pipeline was developed using a GRU-based model named CAM-GRU. The model incorporates IQ signal processing, cross-correlation enhancement, attention mechanism, and positional embedding. Compared with the conventional method, it achieved a correlation coefficient of 0.8 and reduced prediction time by 77%.
---

## 🔧 Features
- Model: `CAM_GRU` (from `models/cam_gru.py`)
- Inputs: Concatenated raw input & IQ magnitude
- Enhanced by: Cross-correlation features & positional encoding
- Objective: Minimize combined loss (0.4 × MSE + 0.6 × cross-correlation loss)
- Achieved: Correlation ≈ 0.8 with ground-truth shear wave images; 73% reduction in computation time
---

## 📦 Requirements
```bash
pip install -r requirements.txt
```
---

## 🚀 Usages
```bash
python scripts/train_cam_gru.py --data_dir ./samples --epochs 200 --batch_size 30
```
Models will be saved to `outputs/models/`
---

## 🧪 Example Outputs
`.\outputs\models\CAM_GRU_architecture.jpg`: architecture of CAM_GRU  
`.\outputs\models\*.py`: trained model will be saved as this  
`.\outputs\plots\sw_prediction.jpg`: sample predicted SW by CAM_GRU  
Coming soon: Training curve
---

## 📁 File Structure 
```python
cam_gru_shearwave/
├── data/
│   └── dataset.py  # Dataset loader
├── models/
│   └── cam_gru.py  # CAM_GRU model with attention & positional encoding
├── outputs/
│   ├── models/     # Saved models
│   └── plots/      # Saved result plots
├── samples/        # Sample training/validation data (I, Q, labels)
│   ├── training/
│   └── validation/
├── scripts/
│   └── train_cam_gru.py    # Main training script
├── utils/          # Loss functions, I/O helpers, preprocessing
│   ├── losses.py
│   ├── preprocessing.py
│   └── io_utils.py
├── requirements.txt
└── README.md
```
---

## 📎 Notes
- If using your own data, ensure the structure matches the sample set
- IQ signal preprocessing is handled by `utils/preprocessing.py` and `utils/io_utils.py`
- Cross-correlation features are calculated dynamically during training
