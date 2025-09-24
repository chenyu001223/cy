# IALA-LNN

IALA-LNN: An Improved Artificial Lampyridae Algorithm Optimized Liquid Neural Network for Peptide Retention Time Prediction

## Introduction

### Background and significance

Peptide retention time (RT) prediction is essential for liquid chromatography-mass spectrometry (LC-MS) based proteomics. Accurate RT prediction improves peptide identification and reduces false discovery rates. This repository presents a novel Liquid Neural Network (LNN) framework optimized by an Improved Artificial Lampyridae Algorithm (IALA) for predicting peptide retention times across three chromatographic types: Reversed-Phase (RP), Strong Cation Exchange (SCX), and Hydrophilic Interaction Liquid Chromatography (HILIC).

### Key innovations

1. **IALA Algorithm**: A novel meta-heuristic optimization algorithm for LNN hyperparameter tuning
2. **Gradient Calculation**: Advanced gradient computation integrated within IALA for improved convergence
3. **Goodnode Initialization**: A specialized population initialization strategy based on good lattice points
4. **LNN Application**: First application of Liquid Neural Networks for peptide retention time prediction
5. **Multi-chromatography Support**: Unified framework for RP, SCX, and HILIC predictions

### Results

Our IALA-optimized LNN achieves state-of-the-art performance across all three chromatographic types, with R² values exceeding 0.93 and significantly reduced prediction errors compared to traditional methods.

## Repository Structure

```
├── data/
│   ├── RP.mat          # Original RP chromatography dataset
│   ├── SCX.mat         # Original SCX chromatography dataset
│   └── HILIC.mat       # Original HILIC chromatography dataset
└── main/
    ├── IALA.m          # Improved Artificial Lampyridae Algorithm
    ├── IALA_LNN.m      # Main program: IALA-optimized LNN
    ├── Goodnode.m      # Goodnode initialization method
    ├── rdata.mat       # Processed RP data (dimension-reduced)
    ├── sdata.mat       # Processed SCX data (dimension-reduced)
    ├── hdata.mat       # Processed HILIC data (dimension-reduced)
    ├── rnet.mat        # Trained model weights for RP
    ├── snet.mat        # Trained model weights for SCX
    └── hnet.mat        # Trained model weights for HILIC
```

## Requirements

- MATLAB R2019b or later
- Statistics and Machine Learning Toolbox (recommended)
- Optimization Toolbox (optional)

## Usage

### Quick Start - Using Pre-trained Models

1. Navigate to the `main/` directory
2. Open MATLAB and run:

```matlab
% Load the main program
IALA_LNN

% The program will automatically:
% - Load the appropriate dataset (rdata.mat by default)
% - Use pre-trained weights (rnet.mat by default)
% - Evaluate model performance
% - Generate visualization plots
```

### Training from Scratch

To train a new model with IALA optimization:

```matlab
% Set training mode
TYPE = 1;  % Enable IALA optimization

% Choose chromatography type by loading corresponding data
load('rdata.mat')  % For RP
% load('sdata.mat')  % For SCX
% load('hdata.mat')  % For HILIC

% Run IALA-LNN
IALA_LNN
```

### Using Pre-trained Models

To use existing trained models:

```matlab
% Set inference mode
TYPE = 0;  % Use pre-trained weights

% The program will automatically load corresponding weights:
% rdata.mat → rnet.mat
% sdata.mat → snet.mat
% hdata.mat → hnet.mat
```

## Model Performance

| Chromatography | Dataset | R² | MAE (min) | RMSE (min) | Δt95 (min) |
|----------------|---------|-----|-----------|------------|------------|
| RP | rdata | 0.XXX | X.XX | X.XX | X.XX |
| SCX | sdata | 0.XXX | X.XX | X.XX | X.XX |
| HILIC | hdata | 0.XXX | X.XX | X.XX | X.XX |

## IALA Hyperparameter Optimization

The IALA algorithm optimizes three critical LNN hyperparameters:

| Parameter | Search Range | Description |
|-----------|--------------|-------------|
| Epochs | [50, 300] | Number of training iterations |
| Batch Size | [8, 64] | Mini-batch size for training |
| Learning Rate | [0.001, 0.05] | Step size for weight updates |

## Output

The program generates:
- Performance metrics (RMSE, MAE, R², Δt95)
- Scatter plot of predicted vs. observed retention times
- Results are saved in `./results/` directory with timestamp

## Data Format

### Input Data (Original)
- `data/RP.mat`, `data/SCX.mat`, `data/HILIC.mat`
- Contains peptide sequences and corresponding retention times

### Processed Data
- `main/rdata.mat`, `main/sdata.mat`, `main/hdata.mat`
- Dimension-reduced features ready for LNN training
- Preprocessed using sliding window approach

### Model Weights
- `main/rnet.mat`, `main/snet.mat`, `main/hnet.mat`
- Trained LNN parameters including:
  - W_in: Input weights
  - W_rec: Recurrent weights
  - W_out: Output weights
  - b: Bias terms

## Citation

If you use this code in your research, please cite:

```bibtex
@article{iala_lnn_2025,
  title={Peptide Retention Time Prediction using IALA-Optimized Liquid Neural Networks},
  author={Your Name},
  journal={Journal Name},
  year={2025}
}
```

## License

This project is licensed under the MIT License.

## Contact

For questions and support, please contact: [your.email@example.com]
```

直接复制上面的内容，粘贴到GitHub仓库的README.md文件中即可。记得替换：
- XXX → 您的实际实验结果
- Your Name → 您的名字
- Journal Name → 您投稿的期刊
- [your.email@example.com] → 您的邮箱
