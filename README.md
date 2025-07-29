# IALA-LNN

IALA-LNN: Deep learning for peptide retention time prediction based on improved artificial lemming algorithm optimized liquid neural networks

## Introduction

Background and significance:

Liquid chromatography-tandem mass spectrometry (LC-MS/MS) is essential for proteomics analysis, where accurate peptide retention time (RT) prediction significantly improves identification accuracy and reduces false positive rates. Modern proteomics employs three primary chromatographic modes with distinct separation mechanisms: reversed-phase (RP) based on hydrophobic interactions, strong cation exchange (SCX) utilizing electrostatic forces, and hydrophilic interaction liquid chromatography (HILIC) leveraging polar interactions. The fundamental differences in these separation principles, combined with the complex nonlinear relationships between peptide sequences and their chromatographic behavior, necessitate sophisticated predictive models capable of adapting to multiple chromatographic systems. Additionally, retention time data often exhibit long-tailed distributions spanning multiple orders of magnitude, presenting significant challenges for achieving consistent prediction accuracy across all retention time ranges.

Result:

This study successfully developed an improved artificial learner algorithm-optimized liquid neural network framework (IALA-LNN), achieving high-precision prediction of peptide chromatographic retention times. The framework demonstrates outstanding performance across three mainstream separation modes—reverse-phase chromatography, strong cation exchange chromatography, and hydrophilic interaction chromatography—with coefficients of determination (R²) of 0.999, 0.998, and 0.997, respectively, significantly outperforming existing state-of-the-art methods. Particularly under the most challenging reverse-phase chromatographic conditions, compared to the current state-of-the-art Prosit method, prediction accuracy was improved by 26.2% and mean absolute error was reduced by 93.0%, fully demonstrating the effectiveness and superiority of the proposed methodology.

Model:

![image](https://github.com/chenyu001223/cy/blob/main/model.jpg)

# Related Files

| File Name | Description | 
|---------|---------|
| IALA_LNN.m   | Main program: IALA-optimized LNN   |
| IALA.m   | Improved Artificial Lampyridae Algorithm   |
| Goodnode.m     | Goodnode initialization method   |
|  sdata.mat   | Processed SCX data (dimension-reduced)   |
|  hdata.mat   | Processed HILIC data (dimension-reduced)   |
|  rdata.mat   | Processed RP data (dimension-reduced)  |
|  snet.mat     | Trained model weights for SCX   |
|  hnet.mat     | Trained model weights for HILIC   |
|  rnet.mat     | Trained model weights for RP   |
| SCX.txt   | Original SCX    |
| HILIC.txt   | Original HILIC    |
| RP.txt   | Original RP    |

# Requirements

MATLAB>=R2019b
Statistics and Machine Learning Toolbox
Optimization Toolbox 
Signal Processing Toolbox 

# How to use

1. quick Start - using pre-trained models

```matlab
TYPE = 0;  % Use pre-trained model
IALA_LNN   % Run main program
```

2. Training from Scratch
```matlab
TYPE = 1;
IALA_LNN   % Run main program
```
