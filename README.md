# Division and Union: Latent Model Watermarking


This is the code implementation (pytorch) for our paper:  
[Division and Union: Latent Model Watermarking](https://ieeexplore.ieee.org/document/11153578)

We propose a latent model watermarking, constructing upon the model Division and Union operating concept, dubbed as DUO, 
leveraging the strengths of two watermarking methods above while eliminating each shortcoming. 
Once the model owner or provider embeds a watermark into the model using watermark data, 
the watermarked model is divided into two parts: the main model, which corresponds to the primary task and is made publicly available, 
and a small sub-network privately reserved by the owner. 
The watermark resides latently within the main model and can only be activated through the private sub-network(the reserved parameters) when they are united. 
Consequently, DUO does not adversely affect the performance of the main model on its primary task and does not induce any security risks, 
even in the presence of watermark data. We extensively validate DUO on four benchmark datasets 
(CIFAR-10, ImageNette, CIFAR-100, and Tiny-ImageNet) using various model architectures, including standardized ResNet and VGG. 

## Required python packages
Our code is tested under the following environment: Python 3.9, torch 2.5.1, torchvision 0.20.1, numpy 1.26.3, opencv-python 4.10.0.84.



## Quick Start

1. Load dataset and watermark data:  
    e.g. `CIFAR10` on `ResNet18` 
    ```python3
    python load_data.py
    ```

2. Train a watermark model:  
    e.g. `CIFAR10` on `ResNet18` 
    ```python3
    python model_marking.py
    ```

3. Test watermark accuracy:  
    e.g. `CIFAR10` on `ResNet18`
    ```python3
    python model_certification.py
    ```

## Cite our paper
```
@ARTICLE{dai2025division,
  author={Dai, Zhiyang and Gao, Yansong and Kuang, Boyu and Zheng, Yifeng and Mian, Ajmal and Wang, Ruimin and Fu, Anmin},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Division and Union: Latent Model Watermarking}, 
  year={2025},
  volume={20},
  number={},
  pages={9523-9538},
  keywords={Watermarking;Data models;Adaptation models;Training;Fingerprint recognition;Analytical models;Robustness;IP networks;Glass box;Computational modeling;Deep learning;model watermark;IP protection},
  doi={10.1109/TIFS.2025.3607234}}
```
