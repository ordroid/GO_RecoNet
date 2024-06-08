# GO_RecoNet
GO_RecoNet: A deep learning model designed to reconstruct low-quality MRI images using our enhanced U-Net architecture.

## installation
We used conda package manager. The repository has an environment.yml file included.
After installing and activating the environment, run script.sh file which includes the needed commands for training/predicting.

## Overview

In this project, we considered some reconstruction architectures such as U-Net, deep residual network, Transformers, SRGAN (a version of GAN), and custom CNNs. The chosen network was U-Net with skip connections, detailed as follows:

- **Input**: First passes through a sampling layer.
- **U-Net Architecture**:  
  - **Encoding Phase**: Involves sequential application of encoder blocks each consisting of a convolutional layer (kernel size=5, stride=2), normalization, ReLU, and max pooling (kernel size=2).
![Encoder](/images/encoder.png)
  - **Bottleneck**: The input undergoes max pooling, a 1x1 convolution, and interpolation, followed by concatenation + convolution to reduce the features.
![Bottleneck](/images/bottleneck.png)
  - **Decoding Phase**: Sequentially applies decoder blocks with up-sampling, convolutional layers (kernel size=5, stride=2), normalization, ReLU, incorporating skip connections, and finally, an interpolation produces the output image.
![Decoder](/images/decoder.png)

## Why We think that this was a good choice for us?

U-Net is highly favored for medical image reconstruction due to its effective encoder-decoder structure with skip connections, which excel at capturing both high-level and local contextual information. This is especially crucial in medical analysis. Unlike ResNet, which focuses on layer-wise residual learning, U-Net’s symmetric expanding path allows for precise localization and reconstruction of fine details from low-quality inputs, making it very good for tasks requiring exact structural replication. Its skip connections, which bridge the encoder directly to the decoder, can significantly enhance information flow and help to preserve spatial hierarchies often lost in deeper networks like deep residual networks.

## Chosen Loss Criteria

The chosen loss criteria for the GO_RecoNet model is MSE (Mean Squared Error) to ensure both accurate and high-quality MRI reconstructions. Initially, we experimented with both MSE and PSNR as loss functions, combining them through tuning of coefficients. We took the PSNR loss as 1/PSNR to be able to minimize it along with the MSE. The use of the combined loss criteria did not yield superior results compared to using MSE alone. This can be explained because PSNR is inherently dependent on MSE, and using both doesn’t provide more information.

## Results Summary

- **PSNR - Mean and std across train and test sets**:
  - **Train set**:
    - **Drop rate 0.2**: 32.60985±1.39275
    - **Drop rate 0.4**: 31.8118±1.36219
    - **Drop rate 0.6**: 30.66749±1.38163
  - **Test set**:
    - **Drop rate 0.2**: 31.32835±2.03575
    - **Drop rate 0.4**: 30.64634±1.95406
    - **Drop rate 0.6**: 29.35582±1.76597

## Future Work

Assuming we had a year to work on this project, we will suggest experimenting with the following ideas:
- **Transformers**: Explore the use of vision transformers and Swin transformers to improve global dependency capture.
- **Hybrid Models**: U-Net with transformers or U-net with GANs can be used to leverage both structural fidelity and perceptual quality.
- **Hyperparameter Tuning**: Implement advanced techniques like Bayesian optimization or neural architecture search to find optimal hyperparameters.
- **Improved Loss Function**: Conduct experiments with perceptual loss or some variation of multi-scale loss.
- **Mixed Precision Training**: Speed up the training process and allow for handling larger batch sizes without compromising performance.

## Visualization of Results

![Training Graph for Drop rate 0.2 with Learnable Mask](/path/to/image1.png)
