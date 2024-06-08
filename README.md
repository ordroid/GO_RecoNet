# GO_RecoNet
GO_RecoNet: A deep learning model designed to reconstruct low-quality MRI images using our enhanced U-Net architecture.

## installation
We used conda package manager. The repository has an environment.yml file included.
After installing and activating the environment, run script.sh file which includes the needed commands for training/predicting.

## Overview
Note: This is the shortened version of our documentation. For more detailed information, please check out the PDF named: [ReadMe-More_Info.pdf](/ReadMe-More_Info.pdf).

In this project, we considered some reconstruction architectures such as U-Net, deep residual network, Transformers, SRGAN (a version of GAN), and custom CNNs. The chosen network was U-Net with skip connections, detailed as follows:

- **Input**: First passes through a sampling layer.
- **U-Net Architecture**:  
  - **Encoding Phase**: Involves sequential application of encoder blocks each consisting of a convolutional layer (kernel size=5, stride=2), normalization, ReLU, and max pooling (kernel size=2).
    <p align="center">
      <img src="/images/encoder.png" alt="Encoder" width="600">
    </p>
  - **Bottleneck**: The input undergoes max pooling, a 1x1 convolution, and interpolation, followed by concatenation + convolution to reduce the features.
    <p align="center">
      <img src="/images/bottleneck.png" alt="Bottleneck" width="600">
    </p>
  - **Decoding Phase**: Sequentially applies decoder blocks with up-sampling, convolutional layers (kernel size=5, stride=2), normalization, ReLU, incorporating skip connections, and finally, an interpolation produces the output image.
    <p align="center">
      <img src="/images/decoder.png" alt="Decoder" width="600">
    </p>

## Why We think that this was a good choice for us?

U-Net is highly favored for medical image reconstruction due to its effective encoder-decoder structure with skip connections, which excel at capturing both high-level and local contextual information which is especially crucial in medical analysis as can be seen [here](https://ar5iv.labs.arxiv.org/html/2211.14830).
Unlike ResNet, which focuses on layer-wise residual learning, U-Net’s symmetric expanding path allows for precise localization and reconstruction of fine details from low-quality inputs, making it very good for tasks requiring exact structural replication. Its skip connections, which bridge the encoder directly to the decoder can significantly enhance information flow and help to preserve spatial hierarchies, often lost in deeper networks like deep residual networks.
While transformers are adept at capturing global dependencies, they might neglect in learning local details which in our opinion is a very crucial part in MRI reconstruction.
Additionally, transformers might require substantially higher computational resources, which can be an obstacle to us with our given time and resources, In our opinion U-Net can be more resource-efficient because it can be very adapted to the task.
SRGAN aims to enhance perceptual realism but may compromise clinical accuracy, introducing undesirable artifacts through adversarial training. U-Net minimizes this issue by being trained to closely match the ground truth, thus maintaining high fidelity to original medical images. It can also offer more stable and predictable training compared to the often-challenging training dynamics of GANs.
Custom CNNs can be specifically tailored but require extensive experimentation to optimize, whereas U-Net provides a reliable, proven framework with less need for customization, making it particularly advantageous when working within time constraints. It strikes an optimal balance between extracting deep features and utilizing surface-level information, and might avoid unnecessary complexity and computational burdens as can be partially seen [here](https://www.sciencedirect.com/science/article/pii/S1877050923003976).


## Chosen Loss Criteria

The chosen loss criteria for the GO_RecoNet model is MSE (Mean Squared Error) to ensure both accurate and high-quality MRI reconstructions. Initially, we experimented with both MSE and PSNR as loss functions, combining them through tuning of coefficients. We took the PSNR loss as 1/PSNR to be able to minimize it along with the MSE. The use of the combined loss criteria did not yield superior results compared to using MSE alone. This can be explained because PSNR is inherently dependent on MSE, and using both doesn’t provide more information.

## Results Summary

- **PSNR - Mean and std across train and test sets**:
### Train Set Results

| Drop rate | Learned mask (Mean ± Std) | Non-Learned mask (Mean ± Std) |
|-----------|----------------------------|-------------------------------|
| 0.2       | 32.60985 ± 1.39275         | 28.64482 ± 1.2367             |
| 0.4       | 31.8118 ± 1.36219          | 26.48024 ± 1.24633            |
| 0.6       | 30.66749 ± 1.38163         | 24.07635 ± 1.21372            |

### Test Set Results

| Drop rate | Learned mask (Mean ± Std) | Non-Learned mask (Mean ± Std) |
|-----------|----------------------------|-------------------------------|
| 0.2       | 31.32835 ± 2.03575         | 27.231 ± 1.63954              |
| 0.4       | 30.64634 ± 1.95406         | 25.00568 ± 1.57326            |
| 0.6       | 29.35582 ± 1.76597         | 22.55603 ± 1.48975            |

## Visualization of the results

## Drop Rate 0.2 with a Learnable Mask

<p align="center">
  <img src="/images/02_with.jpg" alt="Original, Subsampled, Reconstructed - Learnable Mask" width="400">
</p>
<p align="center">
  <img src="/images/02_with_graph.jpg.png" alt="Training Graph - Learnable Mask" width="400">
</p>

## Drop Rate 0.2 without a Learnable Mask

<p align="center">
  <img src="/images/02_no.jpg" alt="Original, Subsampled, Reconstructed - Non-Learnable Mask" width="400">
</p>
<p align="center">
  <img src="/images/02_no_graph.png" alt="Training Graph - Non-Learnable Mask" width="400">
</p>

## Drop Rate 0.4 with a Learnable Mask

<p align="center">
  <img src="/images/04_with.jpg" alt="Original, Subsampled, Reconstructed - Learnable Mask" width="400">
</p>
<p align="center">
  <img src="/images/04_with_graph.png" alt="Training Graph - Learnable Mask" width="400">
</p>

## Drop Rate 0.4 without a Learnable Mask

<p align="center">
  <img src="/images/04_no.jpg" alt="Original, Subsampled, Reconstructed - Non-Learnable Mask" width="400">
</p>
<p align="center">
  <img src="/images/04_no_graph.png" alt="Training Graph - Non-Learnable Mask" width="400">
</p>

## Drop Rate 0.6 with a Learnable Mask

<p align="center">
  <img src="/images/06_with.jpg" alt="Original, Subsampled, Reconstructed - Learnable Mask" width="400">
</p>
<p align="center">
  <img src="/images/06_with_graph.png" alt="Training Graph - Learnable Mask" width="400">
</p>

## Drop Rate 0.6 without a Learnable Mask

<p align="center">
  <img src="/images/06_no.jpg" alt="Original, Subsampled, Reconstructed - Non-Learnable Mask" width="400">
</p>
<p align="center">
  <img src="/images/06_no_graph.png" alt="Training Graph - Non-Learnable Mask" width="400">
</p>

## Future Work

We will suggest experimenting with the following ideas:
- **Transformers**: Explore the use of vision transformers and Swin transformers to improve global dependency capture.
- **Hybrid Models**: U-Net with transformers or U-net with GANs can be used to leverage both structural fidelity and perceptual quality.
- **Hyperparameter Tuning**: Implement advanced techniques like Bayesian optimization or neural architecture search to find optimal hyperparameters.
- **Improved Loss Function**: Conduct experiments with perceptual loss or some variation of multi-scale loss.
- **Mixed Precision Training**: Speed up the training process and allow for handling larger batch sizes without compromising performance.

