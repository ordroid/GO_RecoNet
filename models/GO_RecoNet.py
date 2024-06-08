import torch
from models.subsampling import SubsamplingLayer
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import freq_to_image



class GO_RecoNet(torch.nn.Module):
    def __init__(self, drop_rate, device, learn_mask, in_size):
        super().__init__()
        # initialize subsampling layer - use this in your own model
        self.subsample = SubsamplingLayer(drop_rate, device, learn_mask)
        self.UNet = UNet(in_size, device).to(device)
        self.learn_mask = learn_mask

    def forward(self, x):
        # get subsampled input in image domain - use this as first line in your own model's forward
        x_sub = self.subsample(x)
        x = self.UNet.forward(x_sub)
        return x, x_sub


class UNet(nn.Module):
    def __init__(self, in_size, device):
        """
        :param in_size: The size of one input (without batch dimension).
        """
        super().__init__()
        self.device = device

        self.e1 = EncoderBlock(1, 32, True, True).to(device)
        self.e2 = EncoderBlock(32, 64, True, True).to(device)

        self.bn = EncoderBlock(64, 128, False, False).to(device)
        #here

        self.d2 = DecoderBlock(128, 64, True, True).to(device)
        self.sc2 = torch.nn.Conv2d(128, 64, kernel_size = 1).to(device) #skip connection
        self.d1 = DecoderBlock(64, 32, True, True).to(device)
        self.sc1 = torch.nn.Conv2d(64, 32, kernel_size = 1).to(device) #skip connection
        self.d0 = DecoderBlock(32, 1, True, True).to(device)

    def encode(self, x):
        x1 = self.e1(x) #1 -> 32
        x2 = self.e2(x1) #32 -> 64
        return x1, x2

    # def decode(self, x1, x2, x3):
    #     #prep = self.fc_prep(z)
    #     #prep_reshaped = prep.view(-1, *self.features_shape)
    #     partially_decoded_features = self.features_decoder_first(fully_extracted_features)
    #     partially_decoded_features_full = torch.cat([partially_decoded_features, partially_extracted_features], dim=1) 
    #     fully_extracted_features_full = self.features_decoder_second(partially_decoded_features_full)
    #     return torch.tanh(fully_extracted_features_full)

    def forward(self, x):
        #Encoder part
        x1, x2 = self.encode(x)

        #Bottle neck part
        x3 = self.bn(x2) #64 -> 128
        x3_aux = F.max_pool2d(x3, kernel_size=2, stride=2).to(self.device)
        x3_aux = (torch.nn.Conv2d(128, 128, kernel_size=1).to(self.device))(x3_aux)
        x3_aux = F.interpolate(x3_aux, size=x3.shape[2:], mode="bilinear", align_corners=False).to(self.device) #so that x3_aux by with same height and width as x3
        x3 = (torch.nn.Conv2d(256, 128, kernel_size=1).to(self.device))(torch.cat([x3, x3_aux], dim=1)) #256 -> 128


        #Decoder + skip connections part
        x4 = self.d2(x3) #128 -> 64
        x2_aux = torch.nn.functional.interpolate(x2, size=(x4.size(2), x4.size(3)), mode="nearest").to(self.device) #making x2 to be same height+width as x4
        x4 = self.sc2(torch.cat([x4, x2_aux], dim=1)) #128 -> 64

        x5 = self.d1(x4) #64 -> 32
        x1_aux = torch.nn.functional.interpolate(x1, size=(x5.size(2), x5.size(3)), mode="nearest").to(self.device) #making x1 to be same height+width as x5
        x5 = self.sc1(torch.cat([x5, x1_aux], dim=1)) #64 -> 32

        x6 = self.d0(x5) #32 -> 1
        x6 = torch.nn.functional.interpolate(x6, size=(320, 320), mode = "nearest").to(self.device)
        return x6

#----------------------------------------------
#-----------------Loss section-----------------
#----------------------------------------------

def MSE_Loss(predicted, target):
    """
        Just a regular everyday normal MSE loss that we all love
    """
    #predicted (1, 1, 320, 320)
    aux_target = target.permute(0, 2, 3, 1) #(B, 2, 320, 320) -> (B, 320, 320, 2)
    aux_target = freq_to_image(aux_target) #(B, 320, 320, 2) -> (B, 320, 320)
    aux_target = aux_target.unsqueeze(1) #(B, 320, 320) -> (B, 1, 320, 320)
    loss = torch.mean((predicted - aux_target) ** 2) 
    return loss

def psnr_loss(predicted, target, min_pixel = 0, max_pixel = 255.0):
    #Calculate min and max pixel values in the batch of images in target
    mse = MSE_Loss(predicted, target)
    mse = mse.detach().cpu().numpy()
    if mse == 0:
        return 100
    psnr_loss = 20 * np.emath.log10((max_pixel-min_pixel) / np.emath.sqrt(mse))
    return torch.tensor(psnr_loss)

def UNet_loss(predicted, target, MSE_reg = 1, PSNR_reg = 0):
    """
        Returns weighted loss with mse part and psnr part
    """
    loss_mse = MSE_reg * MSE_Loss(predicted, target)
    loss_psnr = PSNR_reg * (1/psnr_loss(predicted, target))

    target_temp = target.permute(0, 2, 3, 1) #(B, 2, 320, 320) -> (B, 320, 320, 2)
    target_temp = freq_to_image(target_temp) #(B, 320, 320, 2) -> (B, 320, 320)
    target_temp = target_temp.unsqueeze(1) #(B, 320, 320) -> (B, 1, 320, 320)
    min_p = target_temp.min().item()
    max_p = target_temp.max().item()
    psnr_values = []
    batch_size = target.shape[0]
    for i in range(batch_size):
        target_image = target[i].unsqueeze(0)
        predicted_image = predicted[i].unsqueeze(0)
        psnr_value = psnr_loss(predicted_image, target_image, min_pixel=min_p, max_pixel=max_p)
        psnr_values.append(psnr_value)
        
    psnr_for_statistics = np.mean(psnr_values)
    return (loss_mse + loss_psnr , float(psnr_for_statistics)) #returns MSE + average PSNR for this batch


#----------------------------------------------
#--------------Encoder & Decoder---------------
#----------------------------------------------

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_norm, is_pooling):
        super().__init__()

        modules = []
        modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2))
        if is_norm: # add normalization if needed
            modules.append(nn.BatchNorm2d(out_channels))
        # add an appropriate activation function
        modules.append(nn.ReLU())
        if is_pooling: # add pooling if needed
            modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_norm, is_pooling):
        super().__init__()

        modules = []
        if is_pooling: #add upsampling if needed
            modules.append(nn.Upsample(scale_factor=2, mode='nearest'))
        modules.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1))
        if is_norm: # add normalization if needed
            modules.append(nn.BatchNorm2d(out_channels))
        modules.append(nn.ReLU())
        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        return self.cnn(h)  #maybe add a wrapping tanh activation?

