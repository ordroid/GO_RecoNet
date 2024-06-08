import argparse
import shutil
import random
import torch
import numpy as np
from matplotlib import pyplot as plt
from utils.utils import create_data_loaders, freq_to_image, plot_fit, tensors_as_images
from models.GO_RecoNet import GO_RecoNet, UNet_loss, psnr_loss
import torch.optim as optim
import Trainer
import IPython.display
import os


def main():
    args = create_arg_parser().parse_args()  # get arguments from cmd/defaults
    # Data:
    train_loader, validation_loader, test_loader = create_data_loaders(
        args)  # get dataloaders
    in_size = train_loader.dataset[0][0].shape
    in_size = (1, in_size[0], in_size[1])

    # freeze seeds for result reproducability
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Hyperparameters:
    betas = (0.9, 0.999)

    # Model:
    print(f"Running on: {args.device}")
    print("Model")
    model = GO_RecoNet(drop_rate=args.drop_rate, device=args.device, learn_mask=args.learn_mask,
                       in_size=in_size).to(args.device)
    # Data parallel???

    # Optimizer:
    print("Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=betas)

    # Loss function:
    print("Loss function")
    def loss_fn(predicted, target):
        return UNet_loss(predicted, target)

    # Trainer:
    print("Trainer")
    trainer = Trainer.GOTrainer(
        model=model, loss_fn=loss_fn, optimizer=optimizer, device=args.device)

    # Training process:
    def post_epoch_fn(epoch, train_result, test_result, verbose, model, train_loader):
        if verbose:
            samples = None
            loaded_samples_ground_truth = None
            device = args.device
            rand_int = None
            with torch.no_grad():
                # context:
                # train_loader.dataset[i] contains a tuple where the first item is the freq_space image
                # and the second item is the ground truth image
                success = False
                while not success:
                    try:
                        # Generate a random integer between 0 and 1499
                        rand_int = random.randint(0, 1499)
                        # Attempt to get the image at rand_int
                        loaded_samples_freq = train_loader.dataset[rand_int][0].to(device)
                        loaded_samples_ground_truth = train_loader.dataset[rand_int][1].unsqueeze(0).to(device)
                        success = True
                    except Exception:
                        pass

                samples, subs = model.forward(loaded_samples_freq)
                samples = samples.squeeze(0).to(device)
                subs = subs.squeeze(0).to(device)

            # Detach and move to CPU for display purposes
            loaded_samples_ground_truth = loaded_samples_ground_truth.detach().cpu()
            samples = samples.detach().cpu()
            subs = subs.detach().cpu()

            fig, _ = tensors_as_images([loaded_samples_ground_truth, subs, samples], figsize=(6,2), titles=["Originial Image", "Subsampled Image", "Generated Image"], cmap='gray')
            plt.savefig(f'post_epoch_{epoch+1}_image_number_{rand_int}.jpg')
            plt.close(fig)
            

    print("Training process")
    checkpoint_file = 'checkpoints/go'
    checkpoint_file_final = f'{checkpoint_file}_final'

    if os.path.isfile(f'{checkpoint_file_final}.pt'):
        print(f'!!!! Not training but loading final checkpoint file!!!! {checkpoint_file_final}')
        checkpoint_file = checkpoint_file_final
    else:
        val_res = trainer.fit(dl_train=train_loader, dl_test=test_loader, num_epochs=args.num_epochs,
                              checkpoints=checkpoint_file, early_stopping=8, print_every=args.report_interval, post_epoch_fn=post_epoch_fn)

    saved_state = torch.load(f'{checkpoint_file}.pt')
    model.load_state_dict(saved_state['model_state'])

    
    # Report mean and std of PSNR for the train and test set and choose an image to plot target (fully sampled) and reconstructed one:
    def evaluate_psnr_and_plot(data_loader, model, chosen=0, set_name='train'):
        device = args.device  
        for i, (x,y) in enumerate(data_loader):
            with torch.no_grad():
                data = data_loader.dataset[i][0]
                ground_truth = data_loader.dataset[i][1].unsqueeze(0).to(device)
                data = data.to(device)
                output, _ = model(data)
                # psnr_value = psnr_loss(output, data_for_loss, min_pixel=min_pixel, max_pixel=max_pixel)
                # psnr_values.append(psnr_value.item())

            if i == chosen:  # Plot the chosen pair of original and reconstructed images
                #ground_truth = ground_truth.squeeze(0).detach().cpu()
                output_for_image = output.squeeze(0).detach().cpu()
                tensors_as_images([ground_truth, output_for_image], titles=["Original", "Reconstructed"], figsize=(6, 2), cmap='gray')
                plt.savefig(f'{set_name}_comparison_image_{chosen}.png')
                plt.close()

        # Compute the mean and standard deviation of the PSNR values.
        # psnr_mean = np.mean(psnr_values)
        # psnr_std = np.std(psnr_values)
        # print(f'{set_name} set: Mean PSNR = {psnr_mean}, Std PSNR = {psnr_std}')


    # Evaluate on train set
    evaluate_psnr_and_plot(train_loader, model, chosen = 4, set_name='train')

    # Evaluate on test set
    evaluate_psnr_and_plot(test_loader, model, chosen = 4, set_name='test')

    print("Done")


def create_arg_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0, help='Random Seed for reproducability.')
    parser.add_argument('--data-path', type=str, default='/datasets/fastmri_knee/', help='path to MRI dataset.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Pass "cuda" to use gpu')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of threads used for data handling.')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--report-interval', type=int, default=10, help='Report training stats once per this much iterations.')
    parser.add_argument('--drop-rate', type=float, default=0.8, help='Percentage of data to drop from each image (dropped in freq domain).')
    parser.add_argument('--learn-mask', action='store_true', default=False, help='Whether to learn subsampling mask')
    parser.add_argument('--results-root', type=str, default='results', help='result output dir.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learn rate for your reconstruction model.')
    parser.add_argument('--mask-lr', type=float, default=0.01, help='Learn rate for your mask (ignored if the learn-mask flag is off).')
    parser.add_argument('--val-test-split', type=float, default=0.3, help='Portion of test set (NOT of the entire dataset, since train-test split is pre-defined) to be used for validation.')
    
    return parser
    


if __name__ == "__main__":
    print("Starting now")
    main()




















    # #go over the train and test sets to find minimum and maximum pixels for psnr calculation:
    # def find_min_max_pixel(data_loader):
    #     min_pixel, max_pixel = float('inf'), -float('inf')
    #     for data, _ in data_loader:
    #         min_pixel = min(min_pixel, data.min().item())
    #         max_pixel = max(max_pixel, data.max().item())
    #     return min_pixel, max_pixel

    # min_pixel_train, max_pixel_train = find_min_max_pixel(train_loader)
    # min_pixel_test, max_pixel_test = find_min_max_pixel(test_loader)
    # print("min and max pixels in train:  " + str(min_pixel_train) + " " + str(max_pixel_train))
    # print("min and max pixels in test:  " + str(min_pixel_test) + " " + str(max_pixel_test))