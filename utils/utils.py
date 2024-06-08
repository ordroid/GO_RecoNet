from data.mri_dataset import SliceData
from torch.utils.data import DataLoader
from data.mri_dataset import DataTransform
from data import transforms
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
from train_results import FitResult


    
def create_datasets(args,resolution=320):
    '''This function creates the train and test datasets.
    You probably wouldn't need to change it'''
    
    train_data = SliceData(
        root=f"{args.data_path}/singlecoil_train",
        transform=DataTransform(resolution),
        split=1
    )
    dev_data = SliceData(
        root=f"{args.data_path}/singlecoil_val",
        transform=DataTransform(resolution),
        split = args.val_test_split,
        validation = True
    )
    test_data = SliceData(
        root=f"{args.data_path}/singlecoil_val",
        transform=DataTransform(resolution),
        split = args.val_test_split,
        validation = False
    )
    
    return train_data, dev_data, test_data


def create_data_loaders(args):
    '''Create train, validation and test datasets, and then out of them create the dataloaders. 
       These loaders will automatically apply needed transforms, as dictated in the create_datasets function using the transform parameter.'''
    train_data, dev_data, test_data = create_datasets(args)
    
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, dev_loader, test_loader



def freq_to_image(freq_data):
    ''' 
    This function accepts as input an image in the frequency domain, of size (B,320,320,2) (where B is batch size).
    Returns a tensor of size (B,320,320) representing the data in image domain.
    '''
    return transforms.complex_abs(transforms.ifft2_regular(freq_data))

def tensors_as_images(
    tensors, nrows=1, figsize=(8, 8), titles=[], wspace=0.1, hspace=0.2, cmap=None
):
    """
    Plots a sequence of pytorch tensors as images.

    :param tensors: A sequence of pytorch tensors, should have shape CxWxH
    """
    assert nrows > 0

    num_tensors = len(tensors)

    ncols = math.ceil(num_tensors / nrows)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        gridspec_kw=dict(wspace=wspace, hspace=hspace),
        subplot_kw=dict(yticks=[], xticks=[]),
    )
    axes_flat = axes.reshape(-1)

    # Plot each tensor
    for i in range(num_tensors):
        ax = axes_flat[i]

        image_tensor = tensors[i]
        assert image_tensor.dim() == 3  # Make sure shape is CxWxH

        image = image_tensor.numpy()
        image = image.transpose(1, 2, 0)
        image = image.squeeze()  # remove singleton dimensions if any exist

        # Scale to range 0..1
        min, max = np.min(image), np.max(image)
        image = (image - min) / (max - min)

        ax.imshow(image, cmap=cmap)

        if len(titles) > i and titles[i] is not None:
            ax.set_title(titles[i])

    # If there are more axes than tensors, remove their frames
    for j in range(num_tensors, len(axes_flat)):
        axes_flat[j].axis("off")

    return fig, axes


def dataset_first_n(
    dataset, n, show_classes=False, class_labels=None, random_start=True, **kw
):
    """
    Plots first n images of a dataset containing tensor images.
    """

    if random_start:
        start = np.random.randint(0, len(dataset) - n)
        stop = start + n
    else:
        start = 0
        stop = n

    # [(img0, cls0), ..., # (imgN, clsN)]
    first_n = list(itertools.islice(dataset, start, stop))

    # Split (image, class) tuples
    first_n_images, first_n_classes = zip(*first_n)

    if show_classes:
        titles = first_n_classes
        if class_labels:
            titles = [class_labels[cls] for cls in first_n_classes]
    else:
        titles = []

    return tensors_as_images(first_n_images, titles=titles, **kw)


def plot_fit(fit_res: FitResult, fig=None, log_loss=False, legend=None):
    """
    Plots a FitResult object.
    Creates four plots: train loss, test loss, train acc, test acc.
    :param fit_res: The fit result to plot.
    :param fig: A figure previously returned from this function. If not None,
        plots will the added to this figure.
    :param log_loss: Whether to plot the losses in log scale.
    :param legend: What to call this FitResult in the legend.
    :return: The figure.
    """
    if fig is None:
        fig, axes = plt.subplots(
            nrows=2, ncols=2, figsize=(16, 10), sharex="col", sharey=False
        )
        axes = axes.reshape(-1)
    else:
        axes = fig.axes

    for ax in axes:
        for line in ax.lines:
            if line.get_label() == legend:
                line.remove()

    p = itertools.product(["train", "test"], ["loss", "acc"])
    for idx, (traintest, lossacc) in enumerate(p):
        ax = axes[idx]
        attr = f"{traintest}_{lossacc}"
        data = getattr(fit_res, attr)
        h = ax.plot(np.arange(1, len(data) + 1), data, label=legend)
        ax.set_title(attr)
        if lossacc == "loss":
            ax.set_xlabel("Iteration #")
            ax.set_ylabel("Loss")
            if log_loss:
                ax.set_yscale("log")
                ax.set_ylabel("Loss (log)")
        else:
            ax.set_xlabel("Epoch #")
            ax.set_ylabel("Accuracy (%)")
        if legend:
            ax.legend()
        ax.grid(True)

    return fig, axes