import torch
import numpy as np
import statistics
import einops
import os
import argparse

import tqdm
from opencood.debug.eval_integration import show_pred_gt
import opencood.hypes_yaml.yaml_utils as yaml_utils 

from opencood.tools import train_utils

from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from opencood.models.delay import build_delay_module

torch.manual_seed(42)
np.random.seed(42)

BASE_PATH = "/equilibrium/students/svatamanelu/"

# -------------- SSIM and MS-SSIM implementation from https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py --------------
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor


def _fspecial_gauss_1d(size: int, sigma: float) -> Tensor:
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input: Tensor, win: Tensor) -> Tensor:
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def _ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float,
    win: Tensor,
    size_average: bool = True,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03)
) -> Tuple[Tensor, Tensor]:
    r""" Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        data_range (float or int): value range of input images. (usually 1.0 or 255)
        win (torch.Tensor): 1-D gauss kernel
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    nonnegative_ssim: bool = False,
) -> Tensor:
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu

    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    #if not X.type() == Y.type():
    #    raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


def ms_ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    weights: Optional[List[float]] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03)
) -> Tensor:
    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    #if not X.type() == Y.type():
    #    raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if len(X.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(X.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (
        2 ** 4
    ), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights_tensor = X.new_tensor(weights)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    levels = weights_tensor.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, size_average=False, K=K)

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = [s % 2 for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, padding=padding)
            Y = avg_pool(Y, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # type: ignore  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights_tensor.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)


class SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range: float = 255,
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        channel: int = 3,
        spatial_dims: int = 2,
        K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
        nonnegative_ssim: bool = False,
    ) -> None:
        r""" class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        return ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
        )


class MS_SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range: float = 255,
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        channel: int = 3,
        spatial_dims: int = 2,
        weights: Optional[List[float]] = None,
        K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    ) -> None:
        r""" class for ms-ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """

        super(MS_SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        return ms_ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            weights=self.weights,
            K=self.K,
        )
# -------------- End of SSIM and MS-SSIM implementation --------------


def parse_args():
    parser = argparse.ArgumentParser(description='Moving MNIST Predictor')
    parser.add_argument('--arch',type=str, help='Architecture to use for prediction')
    parser.add_argument('--residual', action=argparse.BooleanOptionalAction, help='Whether to use residual connection')
    parser.add_argument('--bidirectional', action=argparse.BooleanOptionalAction, help='Use bidirectional Mamba blocks')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for Mamba blocks')
    parser.add_argument('--loss', type=str, default='l1', help='Loss function to use')
    parser.add_argument('--layers', type=int, default=5, help='Number of Mamba blocks to use')

    parser.add_argument('--save_ckpt', action='store_true', help='Save checkpoint during training')
    parser.add_argument('--loss_coefficient', type=float, default=1.0, help='Coefficient for the loss function')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')

    args = parser.parse_args()
    return args

def show_pred_gt(pred, gt, input_frames, iteration, save_path="/equilibrium/students/svatamanelu/moving_mnist_predictor_results"):
    #pred = pred.cpu().detach().numpy()
    #gt = gt.cpu().detach().numpy()

    # pred and gt shape: [batch, 10, 1, 64, 64]
    # we will visualize the first sample in the batch
    input_sample = input_frames[0]  # shape: [batch, 10, 1, 64, 64]
    pred_sample = pred[0]  # shape: [10, 1, 64, 64]
    gt_sample = gt[0]      # shape: [10, 1, 64, 64]

    # save the predicted and ground truth frames as images one beside the other for comparison
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1 * pred_sample.shape[0], 3, figsize=(12, 4 * pred_sample.shape[0]))
    
    axes[0][0].set_title(f'Input Frame')
    axes[0][1].set_title(f'Predicted Frame')
    axes[0][2].set_title(f'Ground Truth Frame')

    for i in range(pred_sample.shape[0]):
        input_frame = input_sample[i, 0] * 255.0  # scale back to [0, 255]
        pred_frame = pred_sample[i, 0] * 255.0  # scale back to [0, 255]
        gt_frame = gt_sample[i, 0] * 255.0        # scale back to [0, 255]

        input_frame = input_frame.cpu().detach().numpy().astype(np.uint8)
        pred_frame = pred_frame.cpu().detach().numpy().astype(np.uint8)
        gt_frame = gt_frame.cpu().detach().numpy().astype(np.uint8)

        # Save the frames side by side using matplotlib
        axes[i][0].imshow(input_frame, cmap='gray')
        axes[i][0].axis('off')
        axes[i][1].imshow(pred_frame, cmap='gray')
        axes[i][1].axis('off')
        axes[i][2].imshow(gt_frame, cmap='gray')
        axes[i][2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{iteration}.png'))
    plt.close()
        
class MovingMNIST(Dataset):
    """
    Moving MNIST Dataset
    Args:
        root (string): Path to the .npy file.
        is_train (bool): Whether to split for training or testing.
        n_frames_input (int): Number of frames used as input (history).
        n_frames_output (int): Number of frames to predict (future).
        transform (callable, optional): Optional transform to be applied.
    """
    def __init__(self, root, is_train=True, n_frames_input=5, n_frames_output=5, transform=None, seed=42):
        super(MovingMNIST, self).__init__()

        # Load the data: shape is usually (sequence_length, n_sequences, height, width)
        # e.g., (20, 10000, 64, 64)
        self.dataset = np.load(root)
        self.seed = seed
        np.random.seed(self.seed)
        
        # Transpose to: [n_sequences, sequence_length, height, width]
        self.dataset = self.dataset.transpose(1, 0, 2, 3) 
        
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.total_frames = n_frames_input + n_frames_output
        self.transform = transform

        # Basic 80/20 train/test split
        split_idx = int(len(self.dataset) * 0.8)
        if is_train:
            self.dataset = self.dataset[:split_idx]
        else:
            self.dataset = self.dataset[split_idx:]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Extract full sequence: [20, 64, 64]
        video = self.dataset[idx]
        
        # Add channel dimension: [20, 1, 64, 64]
        video = video[:, np.newaxis, :, :]
        
        # Normalize to [0, 1] and convert to float32
        video = video.astype(np.float32) / 255.0
        
        # return the differences between consecutive frames as input and the future frames as target

        # Split into input (history) and target (future) sequences
        input_frames = video[:self.n_frames_input]
        target_frames = video[self.n_frames_input:self.total_frames]

        if self.transform:
            input_frames = self.transform(input_frames)
            target_frames = self.transform(target_frames)

        return torch.from_numpy(input_frames), torch.from_numpy(target_frames)

if __name__ == '__main__':
    args = parse_args()
    print(args)

    res_string = 'residual' if args.residual else 'no_residual'
    bidir_string = 'bidirectional' if args.bidirectional else 'unidirectional'
    save_path = os.path.join(BASE_PATH, f"moving_mnist/{str(args.arch)}_{args.loss}_{str(args.loss_coefficient)}_{res_string}_{bidir_string}_{str(args.layers)}_{str(args.hidden_dim)}_predictor_results")
    os.makedirs(save_path, exist_ok=True)

    dataset_train = MovingMNIST(root=os.path.join(BASE_PATH, 'mnist_test_seq.npy'), is_train=True)
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)

    dataset_test = MovingMNIST(root=os.path.join(BASE_PATH, 'mnist_test_seq.npy'), is_train=False)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=True)

    delay_config = {
        'core_method': args.arch,
        'args': {
            'future_delay': 500,  # ms
            'future_delay_list': [100, 200, 300, 400, 500],
            'past_k': 5,
            'input_channels': 1,  # the output channel dimension of the encoder
            'height': 64 // (2 ** (args.layers + 1)),
            'width': 64 // (2 ** (args.layers + 1)),
            'hidden_dim': args.hidden_dim,
            'patch_size': 4,
            'num_layers': args.layers,
            'd_state': 64,
            'd_conv': 4,
            'expand': 2,
            'use_bidirectional': args.bidirectional,
            'dropout': 0.0,
            'residual': args.residual
        }
    }
    
    model = build_delay_module(delay_config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)

    # --------------- Loss, scheduler and optimizer setup ---------------
    loss_coefficient = args.loss_coefficient
    # define the loss
    if False:
        loss_config = {
            'args': {
                'cls_weight': 1.0,
                'reg': 2.0,
                'delay': 1.0,
                'delay_coeff': 10.0
            },
            'core_method': 'feature_map_prediction_loss' 
        }
        criterion = train_utils.create_loss(loss_config)
    else:
        #criterion = torch.nn.L1Loss()
        if args.loss == 'l1':
            criterion = torch.nn.L1Loss()
        elif args.loss == 'mse':
            criterion = torch.nn.MSELoss()
        elif args.loss.startswith('huber'):
            # extract delta from the loss name, e.g., 'huber_1.0' -> delta=1.0
            beta = float(args.loss.split('_')[1]) if '_' in args.loss else 1.0
            criterion = torch.nn.SmoothL1Loss(beta=beta)
        elif args.loss == 'bce':
            criterion = torch.nn.BCEWithLogitsLoss()
        elif args.loss == 'perceptual':
            from torchvision import models
            vgg = models.vgg16(pretrained=True).features[:16].eval().to(device)
            for param in vgg.parameters():
                param.requires_grad = False
            
            class PerceptualLoss(torch.nn.Module):
                def __init__(self, vgg):
                    super(PerceptualLoss, self).__init__()
                    self.vgg = vgg

                def forward(self, pred, gt):
                    pred = einops.rearrange(pred, 't b c h w -> (t b) c h w')
                    gt = einops.rearrange(gt, 't b c h w -> (t b) c h w')
                    pred_vgg = self.vgg(pred.repeat(1, 3, 1, 1))  # Convert to 3 channels
                    gt_vgg = self.vgg(gt.repeat(1, 3, 1, 1))      # Convert to 3 channels
                    return torch.nn.functional.mse_loss(pred_vgg, gt_vgg)

            criterion = PerceptualLoss(vgg)
        elif args.loss == 'ssim':
            class VideoSSIM(torch.nn.Module):
                def __init__(self, data_range=1.0, size_average=True, win_size=11, win_sigma=1.5, channel=1, spatial_dims=2):
                    super(VideoSSIM, self).__init__()
                    self.ssim = SSIM(data_range=data_range, size_average=size_average, win_size=win_size, win_sigma=win_sigma, channel=channel, spatial_dims=spatial_dims)

                def forward(self, pred, gt):
                    pred = einops.rearrange(pred, 'b t c h w -> (b t) c h w')
                    gt = einops.rearrange(gt, 'b t c h w -> (b t) c h w')
                    return 1 - self.ssim(pred, gt)

            criterion = VideoSSIM(data_range=1.0, size_average=True, win_size=11, win_sigma=1.5, channel=1, spatial_dims=2)
        elif args.loss == 'ms_ssim':
            class VideoMSSSIM(torch.nn.Module):
                def __init__(self, data_range=1.0, size_average=True, win_size=11, win_sigma=1.5, channel=1, spatial_dims=2):
                    super(VideoMSSSIM, self).__init__()
                    self.ms_ssim = MS_SSIM(data_range=data_range, size_average=size_average, win_size=win_size, win_sigma=win_sigma, channel=channel, spatial_dims=spatial_dims)

                def forward(self, pred, gt):
                    pred = einops.rearrange(pred, 'b t c h w -> (b t) c h w')
                    gt = einops.rearrange(gt, 'b t c h w -> (b t) c h w')
                    return 1 - self.ms_ssim(pred, gt)

            criterion = VideoMSSSIM(data_range=1.0, size_average=True, win_size=11, win_sigma=1.5, channel=1, spatial_dims=2)
        elif args.loss == 'cosine':
            class CosSim(torch.nn.Module):
                def __init__(self):
                    super(CosSim, self).__init__()
                    self.cosine_similarity = torch.nn.CosineSimilarity(dim=1)

                def forward(self, pred, gt):
                    pred = einops.rearrange(pred, 'b t c h w -> (b t) (c h w)')
                    gt = einops.rearrange(gt, 'b t c h w -> (b t) (c h w)')
                    return 1 - self.cosine_similarity(pred, gt).mean()
            
            criterion = CosSim()
        else:
            raise ValueError(f"Unsupported loss type: {args.loss}")

    # optimizer setup
    optimizer_config = {
        'optimizer': {
            'args': {
                'eps': 1.0e-10,
                'weight_decay': 0.0001
            },
            'core_method': 'Adam',
            'lr': 0.0001
        }
    }
    if False:
        optimizer = train_utils.setup_optimizer(optimizer_config, model_without_ddp)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.0001, 
            weight_decay=0.0001, 
            eps=1.0e-10
        )

    # lr scheduler setup
    num_steps = len(train_loader)
    lr_scheduler_config = {
        'lr_scheduler': {
            'core_method': 'multistep',
            'gamma': 0.01,
            'step_size': [20]
        }
    }

    scheduler = train_utils.setup_lr_schedular(lr_scheduler_config, optimizer, num_steps)
    loss_per_epoch = []
    validation_loss_per_epoch = []

    max_epochs = args.epochs
    for epoch in range(0, max_epochs):
        print(f"Epoch {epoch+1}/{max_epochs}")
        valid_train_loss = []

        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # inputs shape: [batch, K, 1, 64, 64]
            # targets shape: [batch, K, 1, 64, 64]
            model.zero_grad()
            optimizer.zero_grad()

            inputs = inputs.to(device)
            targets = targets.to(device)
            
            prediction, intermediate = model(inputs)  # prediction shape: [batch, K, 1, 64, 64]

            final_loss = criterion(intermediate, targets) * loss_coefficient
            valid_train_loss.append(final_loss.item())

            if (batch_idx + 1) % 25 == 0:
                show_pred_gt(intermediate, targets, inputs, f'epoch_{epoch}_iter_{batch_idx}', save_path=save_path)
            
            pbar2.update(1)
            final_loss.backward()
            optimizer.step()
        
        loss_per_epoch.append(statistics.mean(valid_train_loss))

        if lr_scheduler_config['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step()
        if lr_scheduler_config['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 1)

        valid_ave_loss = []
            # Create the dictionary for evaluation.
            # also store the confidence score for each prediction
            
        with torch.no_grad():
            for i, (inputs, targets) in tqdm.tqdm(enumerate(test_loader)):
                #print('test step %d' % i)
                model.eval()

                inputs = inputs.to(device)
                targets = targets.to(device)

                feature_pred, intermediate_preds = model(inputs)

                if (i + 1) % 500 == 0:
                    show_pred_gt(intermediate_preds, targets, inputs, f'test_epoch_{epoch}_iter_{i}', save_path=save_path)
            
                final_loss = criterion(intermediate_preds, targets) * loss_coefficient
                valid_ave_loss.append(final_loss.item())

        if args.save_ckpt:
            torch.save({
                'model_state_dict': model.state_dict()
            }, os.path.join(save_path, f'model_checkpoint_{epoch}.pth'))

        validation_loss_per_epoch.append(statistics.mean(valid_ave_loss))
        print('At epoch %d, the validation loss is %f' % (epoch, validation_loss_per_epoch[-1]))
            
    # Save model chkpoint

    # Save training and validation loss curves
    def plot_loss():
        plt.figure(figsize=(5, 10))
        plt.subplot(2, 1, 1)
        plt.plot(loss_per_epoch, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(validation_loss_per_epoch, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'loss_curves.png'))
        plt.show()

    plot_loss()
