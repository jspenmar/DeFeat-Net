import numpy as np
import torch
import torch.nn.functional as F


def get_device(device=None):
    if isinstance(device, torch.device):
        return device
    return torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def downsample(tensor, factor, bilinear=True):
    """Downsample tensor by a specified scaling factor."""
    if bilinear:
        return F.interpolate(tensor, scale_factor=1/factor, mode='bilinear', align_corners=True)
    else:
        return F.interpolate(tensor, scale_factor=1/factor, mode='nearest')


def upsample(tensor, factor=None, size=None, bilinear=True):
    if bilinear:
        return F.interpolate(tensor, scale_factor=factor, size=size, mode='bilinear', align_corners=True)
    else:
        return F.interpolate(tensor, scale_factor=factor, size=size, mode='nearest')


def upsample_like(tensor, ref_tensor, bilinear=True):
    if bilinear:
        return F.interpolate(tensor, size=ref_tensor.shape[-2:], mode='bilinear', align_corners=True)
    else:
        return F.interpolate(tensor, size=ref_tensor.shape[-2:], mode='nearest')


def channel_mean(tensor, keepdim=False):
    if tensor.ndim != 4:
        raise RuntimeError('Tensor must be 4D')
    return tensor.mean(2, keepdim=keepdim).mean(2+keepdim, keepdim=keepdim)


def make_one_hot(tensor, n=None):
    """
    Convert a label map into a one hot vector map.

    :param tensor: Tensor indicating class of each item (b, 1, h, w)
    :param n: Number of classes (default uses max value in tensor)
    :return: Tensor with one hot vectors, i.e. a 1 at the class index (b, n, h, w)
    """
    n, (b, _, h, w) = n or tensor.max(), tensor.shape
    onehot = torch.zeros((b, n, h, w), device=tensor.device, dtype=torch.long)
    onehot.scatter_(1, tensor, 1)
    return onehot


def extract_kpt_vectors(tensor, kpts, rand_batch=False):
    """
    Pick channel vectors from 2D location in tensor.
    E.g. tensor[b, :, y1, x1]

    :param tensor: Tensor to extract from [b, c, h, w]
    :param kpts: Tensor with 'n' keypoints (x, y) as [b, n, 2]
    :param rand_batch: Randomize tensor in batch the vector is extracted from
    :return: Tensor entries as [b, n, c]
    """
    batch_size, num_kpts = kpts.shape[:-1]  # [b, n]

    # Reshape as a single batch -> [b*n, 2]
    tmp_idx = kpts.contiguous().view(-1, 2).long()

    # Flatten batch number indexes  -> [b*n] e.g. [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    b_num = torch.arange(batch_size)
    b_num = b_num.repeat((num_kpts, 1)).view(-1)
    b_num = torch.sort(b_num)[0] if not rand_batch else b_num[torch.randperm(len(b_num))]

    # Perform indexing and reshape to [b, n, c]
    return tensor[b_num, :, tmp_idx[:, 1], tmp_idx[:, 0]].reshape([batch_size, num_kpts, -1])


def torch2np(tensor):
    """
    Convert from torch tensor to numpy convention.
    If 4D -> [b, c, h, w] to [b, h, w, c]
    If 3D -> [c, h, w] to [h, w, c]

    :param tensor: Torch tensor
    :return: Numpy array
    """
    array, d = tensor.detach().cpu().numpy(), tensor.dim()
    perm = [0, 2, 3, 1] if d == 4 else [1, 2, 0] if d == 3 else None
    return array.transpose(perm) if perm else array


def img2torch(img, batched=False):
    """
    Convert single image to torch tensor convention.
    Image is normalized and converted to 4D: [1, 3, h, w]

    :param img: Numpy image
    :param batched: Return as 4D or 3D (default)
    :return: Torch tensor
    """
    img = torch.from_numpy(img.astype(np.float32)).permute([2, 0, 1])
    if img.max() > 1:
        img /= img.max()
    return img[None] if batched else img
