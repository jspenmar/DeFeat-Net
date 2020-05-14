import torch


def disp2depth(disp, min_depth, max_depth):
    """Convert sigmoid disparity into depth prediction"""
    min_disp, max_disp = 1/max_depth, 1/min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1/scaled_disp
    return scaled_disp, depth


def params2tform(axisangle, translation, invert=False):
    """Convert (axisangle, translation) into a 4x4 matrix"""
    r, t = axisangle2rot(axisangle), translation.clone()
    if invert:
        r, t = r.transpose(1, 2), t*-1

    t = get_translation_matrix(t)
    return torch.matmul(r, t) if invert else torch.matmul(t, r)


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix"""
    t_mat = torch.eye(4).repeat((translation_vector.shape[0], 1, 1)).to(device=translation_vector.device)
    t = translation_vector.contiguous().view(-1, 3, 1)
    t_mat[:, :3, 3, None] = t
    return t_mat


def axisangle2rot(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, p=2, dim=2, keepdim=True)
    axis = vec / (angle + 1e-7)

    cos, sin = torch.cos(angle), torch.sin(angle)
    c = 1 - cos
    coords = x, y, z = axis.split(1, dim=-1)

    xs, ys, zs = [coord * sin for coord in coords]
    x_c, y_c, z_c = [coord * c for coord in coords]
    xy_c, yz_c, zx_c = x * y_c, y * z_c, z * x_c

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * x_c + cos)
    rot[:, 0, 1] = torch.squeeze(xy_c - zs)
    rot[:, 0, 2] = torch.squeeze(zx_c + ys)
    rot[:, 1, 0] = torch.squeeze(xy_c + zs)
    rot[:, 1, 1] = torch.squeeze(y * y_c + cos)
    rot[:, 1, 2] = torch.squeeze(yz_c - xs)
    rot[:, 2, 0] = torch.squeeze(zx_c - ys)
    rot[:, 2, 1] = torch.squeeze(yz_c + xs)
    rot[:, 2, 2] = torch.squeeze(z * z_c + cos)
    rot[:, 3, 3] = 1

    return rot
