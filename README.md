# DeFeat-Net

This repository contains the network architectures, losses and pretrained models from [DeFeat-Net](https://www.researchgate.net/publication/340273870_DeFeat-Net_General_Monocular_Depth_via_Simultaneous_Unsupervised_Representation_Learning).

## Introduction
In the current monocular depth research, the dominant approach is to employ unsupervised training on large datasets, driven by warped photometric consistency. 
Such approaches lack robustness and are unable to generalize to challenging domains such as nighttime scenes or adverse weather conditions where assumptions about photometric consistency break down. 

We propose DeFeat-Net (Depth & Feature network), an approach to simultaneously learn a cross-domain dense feature representation, alongside a robust depth-estimation framework based on warped feature consistency. 
The resulting feature representation is learned in an unsupervised manner with no explicit ground-truth correspondences required.
 
We show that within a single domain, our technique is comparable to both the current state of the art in monocular depth estimation and supervised feature representation learning. 
However, by simultaneously learning features, depth and motion, our technique is able to generalize to challenging domains, allowing DeFeat-Net to outperform the current state-of-the-art with around 10% reduction in all error measures on more challenging sequences such as nighttime driving.

## Prerequisites
- Python >= 3.7 (networks based on dataclasses)
- PyTorch >= 0.4
- PIL 
- Matplotlib

## Usage
A simple [script](main.py) is provided as an example of running the depth and feature networks. 
```
# Create and load model
model = DeFeatNet.from_ckpt(ckpt_file, key=lambda x: x['model']).to(device)

# Sub-networks can be run separately
model.depth_net(images)
model.feat_net(images)
model.pose_net(images, support_images)
```

To visualize the depth maps produced by the network:
```
# Assuming a batched tensor (b, 1, h, w)
disp_np = disp.squeeze(1).cpu().numpy()
ax.imshow(disp_np[0], cmap='magma', vmax=np.percentile(disp_np[0], 95))
```

NOTE: Images should be downsampled to `(480, 352)` to match training resolution.

Checkpoints can be found [here](http://personal.ee.surrey.ac.uk/Personal/S.Hadfield/ckpts_DeFeat.zip), 
which contains variants trained on Kitti and RobotCar-Seasons, and should be placed in the `ckpt` directory.
When restoring from a checkpoint with `DeFeatNet.from_ckpt`, all optional parameters are set to the correct configuration, 
so no additional initialization is required.

## Citation
Please cite the following paper if you find DeFeat-Net useful in your research:
```
@inproceedings{spencer2020,
  title={DeFeat-Net: General Monocular Depth via Simultaneous Unsupervised Representation Learning},
  author={Spencer, Jaime  and Bowden, Richard and Hadfield, Simon},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

We would also like to thank the authors of [Monodepth2](https://github.com/nianticlabs/monodepth2) for their contribution.


## Contact

You can contact me at [jaime.spencer@surrey.ac.uk](mailto:jaime.spencer@surrey.ac.uk)

