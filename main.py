from pathlib import Path
import sys

from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np

root_path = Path(__file__).resolve().parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from networks.defeat_net import DeFeatNet
from utils import ops


if __name__ == '__main__':
    device = ops.get_device()
    ckpt_file = root_path / 'ckpts' / 'ckpt_seasons.pt'

    model = DeFeatNet.from_ckpt(ckpt_file, key=lambda x: x['model']).to(device)
    model = model.eval()

    imfiles = ['image1.png', 'image2.png']

    def load_image(file):
        image = Image.open(root_path / 'images' / file).convert('RGB')
        image = image.resize([480, 352])
        image = ops.img2torch(np.array(image), batched=True).to(device)
        return image

    images = torch.cat([load_image(file) for file in imfiles])

    with torch.no_grad():
        disp = model.depth_net(images)[('disp', 0)]
        dense_features = model.feat_net(images)

    disp_np = disp.squeeze(1).cpu().numpy()

    _, (axs, axs2) = plt.subplots(2, len(imfiles))
    plt.tight_layout()

    for ax, img in zip(axs, ops.torch2np(images)):
        ax.set_xticks([]), ax.set_yticks([])
        ax.imshow(img)

    for ax, d in zip(axs2, disp_np):
        ax.set_xticks([]), ax.set_yticks([])
        ax.imshow(d, cmap='magma', vmax=np.percentile(d, 95))

    plt.show()
