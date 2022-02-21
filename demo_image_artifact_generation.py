## import packages
import numpy as np

import matplotlib.pyplot as plt

## Load image
img = plt.imread("Lenna.png")

# gray image generation
#img = np.mean(img, axis=2, keepdims=True) # channel 방향으로 평균값 -> gray-scale image

sz = img.shape

cmap = 'gray' if sz[2] == 1 else None

plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")
plt.show()

## 1-1. Inpainting: Uniform sampling
ds_y = 2
ds_x = 4

msk = np.zeros(sz)
msk[::ds_y, ::ds_x, :] = 1

dst = img*msk

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(msk), cmap=cmap, vmin=0, vmax=1)
plt.title("Uniform sampling mask")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=camp, vmin=0, vmax=1)
plt.title('Sampling image')


