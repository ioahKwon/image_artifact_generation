## import packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.io import loadmat

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
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title('Sampling image')

## 1-2. Inpainting: Random sampling
# rnd = np.random.rand(sz[0], sz[1], sz[2])
# prob = 0.5
# msk = (rnd > prob).astype(np.float)
# dst = img*msk
#### if, RGB color를 동일한 sampling mask 가지고 싶다면
rnd = np.random.rand(sz[0], sz[1], 1) # single channel로 된 mask를
prob = 0.5
msk = (rnd > prob).astype(np.float)
msk = np.tile(msk, (1, 1, sz[2])) # channel 방향으로 3번 복사
dst = img*msk

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")

plt.subplot(132) # x,y 뿐 아니라 channel 방향으로도 randomly selected -> 색깔 다양한 것.
plt.imshow(np.squeeze(msk), cmap=cmap, vmin=0, vmax=1)
plt.title("Uniform sampling mask")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title('Sampling image')

## 1-3. Inpainting : Gaussian sampling
ly = np.linspace(-1, 1, sz[0])
lx = np.linspace(-1, 1, sz[1])

x, y = np.meshgrid(lx, ly)

x0 = 0 # center position of x
y0 = 0 # center position of y
sgmx = 1 # sigma x
sgmy = 1 # sigma y
a = 1 # amplitude

### command + /
# gauss = a * np.exp(-((x - x0)**2/(2*sgmx**2) + (y - y0)**2/(2*sgmy**2)))
# #plt.imshow(gauss)
# gauss = np.tile(gauss[:, :, np.newaxis], (1, 1, sz[2])) # tile은 복붙의 의미
# rnd = np.random.rand(sz[0], sz[1], sz[2])
# msk = (rnd < gauss).astype(np.float)
# dst = img*msk

# channel 방향으로 동일한 sampling
gauss = a * np.exp(-((x - x0)**2/(2*sgmx**2) + (y - y0)**2/(2*sgmy**2)))
#plt.imshow(gauss)
gauss = np.tile(gauss[:, :, np.newaxis], (1, 1, 1)) # tile은 복붙의 의미
rnd = np.random.rand(sz[0], sz[1], 1)
msk = (rnd < gauss).astype(np.float)
msk = np.tile(msk, (1, 1, sz[2]))

dst = img*msk

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(msk), cmap=cmap, vmin=0, vmax=1)
plt.title("Gaussian sampling mask")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title("Sampling image")

## 2-1. Denoising: Random noise
# sigma 값에 따른 노이즈 보려면 구글에 BM3D 쳐서 보기.
sgm = 90.0

noise = sgm/255.0 * np.random.randn(sz[0], sz[1], sz[2]) # img가 normalized 되어 있기 때문에 꼭 sgm를 normalize 하기!

dst = img + noise

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title('Ground Truth')

plt.subplot(132)
plt.imshow(np.squeeze(noise), cmap=cmap, vmin=0, vmax=1)
plt.title('Noise')

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title('Noisy Image')

## 2-2. Denoising: Poisson noise (image-domain) ... 광학에서 많이 쓰임.
dst = poisson.rvs(255.0 * img) / 255.0 # 포아송은 int scale로 값이 추가됨 -> 0~255로 discretization되어 있는 이미지를 가지고
                                       #                               noise 추가한 뒤, 다시 Normalization 해줌.
noise = dst - img

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title('Ground Truth')

plt.subplot(132)
plt.imshow(np.squeeze(noise), cmap=cmap, vmin=0, vmax=1)
plt.title('Poisson Noise')

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title('Noisy Image')


##

