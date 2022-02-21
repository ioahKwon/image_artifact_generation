## import packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.io import loadmat

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, rescale

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


## 2-3. Denoising: poisson noise (CT-domain)
# SYSTEM SETTING
N = 512
ANG = 180
VIEW = 360
THETA = np.linspace(0, ANG, VIEW, endpoint=False)

A = lambda x: radon(x, THETA, circle=False).astype(np.float32)
AT = lambda y: iradon(y, THETA, circle=False, filter=None, output_size=N).astype(np.float32)
AINV = lambda y: iradon(y, THETA, circle=False, output_size=N).astype(np.float32)

# Low dose CT: adding poisson noise
pht = shepp_logan_phantom()
pht = 0.03 * rescale(pht, scale=512/400, order=0)

prj = A(pht)

i0 = 1e4
dst = i0 * np.exp(-prj)
dst = poisson.rvs(dst)
dst = -np.log(dst / i0)
dst[dst < 0] = 0

noise = dst - prj

rec = AINV(prj)
rec_noise = AINV(noise)
rec_dst = AINV(dst)

plt.subplot(241)
plt.imshow(pht, cmap='gray', vmin=0, vmax=0.03)
plt.title("Ground Truth")

plt.subplot(242)
plt.imshow(rec, cmap='gray', vmin=0, vmax=0.03)
plt.title("Reconstruction")

plt.subplot(243)
plt.imshow(rec_noise, cmap='gray')
plt.title("Reconstruction using Noise")

plt.subplot(244)
plt.imshow(rec_dst, cmap='gray', vmin=0, vmax=0.03)
plt.title("Reconstruction using Noisy data")

plt.subplot(246)
plt.imshow(prj, cmap='gray')
plt.title("Projection data")

plt.subplot(247)
plt.imshow(noise, cmap='gray')
plt.title("Poisson Noise in projection")

plt.subplot(248)
plt.imshow(dst, cmap='gray')
plt.title("Noisy data")

## 3. Super-resolution
"""
------------------------
order options
------------------------
0: Nearest-neighbor
1: Bi-linear (default)
2: Bi-quadratic
3: Bi-cubic
4: Bi-quartic
5: Bi-quintic
"""

dw = 1/5.0 # downsampling ratio -> 5배
order = 1

dst_dw = rescale(img, scale=(dw, dw, 1), order=order)
dst_up = rescale(dst_dw, scale=(1/dw, 1/dw, 1), order=order)

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(dst_dw), cmap=cmap, vmin=0, vmax=1)
plt.title("Downscaled image")

plt.subplot(133)
plt.imshow(np.squeeze(dst_up), cmap=cmap, vmin=0, vmax=1)
plt.title("Upscaled image")

##

