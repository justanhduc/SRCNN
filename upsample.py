import numpy as np
import theano
from theano import tensor as T
import cv2
from matplotlib import pyplot as plt
import os
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from scipy import misc
import argparse

from SRCNN import SRCNN


def test_image(config_file, image):
    image = np.array(image, dtype='float32') / 255.
    x = T.tensor4('input', theano.config.floatX)
    net = SRCNN(config_file)
    net.load_params()
    hi_res_patches = net.inference(x)
    test_network = net.compile([x], hi_res_patches, name='test_srcnn', allow_input_downcast=True)
    prediction = test_network(image)
    return prediction


def make_patches(img, patch_size, patch_stride):
    patch_list = []
    shape = img.shape
    mask = np.zeros((shape[0], shape[1]))
    for i in range(0, shape[0], patch_stride):
        for j in range(0, shape[1], patch_stride):
            if i + patch_size > shape[0] or j + patch_size > shape[1]:
                continue
            patch_list.append(img[i:i+patch_size, j:j+patch_size, :])
            mask[i:i+patch_size, j:j+patch_size] += 1.
    return np.array(patch_list).transpose((0, 3, 1, 2)), mask


def process_image(file, patch_size=30, patch_stride=25):
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = prepare_data2(img, 340, 330)
    img_small = cv2.resize(cv2.resize(img, (img.shape[0] // 2, img.shape[1] // 2)), (img.shape[0], img.shape[1]))
    # img_small = color.rgb2ycbcr(img_small)
    patches, mask = make_patches(img_small, patch_size, patch_stride)
    return img, img_small, patches, mask


def assemble_patches(patches, mask, color=True, patch_size=30, patch_stride=25):
    image = np.zeros(mask.shape + (3,)) if color else np.zeros(mask.shape)
    idx = 0
    for i in range(0, mask.shape[0], patch_stride):
        for j in range(0, mask.shape[1], patch_stride):
            if i + patch_size > mask.shape[0] or j + patch_size > mask.shape[1]:
                continue
            if color:
                image[i:i+patch_size, j:j+patch_size, :] += patches[idx].transpose((1, 2, 0)) / \
                                                            np.expand_dims(mask[i:i+patch_size, j:j+patch_size], 2)
            else:
                image[i:i+patch_size, j:j+patch_size] += patches[idx, 0] / mask[i:i+patch_size, j:j+patch_size]
            idx += 1
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='low-resolution image destination')
    args = parser.parse_args()

    img, img_small, patches, mask = process_image(args.image)
    estimations = test_image('srcnn.config', patches)
    hi_img = assemble_patches(estimations, mask, True)
    img_scaled = img_small / 255.
    hi_img[hi_img < 0] = img_scaled[hi_img < 0]
    hi_img[hi_img > 1.] = img_scaled[hi_img > 1.]

    misc.imsave('%s' % os.path.basename(args.image), img_small)

    psnr(img/255., hi_img, data_range=1)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    plt.imshow(img_small)
    plt.subplot(1, 3, 3)
    plt.imshow(hi_img)
    print('SSIM: %.3f' % (ssim(img/255., hi_img, data_range=1, multichannel=True)))
    print('PSNR: %.3f' % (psnr(img/255., hi_img, data_range=1)))
