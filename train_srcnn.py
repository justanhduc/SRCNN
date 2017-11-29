import os
from random import shuffle
import theano
from theano import tensor as T
import numpy as np
from scipy import misc
import pickle as pkl

from neuralnet.utils import rgb2gray
from SRCNN import SRCNN, DataManager4
from neuralnet.metrics import psnr, MS_SSIM
import time
from matplotlib import pyplot as plt


def train_srcnn(config_file, **kwargs):
    net = SRCNN(config_file)

    X = T.tensor4('input', theano.config.floatX)
    Y = T.tensor4('gt', theano.config.floatX)

    mean_X = T.mean(X)
    X_centered = X
    placeholder_x = theano.shared(np.zeros((net.batch_size,) + net.input_shape[1:], 'float32'), 'patch_placeholder')
    placeholder_y = theano.shared(np.zeros((net.batch_size, net.output_shape[2], net.output_shape[0], net.output_shape[1]),
                                           'float32'), 'img_placeholder')

    output = net.inference(X_centered)
    up_img = output
    # cropped = 9 // 2 + 5 // 2
    cost = net.build_cost(up_img, Y, **{'params': net.regularizable})
    # cost = net.build_cost(up_img, Y[:, :, cropped:-cropped, cropped:-cropped], **{'params': net.regularizable})
    updates = net.build_updates(cost, net.trainable)
    train_network = net.compile([], cost, updates=updates, givens={X: placeholder_x, Y: placeholder_y}, name='train_srcnn')

    psnr_loss = psnr(up_img, Y)
    # psnr_loss = psnr(up_img, Y[:, :, cropped:-cropped, cropped:-cropped])
    msssim_loss = MS_SSIM(rgb2gray(up_img), rgb2gray(Y))
    # msssim_loss = MS_SSIM(up_img, Y[:, :, cropped:-cropped, cropped:-cropped])
    test_network = net.compile([], [cost, psnr_loss, msssim_loss], givens={X: placeholder_x, Y: placeholder_y}, name='test_srcnn')

    epoch = 0
    vote_to_terminate = 0
    best_psnr = 0.
    best_epoch = 0
    if net.display_cost:
        training_cost_to_plot = []
        validation_cost_to_plot = []

    data_manager = DataManager4(net.batch_size, (placeholder_x, placeholder_y), True, False)
    num_training_batches = data_manager.train_data_shape[0] // net.batch_size
    num_validation_batches = data_manager.test_data_shape[0] // net.validation_batch_size

    print('Training... %d training batches, %d developing batches' % (num_training_batches, num_validation_batches))
    start_training_time = time.time()
    while epoch < net.n_epochs:
        epoch += 1
        training_cost = 0.
        start_epoch_time = time.time()
        batches = data_manager.get_batches(epoch=epoch, num_epochs=net.n_epochs)
        idx = 0
        for b in batches:
            iteration = (epoch - 1.) * num_training_batches + idx + 1
            data_manager.update_input(b)
            training_cost += train_network()
            if np.isnan(training_cost):
                raise ValueError('Training failed due to NaN cost')

            if iteration % net.validation_frequency == 0:
                batch_valid = data_manager.get_batches(stage='test')
                validation_cost = 0.
                validation_psnr = 0.
                validation_msssim = 0.
                for b_valid in batch_valid:
                    data_manager.update_input(b_valid)
                    c, p, s = test_network()
                    validation_cost += c
                    validation_psnr += p
                    validation_msssim += s
                validation_cost /= num_validation_batches
                validation_psnr /= num_validation_batches
                validation_msssim /= num_validation_batches
                print('\tvalidation cost: %.4f' % validation_cost)
                print('\tvalidation PSNR: %.4f' % validation_psnr)
                print('\tvalidation MSSSIM: %.4f' % validation_msssim)
                if validation_psnr > best_psnr:
                    best_epoch = epoch
                    best_psnr = validation_psnr
                    vote_to_terminate = 0
                    print('\tbest validation PSNR: %.4f' % best_psnr)
                    if net.extract_params:
                        net.save_params()
                else:
                    vote_to_terminate += 1

                if net.display_cost:
                    training_cost_to_plot.append(training_cost / (idx + 1))
                    validation_cost_to_plot.append(validation_cost)
                    plt.clf()
                    plt.plot(training_cost_to_plot)
                    plt.plot(validation_cost_to_plot)
                    plt.show(block=False)
                    plt.pause(1e-5)
            idx += 1
        training_cost /= num_training_batches
        print('\tepoch %d took %.2f mins' % (epoch, (time.time() - start_epoch_time) / 60.))
        print('\ttraining cost: %.4f' % training_cost)
    if net.display_cost:
        plt.savefig('%s/training_curve.png' % net.save_path)
    print('Best validation PSNR: %.4f' % best_psnr)
    print('Training took %.2f hours' % ((time.time() - start_training_time) / 3600))


def create_mask(shape_full, patch_shape):
    mask = np.zeros(np.roll(shape_full, 1), dtype='float32')
    patch = np.ones((4, 1, patch_shape[0], patch_shape[1]), dtype='float32')
    mask[:, :patch_shape[0], :patch_shape[1]] += patch[0]
    mask[:, :patch_shape[0], shape_full[1] - patch_shape[1]:] += patch[1]
    mask[:, shape_full[0] - patch_shape[0]:, :patch_shape[1]] += patch[2]
    mask[:, shape_full[0] - patch_shape[0]:, shape_full[1] - patch_shape[1]:] += patch[3]
    return 1 / mask


if __name__ == '__main__':
    train_srcnn('srcnn.config')
