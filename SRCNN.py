import numpy as np
import theano
from theano import tensor as T
from skimage import color

from neuralnet import Model
from neuralnet import layers
from neuralnet.utils import DataManager


class SRCNN(Model):
    def __init__(self, config_file, **kwargs):
        super(SRCNN, self).__init__(config_file, **kwargs)

        self.input_shape = (None, self.config['model']['input_shape'][2], self.config['model']['input_shape'][0],
                            self.config['model']['input_shape'][1])
        self.output_shape = self.config['model']['output_shape']
        self.augmentation = self.config['model']['augmentation']
        # self.num_patches = 4

        self.model.append(layers.ConvolutionalLayer(self.input_shape,
                                                    (64, self.input_shape[1], 9, 9), 'normal', layer_name='conv1',
                                                    border_mode='half',))
        self.model.append(layers.ConvolutionalLayer(self.model[-1].output_shape,
                                                    (32, 64, 1, 1), 'normal', layer_name='conv2'))
        self.model.append(layers.ConvolutionalLayer(self.model[-1].output_shape,
                                                    (self.output_shape[2], 32, 5, 5), 'normal', activation='linear',
                                                    layer_name='out_conv', border_mode='half',))

        super(SRCNN, self).get_all_params()
        super(SRCNN, self).get_trainable()
        super(SRCNN, self).get_regularizable()

    def inference2(self, input):
        output = super(SRCNN, self).inference(input)
        output_full = T.zeros(tuple(np.roll(self.output_shape, 1)), dtype='float32')
        output_full = T.inc_subtensor(output_full[:, :self.input_shape[2], :self.input_shape[3]], output[0])
        output_full = T.inc_subtensor(output_full[:, :self.input_shape[2], self.output_shape[1] - self.input_shape[3]:], output[1])
        output_full = T.inc_subtensor(output_full[:, self.output_shape[0] - self.input_shape[2]:, :self.input_shape[3]], output[2])
        output_full = T.inc_subtensor(output_full[:, self.output_shape[0] - self.input_shape[2]:, self.output_shape[1] - self.input_shape[3]:], output[3])
        return output_full


class DataManager4(DataManager):
    def __init__(self, batch_size, placeholders, shuffle=False, no_target=False, augmentation=False,
                 num_cached=10):
        super(DataManager4, self).__init__(batch_size, placeholders, shuffle=shuffle, no_target=no_target,
                                           augmentation=augmentation, num_cached=num_cached)

    def load_data(self):
        import pickle as pkl
        self.training_set = pkl.load(open('patch30_train4000_x2.pkl', 'rb'))
        self.testing_set = pkl.load(open('patch30_test4000_x2.pkl', 'rb'))
        self.train_data_shape = self.training_set[0].shape
        self.test_data_shape = self.testing_set[0].shape

    def augment_minibatches(self, minibatches, *args):
        pass

    def rgb2ycbcr(self, image):
        arr = image.reshape((-1, 3))
        conversion_mat = np.array([[65.738/256, 129.057/256, 25.064/256],
                                   [37.945/256, -74.494/256, 112.439/256],
                                   [112.439/256, - 94.154/256, -18.285/256]])
        bias_mat = np.array([16, 128, 128])
        arr_ycbcr = np.dot(conversion_mat, arr.transpose()).transpose() + bias_mat
        return arr_ycbcr.reshape((image.shape[0], -1, 3))

    def prepare_image(self, batches):
        for batch in batches:
            lo_res = []
            hi_res = []
            for lo_res_img, hi_res_img in zip(batch[0], batch[1]):
                lo_res_ycbcr = color.rgb2ycbcr(lo_res_img.astype('uint8'))
                hi_res_ycbcr = color.rgb2ycbcr(hi_res_img.astype('uint8'))
                lo_res.append(lo_res_ycbcr[..., 0])
                hi_res.append(hi_res_ycbcr[..., 0])
            lo_res = np.expand_dims(np.asarray(lo_res), 1).astype('float32') / 255.
            hi_res = np.expand_dims(np.asarray(hi_res), 1).astype('float32') / 255.
            yield lo_res, hi_res

    def generator(self, stage='train'):
        dataset = self.training_set if stage == 'train' else self.testing_set
        shape = self.train_data_shape if stage == 'train' else self.test_data_shape
        shuffle = self.shuffle if stage == 'train' else False
        num_batches = shape[0] // self.batch_size
        if not self.no_target:
            x, y = dataset
            y = np.asarray(y).transpose((0, 3, 1, 2)) / 255.
        else:
            x = dataset
        x = np.asarray(x, dtype=theano.config.floatX).transpose((0, 3, 1, 2)) / 255.
        if shuffle:
            index = np.arange(0, np.asarray(x).shape[0])
            np.random.shuffle(index)
            x = x[index]
            if not self.no_target:
                y = y[index]
        for i in range(num_batches):
            yield (x[i * self.batch_size:(i + 1) * self.batch_size], y[i * self.batch_size:(i + 1) * self.batch_size]) \
                if not self.no_target else x[i * self.batch_size:(i + 1) * self.batch_size]

    def get_batches(self, stage='train', epoch=None, num_epochs=None, *args):
        batches = self.generator(stage)
        if self.augmentation:
            batches = self.augment_minibatches(batches, *args)
        # batches = self.prepare_image(batches)
        batches = self.generate_in_background(batches)
        if epoch is not None and num_epochs is not None:
            shape = self.train_data_shape if stage == 'train' else self.test_data_shape
            num_batches = shape[0] // self.batch_size
            batches = self.progress(batches, desc='Epoch %d/%d, Batch ' % (epoch, num_epochs), total=num_batches)
        return batches
