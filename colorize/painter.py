import chainer
import cv2
import numpy as np
from chainer import serializers, Variable

from colorize import unet
from colorize.img2imgDataset import ImageAndRefDataset


class Painter:
    def __init__(self, gpu=-1):
        self.line_image_dir = 'images/line/'
        self.ref_image_dir = 'images/ref/'
        self.color_image_dir = 'images/color/'
        self.gpu = gpu
        if self.gpu >= 0:
            chainer.cuda.get_device(self.gpu).use()
            chainer.cuda.set_max_workspace_size(64 * 1024 * 1024)  # 64MB
        self.cnn_128 = self.__load_cnn('colorize/unet_128_standard')
        self.cnn_512 = self.__load_cnn('colorize/unet_512_standard')

    def __load_cnn(self, model):
        cnn = unet.UNET()
        if self.gpu >= 0:
            cnn.to_gpu()
        serializers.load_npz(model, cnn)
        return cnn

    def colorize(self, unique_name):
        dataset = ImageAndRefDataset([f'{unique_name}.png'], self.line_image_dir, self.ref_image_dir)
        # TODO Extract parameter for optimization
        sample = dataset.get_example(0, minimize=True, blur=0, s_size=128)
        sample_container = np.zeros((1, 4, sample[0].shape[1], sample[0].shape[2]), dtype='f')
        sample_container[0, :] = sample[0]

        sample_container = sample_container if self.gpu < 0 else chainer.cuda.to_gpu(sample_container)
        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                image_conv2d_layer = self.cnn_128.calc(Variable(sample_container))
        del sample_container

        input_bat = np.zeros((1, 4, sample[1].shape[1], sample[1].shape[2]), dtype='f')
        input_bat[0, 0, :] = sample[1]

        output = chainer.cuda.to_cpu(image_conv2d_layer.data[0])
        del image_conv2d_layer

        for channel in range(3):
            input_bat[0, 1 + channel, :] = cv2.resize(
                output[channel, :],
                (sample[1].shape[2], sample[1].shape[1]),
                interpolation=cv2.INTER_CUBIC
            )

        link = input_bat if self.gpu < 0 else chainer.cuda.to_gpu(input_bat, None)
        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                image_conv2d_layer = self.cnn_512.calc(Variable(link))
        del link

        color_path = self.color_image_dir + f'{unique_name}.jpg'
        save_as_img(image_conv2d_layer.data[0], color_path)
        del image_conv2d_layer

        return color_path


def save_as_img(array, path):
    array = array.transpose(1, 2, 0)
    array = array.clip(0, 255).astype(np.uint8)
    array = chainer.cuda.to_cpu(array)
    (major, minor, _) = cv2.__version__.split(".")
    code = cv2.COLOR_YUV2RGB if major == '3' else cv2.COLOR_YUV2BGR
    cv2.imwrite(path, cv2.cvtColor(array, code))


if __name__ == '__main__':
    Painter().colorize('test')
