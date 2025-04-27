from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#Import Model
from tensorflow import keras 

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Concatenate, UpSampling2D


import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import Activation
from keras.layers import Add

from keras.layers import DepthwiseConv2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import Layer
from keras.layers import InputSpec
from keras.utils import get_source_inputs
from keras import backend as K
from keras.applications import imagenet_utils

import keras.utils as conv_utils
from tensorflow.keras.utils import get_file

class Unet2D:

    def __init__(self, n_filters, input_dim_x, input_dim_y, num_channels):
        self.input_dim_x = input_dim_x
        self.input_dim_y = input_dim_y
        self.n_filters = n_filters
        self.num_channels = num_channels

    def get_unet_model_5_levels(self):
        unet_input = Input(shape=(self.input_dim_x, self.input_dim_y, self.num_channels))

        conv1 = Conv2D(self.n_filters, kernel_size=3, activation='relu', padding='same')(unet_input)
        conv1 = Conv2D(self.n_filters, kernel_size=3, activation='relu', padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(self.n_filters*2, kernel_size=3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(self.n_filters*2, kernel_size=3, activation='relu', padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(self.n_filters*4, kernel_size=3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(self.n_filters*4, kernel_size=3, activation='relu', padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(self.n_filters*8, kernel_size=3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(self.n_filters*8, kernel_size=3, activation='relu', padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(self.n_filters*16, kernel_size=3, activation='relu', padding='same')(pool4)
        conv5 = Conv2D(self.n_filters*16, kernel_size=3, activation='relu', padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(self.n_filters*16, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop5))
        concat6 = Concatenate()([drop4, up6])
        conv6 = Conv2D(self.n_filters*8, kernel_size=3, activation='relu', padding='same')(concat6)
        conv6 = Conv2D(self.n_filters*8, kernel_size=3, activation='relu', padding='same')(conv6)
        conv6 = BatchNormalization()(conv6)

        up7 = Conv2D(self.n_filters*8, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
        concat7 = Concatenate()([conv3, up7])
        conv7 = Conv2D(self.n_filters*4, kernel_size=3, activation='relu', padding='same')(concat7)
        conv7 = Conv2D(self.n_filters*4, kernel_size=3, activation='relu', padding='same')(conv7)
        conv7 = BatchNormalization()(conv7)

        up8 = Conv2D(self.n_filters*4, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
        concat8 = Concatenate()([conv2, up8])
        conv8 = Conv2D(self.n_filters*2, kernel_size=3, activation='relu', padding='same')(concat8)
        conv8 = Conv2D(self.n_filters*2, kernel_size=3, activation='relu', padding='same')(conv8)
        conv8 = BatchNormalization()(conv8)

        up9 = Conv2D(self.n_filters*2, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
        concat9 = Concatenate()([conv1, up9])
        conv9 = Conv2D(self.n_filters, kernel_size=3, activation='relu', padding='same')(concat9)
        conv9 = Conv2D(self.n_filters, kernel_size=3, activation='relu', padding='same')(conv9)
        conv9 = BatchNormalization()(conv9)

        conv10 = Conv2D(3, kernel_size=1, activation='sigmoid', padding='same')(conv9)

        return Model(outputs=conv10,  inputs=unet_input), 'unet_model_5_levels'


    def get_unet_model_4_levels(self):
        unet_input = Input(shape=(self.input_dim_x, self.input_dim_y, self.num_channels))

        conv1 = Conv2D(self.n_filters*2, kernel_size=3, activation='relu', padding='same')(unet_input)
        conv1 = Conv2D(self.n_filters*2, kernel_size=3, activation='relu', padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(self.n_filters*4, kernel_size=3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(self.n_filters*4, kernel_size=3, activation='relu', padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(self.n_filters*8, kernel_size=3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(self.n_filters*8, kernel_size=3, activation='relu', padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        drop3 = Dropout(0.5)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

        conv4 = Conv2D(self.n_filters*16, kernel_size=3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(self.n_filters*16, kernel_size=3, activation='relu', padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        drop4 = Dropout(0.5)(conv4)

        up5 = Conv2D(self.n_filters*16, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop4))
        concat5 = Concatenate()([drop3, up5])
        conv5 = Conv2D(self.n_filters*8, kernel_size=3, activation='relu', padding='same')(concat5)
        conv5 = Conv2D(self.n_filters*8, kernel_size=3, activation='relu', padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)

        up6 = Conv2D(self.n_filters*8, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5))
        concat6 = Concatenate()([conv2, up6])
        conv6 = Conv2D(self.n_filters*4, kernel_size=3, activation='relu', padding='same')(concat6)
        conv6 = Conv2D(self.n_filters*4, kernel_size=3, activation='relu', padding='same')(conv6)
        conv6 = BatchNormalization()(conv6)

        up7 = Conv2D(self.n_filters*4, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
        concat7 = Concatenate()([conv1, up7])
        conv7 = Conv2D(self.n_filters*2, kernel_size=3, activation='relu', padding='same')(concat7)
        conv7 = Conv2D(self.n_filters*2, kernel_size=3, activation='relu', padding='same')(conv7)
        conv7 = BatchNormalization()(conv7)

        conv9 = Conv2D(3, kernel_size=1, activation='sigmoid', padding='same')(conv7)

        return Model(outputs=conv9,  inputs=unet_input), 'unet_model_4_levels'


    def get_unet_model_yuanqing(self):
        # Model inspired by https://github.com/yuanqing811/ISIC2018
        unet_input = Input(shape=(self.input_dim_x, self.input_dim_y, self.num_channels))

        conv1 = Conv2D(self.n_filters, kernel_size=3, activation='relu', padding='same')(unet_input)
        conv1 = Conv2D(self.n_filters, kernel_size=3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(self.n_filters * 2, kernel_size=3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(self.n_filters * 2, kernel_size=3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(self.n_filters * 4, kernel_size=3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(self.n_filters * 4, kernel_size=3, activation='relu', padding='same')(conv3)
        conv3 = Conv2D(self.n_filters * 4, kernel_size=3, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(self.n_filters * 8, kernel_size=3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(self.n_filters * 8, kernel_size=3, activation='relu', padding='same')(conv4)
        conv4 = Conv2D(self.n_filters * 8, kernel_size=3, activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(self.n_filters * 8, kernel_size=3, activation='relu', padding='same')(pool4)
        conv5 = Conv2D(self.n_filters * 8, kernel_size=3, activation='relu', padding='same')(conv5)
        conv5 = Conv2D(self.n_filters * 8, kernel_size=3, activation='relu', padding='same')(conv5)

        up6 = Conv2D(self.n_filters * 4, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5))
        feature4 = Conv2D(self.n_filters * 4, kernel_size=3, activation='relu', padding='same')(conv4)
        concat6 = Concatenate()([feature4, up6])
        conv6 = Conv2D(self.n_filters * 4, kernel_size=3, activation='relu', padding='same')(concat6)
        conv6 = Conv2D(self.n_filters * 4, kernel_size=3, activation='relu', padding='same')(conv6)

        up7 = Conv2D(self.n_filters * 2, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
        feature3 = Conv2D(self.n_filters * 2, kernel_size=3, activation='relu', padding='same')(conv3)
        concat7 = Concatenate()([feature3, up7])
        conv7 = Conv2D(self.n_filters * 2, kernel_size=3, activation='relu', padding='same')(concat7)
        conv7 = Conv2D(self.n_filters * 2, kernel_size=3, activation='relu', padding='same')(conv7)

        up8 = Conv2D(self.n_filters * 1, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
        feature2 = Conv2D(self.n_filters * 1, kernel_size=3, activation='relu', padding='same')(conv2)
        concat8 = Concatenate()([feature2, up8])
        conv8 = Conv2D(self.n_filters * 1, kernel_size=3, activation='relu', padding='same')(concat8)
        conv8 = Conv2D(self.n_filters * 1, kernel_size=3, activation='relu', padding='same')(conv8)

        up9 = Conv2D(int(self.n_filters / 2), 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
        feature1 = Conv2D(int(self.n_filters / 2), kernel_size=3, activation='relu', padding='same')(conv1)
        concat9 = Concatenate()([feature1, up9])
        conv9 = Conv2D(int(self.n_filters / 2), kernel_size=3, activation='relu', padding='same')(concat9)
        conv9 = Conv2D(int(self.n_filters / 2), kernel_size=3, activation='relu', padding='same')(conv9)
        conv9 = Conv2D(3, kernel_size=3, activation='relu', padding='same')(conv9)
        conv10 = Conv2D(1, kernel_size=1, activation='sigmoid')(conv9)

        return Model(outputs=conv10, inputs=unet_input), 'unet_model_yuanqing'


#

WEIGHTS_PATH_X = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_MOBILE = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_X_CS = "https://github.com/rdiazgar/keras-deeplab-v3-plus/releases/download/1.2/deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5"
WEIGHTS_PATH_MOBILE_CS = "https://github.com/rdiazgar/keras-deeplab-v3-plus/releases/download/1.2/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5"

def normalize_tuple(value, n, name):
# """用于代替 `conv_utils.normalize_tuple`，确保输入值是 tuple 形式"""
    if isinstance(value, int):
        return (value,) * n
    if isinstance(value, (tuple, list)) and len(value) == n:
        return tuple(value)
    raise ValueError(f"The `{name}` argument must be an integer or a tuple of {n} integers.")

# 示例用法：
kernel_size = normalize_tuple(3, 2, "kernel_size")  # 变成 (3, 3)

class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = K.image_data_format()
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = normalize_tuple(
                output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = normalize_tuple(
                upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return tf.compat.v1.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                       inputs.shape[2] * self.upsampling[1]),
                                              align_corners=True)
        else:
            return tf.compat.v1.image.resize_bilinear(inputs, (self.output_size[0],
                                                       self.output_size[1]),
                                              align_corners=True)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation('relu')(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x


def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    if stride == 1:
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)


def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                    rate=1, depth_activation=False, return_skip=False):
    
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                kernel_size=1,
                                stride=stride)
        shortcut = BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs


def relu6(x):
    return relu(x, max_value=6)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
    in_channels = inputs.shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'expanded_conv_{}_'.format(block_id)
    if block_id:
        # Expand

        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                   use_bias=False, activation=None,
                   name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'expand_BN')(x)
        x = Activation(relu6, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'
    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                        use_bias=False, padding='same', dilation_rate=(rate, rate),
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'depthwise_BN')(x)

    x = Activation(relu6, name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters,
               kernel_size=1, padding='same', use_bias=False, activation=None,
               name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'project_BN')(x)

    if skip_connection:
        return Add(name=prefix + 'add')([inputs, x])

    # if in_channels == pointwise_filters and stride == 1:
    #    return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x

def preprocess_input(x):
    
    return imagenet_utils.preprocess_input(x, mode='tf')


def resize_images_bilinear(X, height_factor=1, width_factor=1, target_height=None, target_width=None, data_format='default'):
    
    if data_format == 'default':
        data_format = K.image_data_format()
    if data_format == 'channels_first':
        # 如果資料格式是 channels_first，需轉換至 channels_last
        X = K.permute_dimensions(X, [0, 2, 3, 1])
    if target_height and target_width:
        new_shape = (target_height, target_width)
    else:
        input_shape = tf.shape(X)
        new_shape = (input_shape[1] * height_factor, input_shape[2] * width_factor)

    # 使用 tf.image.resize 替代過時的方法
    X = tf.image.resize(X, new_shape, method='bilinear')

    if data_format == 'channels_first':
        # 再次轉回 channels_first 格式
        X = K.permute_dimensions(X, [0, 3, 1, 2])

    return X

class BilinearUpSampling2D(Layer):
    def __init__(self, size=(1, 1), target_size=None, data_format='default', **kwargs):
        if data_format == 'default':
            data_format = K.image_data_format()
        self.size = tuple(size)
        self.target_size = tuple(target_size) if target_size is not None else None
        assert data_format in {'channels_last', 'channels_first'}, 'data_format must be in {channels_last, channels_first}'
        self.data_format = data_format
        self.input_spec = [InputSpec(ndim=4)]
        super(BilinearUpSampling2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            height = self.target_size[0] if self.target_size else input_shape[2] * self.size[0]
            width = self.target_size[1] if self.target_size else input_shape[3] * self.size[1]
            return (input_shape[0], input_shape[1], height, width)
        elif self.data_format == 'channels_last':
            height = self.target_size[0] if self.target_size else input_shape[1] * self.size[0]
            width = self.target_size[1] if self.target_size else input_shape[2] * self.size[1]
            return (input_shape[0], height, width, input_shape[3])
        else:
            raise ValueError('Invalid data_format: ' + self.data_format)

    def call(self, x, mask=None):
        if self.target_size is not None:
            return resize_images_bilinear(x, target_height=self.target_size[0], target_width=self.target_size[1], data_format=self.data_format)
        else:
            return resize_images_bilinear(x, height_factor=self.size[0], width_factor=self.size[1], data_format=self.data_format)

    def get_config(self):
        config = {'size': self.size, 'target_size': self.target_size, 'data_format': self.data_format}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


##utils

# ------------------------------------------------------------ #
#
# file : metrics.py
# author : CM
# Metrics for evaluation
#
# ------------------------------------------------------------ #

from keras import backend as K


# dice coefficient

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# Recall (true positive rate)
def recall(truth, prediction):
    TP = K.sum(K.round(K.clip(truth * prediction, 0, 1)))
    P = K.sum(K.round(K.clip(truth, 0, 1)))
    return TP / (P + K.epsilon())


# Specificity (true negative rate)
def specificity(truth, prediction):
    TN = K.sum(K.round(K.clip((1-truth) * (1-prediction), 0, 1)))
    N = K.sum(K.round(K.clip(1-truth, 0, 1)))
    return TN / (N + K.epsilon())


# Precision (positive prediction value)
def precision(truth, prediction):
    TP = K.sum(K.round(K.clip(truth * prediction, 0, 1)))
    FP = K.sum(K.round(K.clip((1-truth) * prediction, 0, 1)))
    return TP / (TP + FP + K.epsilon())


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# ------------------------------------------------------------ #
#
# file : losses.py
# author : CM
# Loss function
#
# ------------------------------------------------------------ #
#import keras.backend as K
from keras import backend as K

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

# Jaccard distance
def jaccard_distance_loss(y_true, y_pred, smooth=100):
    
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def dice_coef_(y_true, y_pred, smooth=1):
    
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)

def dice_coef_loss_(y_true, y_pred):
    return 1 - dice_coef_(y_true, y_pred)


# the deeplab version of dice_loss
def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-10):
    
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    truepos = K.sum(y_true * y_pred)
    fp_and_fn = alpha * K.sum(y_pred * (1 - y_true)) + beta * K.sum((1 - y_pred) * y_true)
    answer = (truepos + smooth) / ((truepos + smooth) + fp_and_fn)
    return -answer

def jaccard_coef_logloss(y_true, y_pred, smooth=1e-10):
    
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    truepos = K.sum(y_true * y_pred)
    falsepos = K.sum(y_pred) - truepos
    falseneg = K.sum(y_true) - truepos
    jaccard = (truepos + smooth) / (smooth + truepos + falseneg + falsepos)
    return -K.log(jaccard + smooth)

##################################################
import os
import cv2
import json
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt

class DataGen:
    def __init__(self, path, split_ratio, x, y, color_space='rgb'):
        self.x = x
        self.y = y
        self.path = path
        self.color_space = color_space
        self.path_train_images = path + "train/images/"
        self.path_train_labels = path + "train/labels/"
        self.path_test_images = path + "test/images/"
        self.path_test_labels = path + "test/labels/"
        self.image_file_list = get_png_filename_list(self.path_train_images)
        self.label_file_list = get_png_filename_list(self.path_train_labels)
        # self.image_file_list = get_jpg_filename_list(self.path_train_images)
        # self.label_file_list = get_jpg_filename_list(self.path_train_labels)
        self.image_file_list[:], self.label_file_list[:] = self.shuffle_image_label_lists_together()
        self.split_index = int(split_ratio * len(self.image_file_list))
        self.x_train_file_list = self.image_file_list[self.split_index:]
        self.y_train_file_list = self.label_file_list[self.split_index:]
        self.x_val_file_list = self.image_file_list[:self.split_index]
        self.y_val_file_list = self.label_file_list[:self.split_index]
        self.x_test_file_list = get_png_filename_list(self.path_test_images)
        self.y_test_file_list = get_png_filename_list(self.path_test_labels)
        # self.x_test_file_list = get_jpg_filename_list(self.path_test_images)
        # self.y_test_file_list = get_jpg_filename_list(self.path_test_labels)

        print("Train images:", self.image_file_list)
        print("Train labels:", self.label_file_list)
    # @staticmethod
    # def get_image_filename_list(directory):
    #     return [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
   
    def generate_data(self, batch_size, train=False, val=False, test=False):
        """Replaces Keras' native ImageDataGenerator."""
        try:
            if train is True:
                image_file_list = self.x_train_file_list
                label_file_list = self.y_train_file_list
            elif val is True:
                image_file_list = self.x_val_file_list
                label_file_list = self.y_val_file_list
            elif test is True:
                image_file_list = self.x_test_file_list
                label_file_list = self.y_test_file_list
        except ValueError:
            print('one of train or val or test need to be True')

        i = 0
        while True:
            image_batch = []
            label_batch = []
            for b in range(batch_size):
                if i == len(self.x_train_file_list):
                    i = 0
                if i < len(image_file_list):
                    sample_image_filename = image_file_list[i]
                    sample_label_filename = label_file_list[i]
                    # print('image: ', image_file_list[i])
                    # print('label: ', label_file_list[i])
                    if train or val:
                        image = cv2.imread(self.path_train_images + sample_image_filename, 1)
                        label = cv2.imread(self.path_train_labels + sample_label_filename, 0)
                    elif test is True:
                        image = cv2.imread(self.path_test_images + sample_image_filename, 1)
                        label = cv2.imread(self.path_test_labels + sample_label_filename, 0)
                    # image, label = self.change_color_space(image, label, self.color_space)
                    label = np.expand_dims(label, axis=2)
                    if image.shape[0] == self.x and image.shape[1] == self.y:
                        image_batch.append(image.astype("float32"))
                    else:
                        print('the input image shape is not {}x{}'.format(self.x, self.y))
                    if label.shape[0] == self.x and label.shape[1] == self.y:
                        label_batch.append(label.astype("float32"))
                    else:
                        print('the input label shape is not {}x{}'.format(self.x, self.y))
                i += 1
            if image_batch and label_batch:
                image_batch = normalize(np.array(image_batch))
                label_batch = normalize(np.array(label_batch))
                yield (image_batch, label_batch)

    def get_num_data_points(self, train=False, val=False):
        try:
            image_file_list = self.x_train_file_list if val is False and train is True else self.x_val_file_list
        except ValueError:
            print('one of train or val need to be True')

        return len(image_file_list)

    # def shuffle_image_label_lists_together(self):
    #     combined = list(zip(self.image_file_list, self.label_file_list))
    #     random.shuffle(combined)
    #     return zip(*combined)

    def shuffle_image_label_lists_together(self):
        image_files = self.image_file_list
        label_files = self.label_file_list
        if len(image_files) == 0 or len(label_files) == 0:
            print("Error: No images or labels found!")
            return [], []  # 確保回傳兩個 list
        combined = list(zip(image_files, label_files))
        random.shuffle(combined)
        return zip(*combined)  # 確保仍然回傳兩個 list

    @staticmethod
    def change_color_space(image, label, color_space):
        if color_space.lower() is 'hsi' or 'hsv':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2HSV)
        elif color_space.lower() is 'lab':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2LAB)
        return image, label
    
def normalize(arr):
    diff = np.amax(arr) - np.amin(arr)
    diff = 255 if diff == 0 else diff
    arr = arr / np.absolute(diff)
    return arr


def get_png_filename_list(path):
    file_list = []
    for FileNameLength in range(0, 500):
        for dirName, subdirList, fileList in os.walk(path):
            for filename in fileList:
                # check file extension
                if ".png" in filename.lower() and len(filename) == FileNameLength:
                    file_list.append(filename)
            break
    file_list.sort()
    return file_list


def get_jpg_filename_list(path):
    file_list = []
    for FileNameLength in range(0, 500):
        for dirName, subdirList, fileList in os.walk(path):
            for filename in fileList:
                # check file extension
                if ".jpg" in filename.lower() and len(filename) == FileNameLength:
                    file_list.append(filename)
            break
    file_list.sort()
    return file_list


def load_jpg_images(path):
    file_list = get_jpg_filename_list(path)
    temp_list = []
    for filename in file_list:
        img = cv2.imread(path + filename, 1)
        temp_list.append(img.astype("float32"))

    temp_list = np.array(temp_list)
    # x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    return temp_list, file_list


def load_png_images(path):

    temp_list = []
    file_list = get_png_filename_list(path)
    for filename in file_list:
        img = cv2.imread(path + filename, 1)
        temp_list.append(img.astype("float32"))

    temp_list = np.array(temp_list)
    #temp_list = np.reshape(temp_list,(temp_list.shape[0], temp_list.shape[1], temp_list.shape[2], 3))
    return temp_list, file_list


def load_data(path):
    # path_train_images = path + "train/images/padded/"
    # path_train_labels = path + "train/labels/padded/"
    # path_test_images = path + "test/images/padded/"
    # path_test_labels = path + "test/labels/padded/"
    path_train_images = path + "train/images/"
    path_train_labels = path + "train/labels/"
    path_test_images = path + "test/images/"
    path_test_labels = path + "test/labels/"
    x_train, train_image_filenames_list = load_png_images(path_train_images)
    y_train, train_label_filenames_list = load_png_images(path_train_labels)
    x_test, test_image_filenames_list = load_png_images(path_test_images)
    y_test, test_label_filenames_list = load_png_images(path_test_labels)
    x_train = normalize(x_train)
    y_train = normalize(y_train)
    x_test = normalize(x_test)
    y_test = normalize(y_test)
    return x_train, y_train, x_test, y_test, test_label_filenames_list


def load_test_images(path):
    path_test_images = path + "test_images/"
    x_test, test_image_filenames_list = load_jpg_images(path_test_images)
    x_test = normalize(x_test)
    return x_test, test_image_filenames_list


def save_results(np_array, color_space, outpath, test_label_filenames_list):
    i = 0
    for filename in test_label_filenames_list:
        # predict_img = np.reshape(predict_img,(predict_img[0],predict_img[1]))
        pred = np_array[i]
        # if color_space.lower() is 'hsi' or 'hsv':
        #     pred = cv2.cvtColor(pred, cv2.COLOR_HSV2RGB)
        # elif color_space.lower() is 'lab':
        #     pred = cv2.cvtColor(pred, cv2.COLOR_Lab2RGB)
        cv2.imwrite(outpath + filename, pred * 255.)
        i += 1


def save_rgb_results(np_array, outpath, test_label_filenames_list):
    i = 0
    for filename in test_label_filenames_list:
        # predict_img = np.reshape(predict_img,(predict_img[0],predict_img[1]))
        cv2.imwrite(outpath + filename, np_array[i] * 255.)
        i += 1

def save_history(model, model_name, training_history, dataset, n_filters, epoch, learning_rate, loss,
                 color_space, path=None, temp_name=None):
    # 修正：移除非法字符或替換
    save_weight_filename = temp_name if temp_name else datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # 確保路徑存在
    if path and not os.path.exists(path):
        os.makedirs(path)

    # 儲存模型
    model.save('{}{}.hdf5'.format(path, save_weight_filename))
    
    # 儲存歷史記錄
    with open('{}{}.json'.format(path, save_weight_filename), 'w') as f:
        json.dump(training_history.history, f, indent=2)

    # 繪製訓練歷史圖
    json_list = ['{}{}.json'.format(path, save_weight_filename)]
    for json_filename in json_list:
        with open(json_filename) as f:
            # 轉換 JSON 為字典
            loss_dict = json.load(f)
        print_list = ['loss', 'val_loss', 'dice_coef', 'val_dice_coef']
        for item in print_list:
            item_list = []
            if item in loss_dict:
                item_list.extend(loss_dict.get(item))
                plt.plot(item_list)
        plt.title('model:{} lr:{} epoch:{} #filtr:{} Colorspaces:{}'.format(model_name, learning_rate,
                                                                            epoch, n_filters, color_space))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train_loss', 'test_loss', 'train_dice', 'test_dice'], loc='upper left')
        
        # 儲存圖像
        plt.savefig('{}{}.png'.format(path, save_weight_filename))
        plt.show()
        plt.clf()

#### Evaluate

## fill_holes

import cv2
import numpy as np
from scipy.ndimage.measurements import label

def fill_holes(img, threshold, rate):
    binary_img = np.where(img > threshold, 0, 1) #reversed image
    structure = np.ones((3, 3, 3), dtype=np.int)
    labeled, ncomponents = label(binary_img, structure)
    # print(labeled.shape, ncomponents)
    count_list = []
    #count
    for pixel_val in range(ncomponents):
        count = 0
        for y in range(labeled.shape[1]):
            for x in range(labeled.shape[0]):
                if labeled[x][y][0] == pixel_val + 1:
                    count += 1
        count_list.append(count)
    # print(count_list)

    for i in range(len(count_list)):
        # print(i)
        if sum(count_list) != 0:
            if count_list[i] / sum(count_list) < rate:
                for y in range(labeled.shape[1]):
                    for x in range(labeled.shape[0]):
                        if labeled[x][y][0] == i + 1:
                            labeled[x][y] = [0,0,0]
    labeled = np.where(labeled < 1, 1, 0)
    labeled *= 255
    return labeled

## remove_small_areas

import cv2
import numpy as np
from scipy.ndimage.measurements import label

def remove_small_areas(img, threshold, rate):
    structure = np.ones((3, 3, 3), dtype=np.int)
    labeled, ncomponents = label(img, structure)
    # print(labeled.shape, ncomponents)
    count_list = []
    # count
    for pixel_val in range(ncomponents):
        count = 0
        for y in range(labeled.shape[1]):
            for x in range(labeled.shape[0]):
                if labeled[x][y][0] == pixel_val + 1:
                    count += 1
        count_list.append(count)
    # print(count_list)

    for i in range(len(count_list)):
        # print(i)
        if sum(count_list) != 0:
            if count_list[i] / sum(count_list) < rate:
                for y in range(labeled.shape[1]):
                    for x in range(labeled.shape[0]):
                        if labeled[x][y][0] == i + 1:
                            labeled[x][y] = [0, 0, 0]
    labeled = np.where(labeled < 1, 0, 1)
    labeled *= 255
    return labeled

def evaluate(threshold, file_list, label_path, post_prosecced_path):
    false_positives = 0
    false_negatives = 0
    true_positives = 0

    for img_name in tqdm(file_list):
        img = cv2.imread(pred_dir + img_name)
        _, threshed = cv2.threshold(img, threshold, 255, type=cv2.THRESH_BINARY)
        ################################################################################################################
        # call image post processing functions
        mask = np.zeros((226, 226, 3))
        filled = fill_holes(threshed, threshold,0.1)
        denoised = remove_small_areas(filled, threshold, 0.05)
        ################################################################################################################
        cv2.imwrite('whatever/filled/' + img_name, filled)
        cv2.imwrite('whatever/post_processed/' + img_name, denoised)


    for filename in tqdm(file_list):
        label = cv2.imread(label_path + filename,0)
        post_prosecced = cv2.imread(post_prosecced_path + filename,0)
        xdim = label.shape[0]
        ydim = label.shape[1]
        for x in range(xdim):
            for y in range(ydim):
                if post_prosecced[x, y] and label[x, y] > threshold:
                    true_positives += 1
                if label[x, y] > threshold > post_prosecced[x, y]:
                    false_negatives += 1
                if label[x, y] < threshold < post_prosecced[x, y]:
                    false_positives += 1
 # IOU = float(true_positives) / (true_positives + false_negatives + false_positives)
 # Add a check for zero division
    denominator = true_positives + false_negatives + false_positives
    if denominator == 0:
        IOU = 0  # or any other appropriate value for this case
        Dice = 0 # or any other appropriate value for this case
        print("Warning: Denominator is zero. IOU and Dice set to 0.")
    else:
        IOU = float(true_positives) / denominator
        Dice = 2 * float(true_positives) / (2 * true_positives + false_negatives + false_positives)



    print("--------------------------------------------------------")
    print("Weight file: ",post_prosecced_path.rsplit("/")[1])
    print("--------------------------------------------------------")
    print("Threshold: ", threshold)
    print("True  pos = " + str(true_positives))
    print("False neg = " + str(false_negatives))
    print("False pos = " + str(false_positives))
    print("IOU = " + str(IOU))
    print("Dice = " + str(Dice))

### Predict ###

import cv2
from keras.models import load_model
#from keras.utils.generic_utils import CustomObjectScope
from keras.utils import CustomObjectScope
from datetime import datetime, timezone, timedelta

# settings
input_dim_x = 224
input_dim_y = 224
color_space = 'rgb'
# path = './data/Medetec_foot_ulcer_224/'
# path = './data/wound_datasets_project/'
path = "./photo/"
weight_file_name = '2025-01-22_21-26-41.hdf5'


# 台灣時間（UTC+8）
taiwan_timezone = timezone(timedelta(hours=8))
# 格式化成字串
pred_save_path = datetime.now(tz=taiwan_timezone).strftime("%Y-%m-%d_%H-%M-%S") + "/"


data_gen = DataGen(path, split_ratio=0.0, x=input_dim_x, y=input_dim_y, color_space=color_space)
x_test, test_label_filenames_list = load_test_images(path)


### get unet model
unet2d = Unet2D(n_filters=64, input_dim_x=input_dim_x, input_dim_y=input_dim_y, num_channels=3)
model = unet2d.get_unet_model_yuanqing()
model = load_model('./woundSeverity/' + weight_file_name
                , custom_objects={'recall':recall,
                                  'precision':precision,
                                  'dice_coef': dice_coef,
                                  'relu6':relu6,
                                  'DepthwiseConv2D':DepthwiseConv2D,
                                  'BilinearUpsampling':BilinearUpsampling})


for image_batch, label_batch in data_gen.generate_data(batch_size=len(x_test), test=True):
    prediction = model.predict(image_batch, verbose=1)
    save_results(prediction, 'rgb', path + 'test_predictions/' + pred_save_path, test_label_filenames_list)
    break

print(prediction)


#### 模型偵測位置後框出邊緣，疊合原圖 ###
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 假設 image_batch 是原圖的批次，prediction 是模型的輸出
for i in range(min(5, len(prediction))):  # 顯示前5個預測結果
    original_image = (image_batch[i] * 255).astype(np.uint8)  # 轉成 uint8 格式
    mask = (prediction[i] > 0.5).astype(np.uint8) * 255  # 二值化遮罩 (0 or 255)
    
    # 找到輪廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 建立一個副本來畫輪廓
    contour_image = original_image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)  # 綠色輪廓

    # 顯示結果
    plt.figure(figsize=(12, 6))

    # 原始圖像
    plt.subplot(1, 3, 1)
    plt.title("original")
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # 預測傷口位置
    plt.subplot(1, 3, 2)
    plt.title("wound_prediction")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    # 疊合輪廓的圖像
    plt.subplot(1, 3, 3)
    plt.title("location")
    plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()

#### 20250321_分類傷口嚴重程度(test 1)整張圖 #######
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def extract_pixel_features(image):
    """
    提取影像的像素特徵，包括：
    1. RGB 色彩
    2. HSV 色彩
    3. 梯度 (Sobel)
    """
    # 轉換成 HSV 色彩空間
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 轉換成灰階影像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 計算 X 和 Y 方向的 Sobel 梯度
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 將梯度轉換為 0~255
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    
    # 將所有特徵堆疊
    features = np.dstack([image, hsv, sobel_x, sobel_y])
    
    # 攤平成 (num_pixels, num_features)
    features = features.reshape(-1, features.shape[-1])
    
    return features

def kmeans_clustering(image, n_clusters=4):
    """
    使用 K-Means 進行像素級別分類。
    """
    # 提取像素特徵
    features = extract_pixel_features(image)

    # 使用 K-Means 進行分類
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    # 轉回影像尺寸
    clustered_image = labels.reshape(image.shape[:2])

    return clustered_image

# 讀取影像
#image_path = "data/wound_datasets_project/images/train/6_0.jpg"  # 請換成你的影像路徑
image_path = "./photo/test_images/diabetic_foot_ulcer_0028.jpg"  # 影像路徑
image = cv2.imread(image_path)

# 進行 K-Means 分類
clustered_image = kmeans_clustering(image, n_clusters=4)

# 顯示原始影像和分類結果
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(clustered_image, cmap="jet")  # 使用彩色 colormap 顯示不同分類
plt.title("K-Means Clustering")
plt.axis("off")

plt.show()

#### 20250321_分類傷口嚴重程度(test 2)只對傷口 #######
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.cluster import KMeans

def get_wound_mask(image):
    # 影像預處理 (縮放到模型輸入大小)
    input_image = cv2.resize(image, (256, 256))  # 根據你的模型大小調整
    input_image = input_image / 255.0  # 正規化
    input_image = np.expand_dims(input_image, axis=0)  # 增加 batch 維度

    # 讓模型預測傷口區域
    mask = model.predict(input_image)[0, :, :, 0]  # 取出 mask
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))  # 轉回原圖大小

    # 轉為二值圖：將機率 > 0.5 的區域視為傷口
    binary_mask = (mask > 0.3).astype(np.uint8)

    return binary_mask

# 讀取影像
#image_path = "data/Medetec_foot_ulcer_224/test/images/foot-ulcer-0028_1.png"  # 影像路徑
image_path = "./photo/test_images/diabetic_foot_ulcer_0028.jpg"  # 請換成你的影像路徑
image = cv2.imread(image_path)
mask = get_wound_mask(image)


# 顯示結果
cv2.imshow("Wound Mask", mask * 255)  # 乘 255 讓掩碼變白色
cv2.waitKey(0)
cv2.destroyAllWindows()

#####
import matplotlib.pyplot as plt
mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
plt.figure(figsize=(6,6))
plt.subplot(1,2,1)
plt.imshow(image)  # 原圖
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(mask, cmap="gray")  # 預測的傷口 mask
plt.axis("off")

plt.show()

#####
# 傷口掩碼 (來自 UNet / YOLO)

mask = get_wound_mask(image)
mask = (mask > 0.5).astype(np.uint8)  # 轉為 0-1 格式

# def classify_wound_severity(image, mask, num_clusters=3):
#     """
#     使用 K-Means 對傷口區域進行分類，並回傳分類結果（severity_map）。
    
#     參數：
#     - image: (H, W, 3) 原始 RGB 圖片
#     - mask: (H, W) 傷口區域的二值遮罩（0：非傷口，1：傷口）
#     - num_clusters: 要分成幾類（預設 4 類）
    
#     回傳：
#     - severity_map: (H, W) 分類後的傷口嚴重度地圖，數值範圍為 0 ~ num_clusters-1
#     """
#     # 確保 mask 為二值化
#     mask = (mask > 0.5).astype(np.uint8)

#     # 取得傷口區域的像素索引
#     wound_pixels = np.where(mask == 1)
    
#     if len(wound_pixels[0]) == 0:
#         print("⚠️ 沒有偵測到傷口，回傳全零的分類圖")
#         return np.zeros_like(mask, dtype=np.uint8)

#     # 取得傷口區域的 RGB 值
#     wound_rgb = image[wound_pixels]

#     # 使用 K-Means 進行分類 (分成 4 類)
#     kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
#     labels = kmeans.fit_predict(wound_rgb)

#     # 創建分類結果圖
#     severity_map = np.zeros_like(mask, dtype=np.uint8)
#     severity_map[wound_pixels[0], wound_pixels[1]] = labels+1  # K-Means 結果是 0~3，直接對應 W0~W3

#     return severity_map

def classify_wound_severity(image, mask, num_clusters=3):
    # 轉 HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # 取得傷口區域索引
    wound_pixels = np.where(mask == 1)
    if len(wound_pixels[0]) == 0:
        print("⚠️ 沒偵測到傷口，回傳全 0 分群圖")
        return np.zeros_like(mask, dtype=np.uint8)

    # 擷取傷口區域 HSV 值
    wound_hsv = hsv_image[wound_pixels]

    # KMeans 分群
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(wound_hsv)

    # 建立分群結果圖
    severity_map = np.zeros_like(mask, dtype=np.uint8)
    severity_map[wound_pixels[0], wound_pixels[1]] = labels + 1  # 從 1 開始，0 當背景

    # 顯示群中心（可用來理解每類代表什麼）
    print("HSV 群中心 (H, S, V)：\n", kmeans.cluster_centers_)

    return severity_map

def visualize_classification(image, severity_map):
    classified_image = image.copy()

    # 確保 severity_map 尺寸正確
    if severity_map.shape[:2] != image.shape[:2]:
        raise ValueError("severity_map shape does not match image shape")

    # 繪製分類結果（假設不同分類用不同顏色）
    classified_image[severity_map == 1] = [0, 255, 0]  # 綠色
    classified_image[severity_map == 2] = [255, 0, 0]  # 藍色
    classified_image[severity_map == 3] = [0, 255, 255]    # 黃色
    #classified_image[severity_map == 4] = [0, 0, 255]   #紅色


    return classified_image

# 假設你的影像是 RGB (H, W, 3)
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 執行分類
severity_map = classify_wound_severity(image, mask, num_clusters=3)
visual_result = +visualize_classification(image_rgb, severity_map)

# 顯示
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title("mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(visual_result)
plt.title("location")
plt.axis("off")

plt.tight_layout()
plt.show()

###20250323_分類框去原圖上[要加前面的一起用](method 1)#####

severity_colors = {
    1: (0, 255, 0),    # W0 - 快癒合 (綠色)
    2: (255, 0, 0),    # W1 - 輕微 (藍色)
    3: (0, 255, 255),  # W2 - 中度 (黃色)
    #4: (0, 0, 255)     # W3 - 嚴重 (紅色)
}

overlay = image.copy()
for severity, color in severity_colors.items():
    mask = (severity_map == severity).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, thickness=1)

plt.figure(figsize=(6,6))
plt.imshow(overlay)
plt.axis("off")
plt.show()


############# 檢測使用 ############
##須出現 Severity Levels in Map: [0 1 2 3] ###
unique_values = np.unique(severity_map)
print("Severity Levels in Map:", unique_values)

##看程度分布用####
plt.imshow(severity_map, cmap='jet')  # 或 cmap='gray'

############# 20250327_嚴重程度分級分類_RGB比例 ##############
import numpy as np

def classify_wound_by_rgb(image, mask):
    """
    根據傷口區域的 RGB 比例來分類傷口嚴重程度 (W0~W3)

    參數：
    - image: (H, W, 3) 原始 RGB 圖片
    - mask: (H, W) 傷口區域遮罩 (0: 非傷口, 1: 傷口)

    回傳：
    - severity_map: (H, W) 每個像素的傷口嚴重程度 (0~3)
    """
    # 取得傷口區域像素索引
    mask = (mask > 0.5).astype(np.uint8)
    wound_pixels = np.where(mask == 1)

    if len(wound_pixels[0]) == 0:
        print("⚠️ 沒有偵測到傷口，回傳全零的分類圖")
        return np.zeros_like(mask, dtype=np.uint8)

    # 取得傷口區域的 RGB 值
    wound_rgb = image[wound_pixels]

    # 計算 RGB 比例
    R = wound_rgb[:, 0].astype(np.float32)
    G = wound_rgb[:, 1].astype(np.float32)
    B = wound_rgb[:, 2].astype(np.float32)
    total = R + G + B + 1e-6  # 避免除以 0

    r_ratio = R / total
    g_ratio = G / total
    b_ratio = B / total

    # 初始化分類結果
    severity_map = np.zeros_like(mask, dtype=np.uint8)

    # **W0 (粉色傷口) - 逐漸癒合**  
    # - R 高 (0.5 以上)，G 適中 (0.3~0.5)，B 低 (0.2 以下)
    w0_condition = (r_ratio > 0.5) & (g_ratio > 0.3) & (g_ratio < 0.5) & (b_ratio < 0.2)

    # **W1 (紅色傷口) - 正在癒合**  
    # - R 非常高 (0.6 以上)，G 低 (0.2 以下)，B 低 (0.2 以下)
    w1_condition = (r_ratio > 0.6) & (g_ratio < 0.2) & (b_ratio < 0.2)

    # **W2 (黃綠色傷口) - 感染/腐肉**  
    # - G 高 (0.4 以上)，R 低 (0.3 以下)，B 低 (0.3 以下)
    w2_condition = (g_ratio > 0.4) & (r_ratio < 0.3) & (b_ratio < 0.3)

    # **W3 (黑色傷口) - 結痂壞死**  
    # - R、G、B 都低 (0.3 以下)
    w3_condition = (r_ratio < 0.3) & (g_ratio < 0.3) & (b_ratio < 0.3)

    # 依照條件分類
    severity_map[wound_pixels[0][w0_condition], wound_pixels[1][w0_condition]] = 1  # W0
    severity_map[wound_pixels[0][w1_condition], wound_pixels[1][w1_condition]] = 2  # W1
    severity_map[wound_pixels[0][w2_condition], wound_pixels[1][w2_condition]] = 3  # W2
    severity_map[wound_pixels[0][w3_condition], wound_pixels[1][w3_condition]] = 4  # W3

    return severity_map


#image_path = "data/Medetec_foot_ulcer_224/test/images/foot-ulcer-0028_1.png"  # 影像路徑
image_path = "./photo/test_images/diabetic_foot_ulcer_0028.jpg"  # 請換成你的影像路徑
image = cv2.imread(image_path)
mask = get_wound_mask(image)
severity_map = classify_wound_by_rgb(image, mask)


# 顯示結果
plt.imshow(severity_map, cmap="jet", vmin=0, vmax=5)
plt.colorbar()
plt.axis("off")
plt.show()

print(np.unique(severity_map))
print("Unique values in mask:", np.unique(mask))
print("Number of wound pixels:", len(wound_pixels[0]))

##20250422

def plot_rgb_ratios(image):
    R = image[:, :, 0].astype(np.float32)
    G = image[:, :, 1].astype(np.float32)
    B = image[:, :, 2].astype(np.float32)
    
    total = R + G + B + 1e-6
    r_ratio = R / total
    g_ratio = G / total
    b_ratio = B / total

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(r_ratio, cmap='Reds')
    plt.title("R ratio")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(g_ratio, cmap='Greens')
    plt.title("G ratio")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(b_ratio, cmap='Blues')
    plt.title("B ratio")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

plot_rgb_ratios(image)

#2
def show_rgb_ratio_image(image):
    R = image[:, :, 0].astype(np.float32)
    G = image[:, :, 1].astype(np.float32)
    B = image[:, :, 2].astype(np.float32)

    total = R + G + B + 1e-6
    r_ratio = R / total
    g_ratio = G / total
    b_ratio = B / total

    ratio_image = np.stack([r_ratio, g_ratio, b_ratio], axis=2)
    plt.figure(figsize=(6, 6))
    plt.imshow(ratio_image)
    plt.title("RGB Ratio Composite")
    plt.axis("off")
    plt.show()

show_rgb_ratio_image(image)

#3
def plot_rgb_ratios_on_wound(image, mask):
    """
    顯示 RGB ratio 熱圖，只針對傷口區域
    - image: 原始 RGB 圖像 (H, W, 3)
    - mask: 傷口區域的 mask (0: 非傷口, 1: 傷口)
    """
    # 計算 RGB
    R = image[:, :, 0].astype(np.float32)
    G = image[:, :, 1].astype(np.float32)
    B = image[:, :, 2].astype(np.float32)
    total = R + G + B + 1e-6  # 防止除以 0

    r_ratio = R / total
    g_ratio = G / total
    b_ratio = B / total

    # 將非傷口區域設為 NaN 不畫出來
    r_ratio_masked = np.where(mask == 1, r_ratio, np.nan)
    g_ratio_masked = np.where(mask == 1, g_ratio, np.nan)
    b_ratio_masked = np.where(mask == 1, b_ratio, np.nan)

    # 畫圖
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(r_ratio_masked, cmap='Reds')
    plt.title("R ratio (wound only)")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(g_ratio_masked, cmap='Greens')
    plt.title("G ratio (wound only)")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(b_ratio_masked, cmap='Blues')
    plt.title("B ratio (wound only)")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

# 假設你有 image（RGB 格式）和 mask（0-1）
plot_rgb_ratios_on_wound(image, mask)

#4
def analyze_rgb_ratios_on_wound(image, mask):
    """
    顯示傷口區域的 RGB 比例熱圖與統計資訊。
    - image: 原始 RGB 圖像 (H, W, 3)
    - mask: 傷口區域的 mask (0: 非傷口, 1: 傷口)
    """
    # 計算 RGB 原始值
    R = image[:, :, 0].astype(np.float32)
    G = image[:, :, 1].astype(np.float32)
    B = image[:, :, 2].astype(np.float32)
    total = R + G + B + 1e-6

    # 計算 RGB 比例
    r_ratio = R / total
    g_ratio = G / total
    b_ratio = B / total

    # 僅針對傷口區域擷取值
    wound_mask = (mask == 1)
    r_vals = r_ratio[wound_mask]
    g_vals = g_ratio[wound_mask]
    b_vals = b_ratio[wound_mask]

    # 印出統計資訊
    print("🔍 RGB 比例（僅限傷口區域）")
    print(f"R - Min: {np.min(r_vals):.3f}, Max: {np.max(r_vals):.3f}, Mean: {np.mean(r_vals):.3f}")
    print(f"G - Min: {np.min(g_vals):.3f}, Max: {np.max(g_vals):.3f}, Mean: {np.mean(g_vals):.3f}")
    print(f"B - Min: {np.min(b_vals):.3f}, Max: {np.max(b_vals):.3f}, Mean: {np.mean(b_vals):.3f}")

    # 畫圖（只顯示傷口區域）
    r_map = np.where(wound_mask, r_ratio, np.nan)
    g_map = np.where(wound_mask, g_ratio, np.nan)
    b_map = np.where(wound_mask, b_ratio, np.nan)

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(r_map, cmap='Reds')
    plt.title("🔴 R ratio (wound only)")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(g_map, cmap='Greens')
    plt.title("🟢 G ratio (wound only)")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(b_map, cmap='Blues')
    plt.title("🔵 B ratio (wound only)")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

analyze_rgb_ratios_on_wound(image, mask)

##繪製
def classify_wound_by_rgb(image, mask):
    """
    根據實際傷口區域 RGB 比例分類 W1:紅、W2:黃、W3:黑
    """
    mask = (mask > 0.5).astype(np.uint8)
    wound_pixels = np.where(mask == 1)

    if len(wound_pixels[0]) == 0:
        print("⚠️ 沒有偵測到傷口，回傳全零的分類圖")
        return np.zeros_like(mask, dtype=np.uint8), wound_pixels

    wound_rgb = image[wound_pixels]
    R = wound_rgb[:, 0].astype(np.float32)
    G = wound_rgb[:, 1].astype(np.float32)
    B = wound_rgb[:, 2].astype(np.float32)
    total = R + G + B + 1e-6

    r_ratio = R / total
    g_ratio = G / total
    b_ratio = B / total

    severity_map = np.zeros_like(mask, dtype=np.uint8)

    # 新分類條件（根據你實際的色彩比例）
    w1 = (r_ratio > 0.31) & (g_ratio > 0.27) & (b_ratio < 0.39)
    w2 = (r_ratio > 0.30) & (g_ratio > 0.30) & (b_ratio < 0.36)
    w3 = (r_ratio < 0.26) & (g_ratio < 0.26) & (b_ratio < 0.38)

    severity_map[wound_pixels[0][w1], wound_pixels[1][w1]] = 1  # W1 紅
    severity_map[wound_pixels[0][w2], wound_pixels[1][w2]] = 2  # W2 黃
    severity_map[wound_pixels[0][w3], wound_pixels[1][w3]] = 3  # W3 黑

    return severity_map, wound_pixels

# ==== 主程式區 ====
image_path = "./photo/test_images/diabetic_foot_ulcer_0028.jpg"  # 影像路徑
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 轉成 RGB

mask = get_wound_mask(image)
severity_map, wound_pixels = classify_wound_by_rgb(image, mask)

# 顯示分類圖
plt.imshow(severity_map, cmap="jet", vmin=0, vmax=3)
plt.colorbar(label="Wound Class (0:None, 1:Red, 2:Yellow, 3:Black)")
plt.axis("off")
plt.title("Wound Classification Result")
plt.show()

# 統計資訊
print("✅ 分類結果類別:", np.unique(severity_map))
print("🩹 原始遮罩類別:", np.unique(mask))
print("🧮 傷口像素數量:", len(wound_pixels[0]))

import matplotlib.pyplot as plt
import matplotlib.cm as cm
# 將分類圖映射為彩色
colored_severity = cm.get_cmap('jet')(severity_map / 3.0)[..., :3]  # Normalize to [0,1] 並取 RGB
colored_severity = (colored_severity * 255).astype(np.uint8)

# 疊合：原圖 + 分類圖 (控制透明度)
alpha = 0.5  # 透明度控制 (0 = 只看原圖，1 = 只看標註)
overlay = (image * (1 - alpha) + colored_severity * alpha).astype(np.uint8)

# 顯示疊合圖
plt.figure(figsize=(10, 5))
plt.imshow(overlay)
plt.axis("off")
plt.title("Overlay: Original + Wound Classification")
plt.show()