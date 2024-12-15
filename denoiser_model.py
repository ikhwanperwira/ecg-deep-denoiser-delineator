# @title Denoiser Architecture Pretrained Model (Baseline Wander Remover)
from keras.api.layers import Input, Conv1D, Dropout, BatchNormalization, concatenate
from keras.api.models import Model


def LANLFilter_module(x, layers):
  LB0 = Conv1D(filters=int(layers / 8),
               kernel_size=3,
               activation='linear',
               strides=1,
               padding='same')(x)
  LB1 = Conv1D(filters=int(layers / 8),
               kernel_size=5,
               activation='linear',
               strides=1,
               padding='same')(x)
  LB2 = Conv1D(filters=int(layers / 8),
               kernel_size=9,
               activation='linear',
               strides=1,
               padding='same')(x)
  LB3 = Conv1D(filters=int(layers / 8),
               kernel_size=15,
               activation='linear',
               strides=1,
               padding='same')(x)

  NLB0 = Conv1D(filters=int(layers / 8),
                kernel_size=3,
                activation='relu',
                strides=1,
                padding='same')(x)
  NLB1 = Conv1D(filters=int(layers / 8),
                kernel_size=5,
                activation='relu',
                strides=1,
                padding='same')(x)
  NLB2 = Conv1D(filters=int(layers / 8),
                kernel_size=9,
                activation='relu',
                strides=1,
                padding='same')(x)
  NLB3 = Conv1D(filters=int(layers / 8),
                kernel_size=15,
                activation='relu',
                strides=1,
                padding='same')(x)

  x = concatenate([LB0, LB1, LB2, LB3, NLB0, NLB1, NLB2, NLB3])

  return x


def LANLFilter_module_dilated(x, layers):
  LB1 = Conv1D(filters=int(layers / 6),
               kernel_size=5,
               activation='linear',
               dilation_rate=3,
               padding='same')(x)
  LB2 = Conv1D(filters=int(layers / 6),
               kernel_size=9,
               activation='linear',
               dilation_rate=3,
               padding='same')(x)
  LB3 = Conv1D(filters=int(layers / 6),
               kernel_size=15,
               dilation_rate=3,
               activation='linear',
               padding='same')(x)

  NLB1 = Conv1D(filters=int(layers / 6),
                kernel_size=5,
                activation='relu',
                dilation_rate=3,
                padding='same')(x)
  NLB2 = Conv1D(filters=int(layers / 6),
                kernel_size=9,
                activation='relu',
                dilation_rate=3,
                padding='same')(x)
  NLB3 = Conv1D(filters=int(layers / 6),
                kernel_size=15,
                dilation_rate=3,
                activation='relu',
                padding='same')(x)

  x = concatenate([LB1, LB2, LB3, NLB1, NLB2, NLB3])
  # x = BatchNormalization()(x)

  return x


def get_denoiser(signal_size=512):
  input_shape = (signal_size, 1)

  # Denoiser
  denoiser_input_layer = Input(shape=input_shape, dtype='float64')
  denoiser_hidden_layer = LANLFilter_module(denoiser_input_layer, 64)
  denoiser_hidden_layer = Dropout(0.4)(denoiser_hidden_layer)
  denoiser_hidden_layer = BatchNormalization()(denoiser_hidden_layer)
  denoiser_hidden_layer = LANLFilter_module_dilated(denoiser_hidden_layer, 64)
  denoiser_hidden_layer = Dropout(0.4)(denoiser_hidden_layer)
  denoiser_hidden_layer = BatchNormalization()(denoiser_hidden_layer)
  denoiser_hidden_layer = LANLFilter_module(denoiser_hidden_layer, 32)
  denoiser_hidden_layer = Dropout(0.4)(denoiser_hidden_layer)
  denoiser_hidden_layer = BatchNormalization()(denoiser_hidden_layer)
  denoiser_hidden_layer = LANLFilter_module_dilated(denoiser_hidden_layer, 32)
  denoiser_hidden_layer = Dropout(0.4)(denoiser_hidden_layer)
  denoiser_hidden_layer = BatchNormalization()(denoiser_hidden_layer)
  denoiser_hidden_layer = LANLFilter_module(denoiser_hidden_layer, 16)
  denoiser_hidden_layer = Dropout(0.4)(denoiser_hidden_layer)
  denoiser_hidden_layer = BatchNormalization()(denoiser_hidden_layer)
  denoiser_hidden_layer = LANLFilter_module_dilated(denoiser_hidden_layer, 16)
  denoiser_hidden_layer = Dropout(0.4)(denoiser_hidden_layer)
  denoiser_hidden_layer = BatchNormalization()(denoiser_hidden_layer)
  denoiser_output_layer = Conv1D(filters=1,
                                 kernel_size=9,
                                 activation='linear',
                                 strides=1,
                                 padding='same')(denoiser_hidden_layer)
  denoiser_model = Model(inputs=denoiser_input_layer,
                         outputs=denoiser_output_layer)

  # load the weights
  denoiser_model.load_weights(
      'pretrain_deepfilter_mblanld_isysrg_360hz.weights.h5')
  return denoiser_model
