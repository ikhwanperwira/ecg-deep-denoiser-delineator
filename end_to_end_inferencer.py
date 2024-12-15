from keras.api import ops
from scipy.signal import resample_poly
import tensorflow as tf
from denoiser_model import get_denoiser
import numpy as np
from delineator_model import get_delineator


def resample_amplitude(signal, current_sample_rate, desired_sample_rate):
  """
  Resample the signal and annotation to the desired sample rate.

  Args:
    signal (tensor): The ECG signal with shape (batch_size, signal_length).
    current_sample_rate (uint): The current sample rate of the signal and annotation.
    desired_sample_rate (uint): The desired sample rate of the signal and annotation.

  Returns:
    tensor: The resampled signal with shape (batch_size, resampled_length).
  """

  # checkk if scalar
  assert isinstance(current_sample_rate,
                    int), "current_sample_rate must be an integer"
  assert isinstance(desired_sample_rate,
                    int), "desired_sample_rate must be an integer"

  # check dims
  assert signal.ndim == 2, "signal must have 2 dimensions which are (batch_size, signal_length)"

  # is downsample or upsample?
  ratio = desired_sample_rate / current_sample_rate
  if ratio < 1:
    # downsample
    tensor = ops.convert_to_tensor(signal, dtype='float32')
    siglen = ops.shape(tensor)[-1]
    resampled_indices = ops.cast(
        ops.round(ops.arange(0, siglen, 1/ratio)), dtype='int64')
    # make sure indices are within the bounds of the original array
    resampled_indices = ops.clip(resampled_indices, 0, siglen - 1)
    # resampled_signal = signal[:, resampled_indices]
    resampled_signal = tf.gather(tensor, resampled_indices, axis=-1)
  elif ratio > 1:
    # upsample
    resampled_signal = ops.convert_to_tensor(resample_poly(
        signal, desired_sample_rate, current_sample_rate, axis=-1), dtype='float32')
  else:
    # same sample rate
    resampled_signal = signal

  return ops.cast(resampled_signal, 'float32')


def perform_bw_denoising(signal, sample_rate):
  """
  Perform baseline wander denoising to the signal. Note, the signal must be 360Hz sample rate.

  Args:
    signal (ndarray): The ECG signal with shape (batch_size, signal_length).
    sample_rate (uint): The sample rate of the signal.

  Returns:
    denoised_signal (ndarray): The denoised signal with shape (batch_size, signal_length).
  """

  # check dims
  assert signal.ndim == 2, "signal must have 2 dimensions which are (batch_size, signal_length)"

  # check sample rate
  assert sample_rate == 360, "sample_rate must be 360Hz"

  # add channel for model input compatibility
  signal = ops.reshape(signal, (-1, ops.shape(signal)[1], 1))

  # to keras tensor
  signal = ops.convert_to_tensor(signal, dtype='float64')

  # instantiate denoiser
  denoiser = get_denoiser(signal_size=ops.shape(signal)[1])

  # perform actual denoising
  denoised_signal = denoiser.predict(signal)

  # remove channel dimension
  denoised_signal = signal = ops.reshape(signal, (-1, ops.shape(signal)[1]))

  return denoised_signal


def ieee754_to_uint8(x, axis=-1):
  """
  Normalize a tensor using IEEE 754 logic and map it to uint8 values using Keras 3 ops.

  Args:
      x (Tensor): A Keras tensor of shape (batch_size, seq_len).
      axis (int): Axis along which to normalize.

  Returns:
      A Keras tensor with dtype uint8, returning the same shape as input x.
  """
  # Find the maximum absolute value along the given axis
  m = ops.max(ops.abs(x), axis=axis, keepdims=True)
  m = ops.where(m == 0, ops.ones_like(m), m)  # Prevent division by zero

  # Perform normalization
  y = (2**7 - 1 * ops.cast(x > 0, dtype="float32")) * x / m

  # Convert to uint8
  y = ops.cast(ops.round(y + 128), dtype="uint8")
  return y


def windower(signal, window_size=256):
  """
  Window the signal.

  Args:
    signal (tensor: uint): The ECG signal with shape (batch_size, signal_length).
    window_size (uint): The window size.
    stride (uint): The stride size.

  Returns:
    windows (tensor: uint): The windows with shape (batch_size, window_size)
    pad_len (uint): Reference for post-processing of real signal length
    num_win (uint): Reference for post-processing of real signal length
  """

  # check dims
  assert signal.ndim == 2, "signal must have 2 dimensions which are (batch_size, signal_length)"

  # check if window_size is less than signal_length
  assert window_size <= signal.shape[-1], "window_size must be less than or equal to signal_length"

  pad_length = (window_size - (ops.shape(signal)
                [-1] % window_size)) % window_size
  padded_signal = ops.pad(signal, ((0, 0), (0, pad_length)), mode='constant',
                          constant_values=128)  # 128 is considered as zero in uint8 space

  windowed = ops.reshape(
      padded_signal, (ops.shape(padded_signal)[0], -1, window_size))
  windowed = ops.reshape(windowed, (ops.shape(
      windowed)[0]*ops.shape(windowed)[1], window_size))  # batched windowed

  num_win = (ops.shape(signal)[-1]+pad_length)//window_size

  return windowed, pad_length, num_win


def dewindower(signal, pad_len, num_win):
  """
  Dewindow the signal.

  Args:
    signal (tensor: uint): The windowed signal with shape (batch_size * num_win, window_size).
    pad_len (uint): Pad length generated
    num_win (uint): The number of windows.

  Returns:
    dewindowed_signal (tensor: uint): The dewindowed signal with shape (batch_size, signal_length).
  """

  # check dims
  assert signal.ndim == 2, f"signal must have 2 dimensions which are (batch_size * num_win, window_size), but got {ops.shape(signal)}"

  dewindowed = ops.reshape(signal, (-1, num_win*ops.shape(signal)[-1]))

  return dewindowed[:, :-pad_len]


def infer(batch_signal, sample_rate):
  """
  Infer the batch signal end-to-end.

  NOTE: signal_length must be above 3.2 x sample_rate for proper inference.

  Args:
    batch_signal (tensor: float): The ECG signal with shape (batch_size, signal_length).
    sample_rate (uint): The sample rate of the signal.
    pad_mask (tensor: bool): The padding mask with shape (batch_size, signal_length)

  Returns:
    tensor (tensor: uint): The preprocessed tensor with shape (batch_size, signal_length)
  """

  # check dims
  assert batch_signal.ndim == 2, "batch_signal must have 2 dimensions which are (batch_size, signal_length)"

  # check signal_length
  # assert batch_signal.shape[-1] > 3.2 * \
  #     sample_rate, "signal_length must be above 3.2 x sample_rate"

  # if signal_length is less than 3.2 x sample_rate, then pad it
  is_padded = False
  if batch_signal.shape[-1] <= int(3.2 * sample_rate):
    is_padded = True
    pad_length = int(3.2 * sample_rate - batch_signal.shape[-1]) + 1
    batch_signal = ops.pad(batch_signal, ((0, 0), (0, pad_length)),
                           mode='constant', constant_values=0)

  DENOISER_SAMPLE_RATE = 360
  DELINEATOR_SAMPLE_RATE = 80

  origin_siglen = batch_signal.shape[-1]

  # Denoiser resample to 360Hz for denoiser compatibility
  tensor = resample_amplitude(batch_signal, sample_rate, DENOISER_SAMPLE_RATE)
  denoiser = get_denoiser(tensor.shape[-1])
  tensor = denoiser(tensor)
  denoised = tensor[..., 0]  # remove the channel dimension

  # After Denoising must convert from 360Hz to 80Hz for delineator compatibility
  tensor = resample_amplitude(
      denoised, DENOISER_SAMPLE_RATE, DELINEATOR_SAMPLE_RATE)

  tensor = ieee754_to_uint8(tensor)

  # Windowing for delineator model compatibility
  tensor, pad_ref, win_ref = windower(tensor)

  # Delineator
  delineator = get_delineator()

  tensor = delineator.predict(tensor, batch_size=32)
  tensor = ops.argmax(tensor, axis=-1)

  tensor = dewindower(tensor, pad_ref, win_ref)

  # find GCD for upsampling the delineator
  GCD = np.gcd(sample_rate, DELINEATOR_SAMPLE_RATE)  # desired: 360 | from: 80
  a, b = sample_rate//GCD, DELINEATOR_SAMPLE_RATE//GCD  # ratio simplification
  GCD = np.gcd(a, b)

  # Upsampling the delineator (EXPANDED)
  tensor = ops.repeat(tensor, a//GCD, axis=-1)

  # Upsampling the delineator (DESIRED)
  tensor = tensor[:, ::(b//GCD)]  # this is performing downsampling actually

  delineated = ops.cast(tensor, dtype='uint8')

  # resample back denoiser to origin
  denoised = resample_amplitude(denoised, DENOISER_SAMPLE_RATE, sample_rate)

  # truncation
  denoised = ops.convert_to_numpy(denoised)[:, :int(
      sample_rate*(origin_siglen/sample_rate))]
  delineated = ops.convert_to_numpy(
      delineated)[:, :int(sample_rate*origin_siglen/sample_rate)]

  # remove padding
  if is_padded:
    denoised = denoised[:, :-pad_length]
    delineated = delineated[:, :-pad_length]

  return denoised, delineated
