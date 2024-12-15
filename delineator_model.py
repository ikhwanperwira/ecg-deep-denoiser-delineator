# @title Define SinePositionEncoding and TransformerEncoder

import keras
from keras.api import ops
from absl import logging
from keras.api import layers
from keras.api.models import Model


def clone_initializer(initializer):

  # If we get a string or dict, just return as we cannot and should not clone.
  if not isinstance(initializer, keras.initializers.Initializer):
    return initializer
  config = initializer.get_config()
  return initializer.__class__.from_config(config)


def _check_masks_shapes(inputs, padding_mask, attention_mask):
  mask = padding_mask
  if hasattr(inputs, "_keras_mask") and mask is None:
    mask = inputs._keras_mask
  if mask is not None:
    if len(mask.shape) != 2:
      raise ValueError(
          "`padding_mask` should have shape "
          "(batch_size, target_length). "
          f"Received shape `{mask.shape}`."
      )
  if attention_mask is not None:
    if len(attention_mask.shape) != 3:
      raise ValueError(
          "`attention_mask` should have shape "
          "(batch_size, target_length, source_length). "
          f"Received shape `{mask.shape}`."
      )


def merge_padding_and_attention_mask(
    inputs,
    padding_mask,
    attention_mask,
):

  _check_masks_shapes(inputs, padding_mask, attention_mask)
  mask = padding_mask
  if hasattr(inputs, "_keras_mask"):
    if mask is None:
      # If no padding mask is explicitly provided, we look for padding
      # mask from the input data.
      mask = inputs._keras_mask
    else:
      logging.warning(
          "You are explicitly setting `padding_mask` while the `inputs` "
          "have built-in mask, so the built-in mask is ignored."
      )
  if mask is not None:
    # Add an axis for broadcasting, the attention mask should be 2D
    # (not including the batch axis).
    mask = ops.cast(ops.expand_dims(mask, axis=1), "int32")
  if attention_mask is not None:
    attention_mask = ops.cast(attention_mask, "int32")
    if mask is None:
      return attention_mask
    else:
      return ops.minimum(mask, attention_mask)
  return mask


class SinePositionEncoding(keras.layers.Layer):

  def __init__(
      self,
      max_wavelength=10000,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self.max_wavelength = max_wavelength
    self.built = True

  def call(self, inputs, start_index=0):
    shape = ops.shape(inputs)
    seq_length = shape[-2]
    hidden_size = shape[-1]
    positions = ops.arange(seq_length)
    positions = ops.cast(positions + start_index, self.compute_dtype)
    min_freq = ops.cast(1 / self.max_wavelength, dtype=self.compute_dtype)
    timescales = ops.power(
        min_freq,
        ops.cast(2 * (ops.arange(hidden_size) // 2), self.compute_dtype)
        / ops.cast(hidden_size, self.compute_dtype),
    )
    angles = ops.expand_dims(positions, 1) * ops.expand_dims(timescales, 0)
    # even indices are sine, odd are cosine
    cos_mask = ops.cast(ops.arange(hidden_size) % 2, self.compute_dtype)
    sin_mask = 1 - cos_mask
    # embedding shape is [seq_length, hidden_size]
    positional_encodings = (
        ops.sin(angles) * sin_mask + ops.cos(angles) * cos_mask
    )

    return ops.broadcast_to(positional_encodings, shape)

  def get_config(self):
    config = super().get_config()
    config.update(
        {
            "max_wavelength": self.max_wavelength,
        }
    )
    return config

  def compute_output_shape(self, input_shape):
    return input_shape


class TransformerEncoder(keras.layers.Layer):

  def __init__(
      self,
      intermediate_dim,
      num_heads,
      dropout=0,
      activation="relu",
      layer_norm_epsilon=1e-05,
      kernel_initializer="glorot_uniform",
      bias_initializer="zeros",
      normalize_first=False,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self.intermediate_dim = intermediate_dim
    self.num_heads = num_heads
    self.dropout = dropout
    self.activation = keras.activations.get(activation)
    self.layer_norm_epsilon = layer_norm_epsilon
    self.kernel_initializer = keras.initializers.get(kernel_initializer)
    self.bias_initializer = keras.initializers.get(bias_initializer)
    self.normalize_first = normalize_first
    self.supports_masking = True

  def build(self, inputs_shape):
    # Infer the dimension of our hidden feature size from the build shape.
    hidden_dim = inputs_shape[-1]
    # Attention head size is `hidden_dim` over the number of heads.
    key_dim = int(hidden_dim // self.num_heads)
    if key_dim == 0:
      raise ValueError(
          "Attention `key_dim` computed cannot be zero. "
          f"The `hidden_dim` value of {hidden_dim} has to be equal to "
          f"or greater than `num_heads` value of {self.num_heads}."
      )

    # Self attention layers.
    self._self_attention_layer = keras.layers.MultiHeadAttention(
        num_heads=self.num_heads,
        key_dim=key_dim,
        dropout=self.dropout,
        kernel_initializer=clone_initializer(self.kernel_initializer),
        bias_initializer=clone_initializer(self.bias_initializer),
        dtype=self.dtype_policy,
        name="self_attention_layer",
    )
    if hasattr(self._self_attention_layer, "_build_from_signature"):
      self._self_attention_layer._build_from_signature(
          query=inputs_shape,
          value=inputs_shape,
      )
    else:
      self._self_attention_layer.build(
          query_shape=inputs_shape,
          value_shape=inputs_shape,
      )
    self._self_attention_layer_norm = keras.layers.LayerNormalization(
        epsilon=self.layer_norm_epsilon,
        dtype=self.dtype_policy,
        name="self_attention_layer_norm",
    )
    self._self_attention_layer_norm.build(inputs_shape)
    self._self_attention_dropout = keras.layers.Dropout(
        rate=self.dropout,
        dtype=self.dtype_policy,
        name="self_attention_dropout",
    )

    # Feedforward layers.
    self._feedforward_layer_norm = keras.layers.LayerNormalization(
        epsilon=self.layer_norm_epsilon,
        dtype=self.dtype_policy,
        name="feedforward_layer_norm",
    )
    self._feedforward_layer_norm.build(inputs_shape)
    self._feedforward_intermediate_dense = keras.layers.Dense(
        self.intermediate_dim,
        activation=self.activation,
        kernel_initializer=clone_initializer(self.kernel_initializer),
        bias_initializer=clone_initializer(self.bias_initializer),
        dtype=self.dtype_policy,
        name="feedforward_intermediate_dense",
    )
    self._feedforward_intermediate_dense.build(inputs_shape)
    self._feedforward_output_dense = keras.layers.Dense(
        hidden_dim,
        kernel_initializer=clone_initializer(self.kernel_initializer),
        bias_initializer=clone_initializer(self.bias_initializer),
        dtype=self.dtype_policy,
        name="feedforward_output_dense",
    )
    intermediate_shape = list(inputs_shape)
    intermediate_shape[-1] = self.intermediate_dim
    self._feedforward_output_dense.build(tuple(intermediate_shape))
    self._feedforward_dropout = keras.layers.Dropout(
        rate=self.dropout,
        dtype=self.dtype_policy,
        name="feedforward_dropout",
    )
    self.built = True

  def call(
      self,
      inputs,
      padding_mask=None,
      attention_mask=None,
      training=None,
      return_attention_scores=False,
  ):

    x = inputs  # Intermediate result.

    # Compute self attention mask.
    self_attention_mask = merge_padding_and_attention_mask(
        inputs, padding_mask, attention_mask
    )

    # Self attention block.
    residual = x
    if self.normalize_first:
      x = self._self_attention_layer_norm(x)

    if return_attention_scores:
      x, attention_scores = self._self_attention_layer(
          query=x,
          value=x,
          attention_mask=self_attention_mask,
          return_attention_scores=return_attention_scores,
          training=training,
      )
      return x, attention_scores
    else:
      x = self._self_attention_layer(
          query=x,
          value=x,
          attention_mask=self_attention_mask,
          training=training,
      )

    x = self._self_attention_dropout(x, training=training)
    x = x + residual
    if not self.normalize_first:
      x = self._self_attention_layer_norm(x)

    # Feedforward block.
    residual = x
    if self.normalize_first:
      x = self._feedforward_layer_norm(x)
    x = self._feedforward_intermediate_dense(x)
    x = self._feedforward_output_dense(x)
    x = self._feedforward_dropout(x, training=training)
    x = x + residual
    if not self.normalize_first:
      x = self._feedforward_layer_norm(x)

    if return_attention_scores:
      return x, attention_scores

    return x

  def get_config(self):
    config = super().get_config()
    config.update(
        {
            "intermediate_dim": self.intermediate_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "activation": keras.activations.serialize(self.activation),
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "kernel_initializer": keras.initializers.serialize(
                self.kernel_initializer
            ),
            "bias_initializer": keras.initializers.serialize(
                self.bias_initializer
            ),
            "normalize_first": self.normalize_first,
        }
    )
    return config

  def compute_output_shape(self, inputs_shape):
    return inputs_shape


# Register the layers manually
keras.utils.get_custom_objects().update({
    "SinePositionEncoding": SinePositionEncoding,
    "TransformerEncoder": TransformerEncoder,
})


# @title Load the model and weights
def get_delineator():
  encoder_input = layers.Input(
      shape=(256,), name="encoder_input", dtype='uint8')

  # embeddings
  token_embedding = layers.Embedding(input_dim=256, output_dim=128, name='token_embedding')(
      encoder_input)  # Input: Token Size, Output: Embed Dim, Token has range [0,255), 255 is excluded
  position_encoding = SinePositionEncoding(name='sine_pe')(token_embedding)
  embedding = layers.Add(name='word_embedding')(
      [token_embedding, position_encoding])

  # transformer encoder
  encoder_output = TransformerEncoder(
      intermediate_dim=128*2, num_heads=4, dropout=0.2, name='encoder_output')(inputs=embedding)

  # Output layer for vocabulary size of 4
  output_predictions = layers.Dense(
      units=4, activation=None, name='linear_output')(encoder_output)

  # Final model
  model = Model(encoder_input, output_predictions, name="transformer_encoder")

  # Load weights
  model.load_weights(
      'delineator_noatrial_d128_80hz_256winsize_latest.weights.h5')

  return model
