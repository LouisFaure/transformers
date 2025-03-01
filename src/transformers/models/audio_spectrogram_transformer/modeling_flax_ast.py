# coding=utf-8
# Copyright 2023 The Hugging Face Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Flax Audio Spectrogram Transformer (AST) model."""

from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict

from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPooling, FlaxSequenceClassifierOutput
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_audio_spectrogram_transformer import ASTConfig


AST_START_DOCSTRING = r"""
This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

This model is also a
[flax.linen.Module](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html) subclass. Use it as
a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and
behavior.

Finally, this model supports inherent JAX features such as:

- [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
- [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
- [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
- [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

Parameters:
    config ([`ASTConfig`]): Model configuration class with all the parameters of the model.
        Initializing with a config file does not load the weights associated with the model, only the
        configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
    dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
        The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
        `jax.numpy.bfloat16` (on TPUs).

        This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
        specified all the computation will be performed with the given `dtype`.

        **Note that this only specifies the dtype of the computation and does not influence the dtype of model
        parameters.**

        If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
        [`~FlaxPreTrainedModel.to_bf16`].
"""

AST_INPUTS_DOCSTRING = r"""
Args:
    input_values (`numpy.ndarray` of shape `(batch_size, max_length, num_mel_bins)`):
        Float values mel features extracted from the raw audio waveform. Raw audio waveform can be obtained by
        loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via
        the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
        [`AutoFeatureExtractor`] should be used for extracting the mel features, padding and conversion into a
        tensor of type `numpy.ndarray`. See [`~ASTFeatureExtractor.__call__`]

    output_attentions (`bool`, *optional*):
        Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
        tensors for more detail.
    output_hidden_states (`bool`, *optional*):
        Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
        more detail.
    return_dict (`bool`, *optional*):
        Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class FlaxASTPatchEmbeddings(nn.Module):
    config: ASTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        patch_size = self.config.patch_size
        frequency_stride = self.config.frequency_stride
        time_stride = self.config.time_stride

        # Get the shape
        frequency_out_dimension = (self.config.num_mel_bins - patch_size) // frequency_stride + 1
        time_out_dimension = (self.config.max_length - patch_size) // time_stride + 1
        self.num_patches = frequency_out_dimension * time_out_dimension

        self.projection = nn.Conv(
            self.config.hidden_size,
            kernel_size=(patch_size, patch_size),
            strides=(frequency_stride, time_stride),
            padding="VALID",
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
        )

    def __call__(self, input_values):
        # Add a dummy channel dimension for the Conv
        input_values = jnp.expand_dims(input_values, axis=-1)
        # Transpose for compatibility with the convolution operation
        input_values = jnp.transpose(input_values, (0, 2, 1, 3))
        embeddings = self.projection(input_values)
        batch_size, _, _, channels = embeddings.shape
        return jnp.reshape(embeddings, (batch_size, -1, channels))


class FlaxASTEmbeddings(nn.Module):
    """
    Construct the CLS token, distillation token, position and patch embeddings.
    """

    config: ASTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.cls_token = self.param(
            "cls_token",
            jax.nn.initializers.variance_scaling(self.config.initializer_range**2, "fan_in", "truncated_normal"),
            (1, 1, self.config.hidden_size),
        )
        self.distillation_token = self.param(
            "distillation_token",
            jax.nn.initializers.variance_scaling(self.config.initializer_range**2, "fan_in", "truncated_normal"),
            (1, 1, self.config.hidden_size),
        )
        
        self.patch_embeddings = FlaxASTPatchEmbeddings(self.config, dtype=self.dtype)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = self.param(
            "position_embeddings",
            jax.nn.initializers.variance_scaling(self.config.initializer_range**2, "fan_in", "truncated_normal"),
            (1, num_patches + 2, self.config.hidden_size),  # +2 for cls and distillation tokens
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, input_values, deterministic=True):
        batch_size = input_values.shape[0]

        embeddings = self.patch_embeddings(input_values)

        cls_tokens = jnp.broadcast_to(self.cls_token, (batch_size, 1, self.config.hidden_size))
        distillation_tokens = jnp.broadcast_to(self.distillation_token, (batch_size, 1, self.config.hidden_size))
        embeddings = jnp.concatenate((cls_tokens, distillation_tokens, embeddings), axis=1)
        
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings, deterministic=deterministic)
        return embeddings


class FlaxASTSelfAttention(nn.Module):
    config: ASTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                f"`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads`:"
                f" {self.config.num_attention_heads}"
            )

        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
            use_bias=self.config.qkv_bias,
        )
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
            use_bias=self.config.qkv_bias,
        )
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
            use_bias=self.config.qkv_bias,
        )

    def __call__(self, hidden_states, deterministic: bool = True, output_attentions: bool = False):
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        query_states = self.query(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        value_states = self.value(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        key_states = self.key(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )

        dropout_rng = None
        if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
            dropout_rng = self.make_rng("dropout")

        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_probs_dropout_prob,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


class FlaxASTSelfOutput(nn.Module):
    config: ASTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


class FlaxASTAttention(nn.Module):
    config: ASTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.attention = FlaxASTSelfAttention(self.config, dtype=self.dtype)
        self.output = FlaxASTSelfOutput(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True, output_attentions: bool = False):
        attn_outputs = self.attention(hidden_states, deterministic=deterministic, output_attentions=output_attentions)
        attn_output = attn_outputs[0]
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_outputs[1],)

        return outputs


class FlaxASTIntermediate(nn.Module):
    config: ASTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            dtype=self.dtype,
        )
        self.activation = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class FlaxASTOutput(nn.Module):
    config: ASTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, attention_output, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = hidden_states + attention_output
        return hidden_states


class FlaxASTLayer(nn.Module):
    config: ASTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.attention = FlaxASTAttention(self.config, dtype=self.dtype)
        self.intermediate = FlaxASTIntermediate(self.config, dtype=self.dtype)
        self.output = FlaxASTOutput(self.config, dtype=self.dtype)
        self.layernorm_before = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.layernorm_after = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic: bool = True, output_attentions: bool = False):
        attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in AST, layernorm is applied before self-attention
            deterministic=deterministic,
            output_attentions=output_attentions,
        )

        attention_output = attention_outputs[0]

        # first residual connection
        attention_output = attention_output + hidden_states

        # in AST, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(attention_output)
        layer_output = self.intermediate(layer_output)
        
        # second residual connection is done here
        layer_output = self.output(layer_output, attention_output, deterministic=deterministic)

        outputs = (layer_output,)

        if output_attentions:
            outputs += (attention_outputs[1],)
        return outputs


class FlaxASTLayerCollection(nn.Module):
    config: ASTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layers = [
            FlaxASTLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(hidden_states, deterministic=deterministic, output_attentions=output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class FlaxASTEncoder(nn.Module):
    config: ASTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layer = FlaxASTLayerCollection(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return self.layer(
            hidden_states,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class FlaxASTPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ASTConfig
    base_model_prefix = "audio_spectrogram_transformer"
    main_input_name = "input_values"
    module_class: nn.Module = None

    def __init__(
        self,
        config: ASTConfig,
        input_shape=None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        if input_shape is None:
            input_shape = (1, config.max_length, config.num_mel_bins)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        input_values = jnp.zeros(input_shape, dtype=self.dtype)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(rngs, input_values, return_dict=False)["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    @add_start_docstrings_to_model_forward(AST_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_values,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_values, dtype=jnp.float32),
            not train,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )


class FlaxASTModule(nn.Module):
    config: ASTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.embeddings = FlaxASTEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxASTEncoder(self.config, dtype=self.dtype)
        self.layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(
        self,
        input_values,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        hidden_states = self.embeddings(input_values, deterministic=deterministic)

        outputs = self.encoder(
            hidden_states,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.layernorm(hidden_states)

        # Average the outputs from the CLS and distillation tokens as pooler output
        pooled_output = (hidden_states[:, 0] + hidden_states[:, 1]) / 2

        if not return_dict:
            return (hidden_states, pooled_output) + outputs[1:]

        return FlaxBaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    "The bare AST Model transformer outputting raw hidden-states without any specific head on top.",
    AST_START_DOCSTRING,
)
class FlaxASTModel(FlaxASTPreTrainedModel):
    module_class = FlaxASTModule


FLAX_AST_MODEL_DOCSTRING = """
Returns:

Examples:

```python
>>> from transformers import AutoFeatureExtractor, FlaxASTModel
>>> from datasets import load_dataset
>>> import numpy as np

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
>>> model = FlaxASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

>>> # Audio preprocessing
>>> inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="np")
>>> input_values = inputs.input_values

>>> outputs = model(input_values)
>>> last_hidden_states = outputs.last_hidden_state
```
"""

overwrite_call_docstring(FlaxASTModel, FLAX_AST_MODEL_DOCSTRING)
append_replace_return_docstrings(FlaxASTModel, output_type=FlaxBaseModelOutputWithPooling, config_class=ASTConfig)


class FlaxASTForAudioClassificationModule(nn.Module):
    config: ASTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.audio_spectrogram_transformer = FlaxASTModule(config=self.config, dtype=self.dtype)
        
        # Classifier head - using LayerNorm followed by Dense as in PyTorch implementation
        self.classifier = nn.Sequential([
            nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype),
            nn.Dense(
                self.config.num_labels,
                dtype=self.dtype,
                kernel_init=jax.nn.initializers.variance_scaling(
                    self.config.initializer_range**2, "fan_in", "truncated_normal"
                ),
            )
        ])

    def __call__(
        self,
        input_values=None,
        deterministic: bool = True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.audio_spectrogram_transformer(
            input_values,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs.pooler_output if return_dict else outputs[1]
        logits = self.classifier(pooled_output)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return output

        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    AST Model transformer with an audio classification head on top (a linear layer on top of the pooled output) for
    tasks like AudioSet, Speech Commands v2.
    """,
    AST_START_DOCSTRING,
)
class FlaxASTForAudioClassification(FlaxASTPreTrainedModel):
    module_class = FlaxASTForAudioClassificationModule


FLAX_AST_AUDIO_CLASSIFICATION_DOCSTRING = """
Returns:

Example:

```python
>>> from transformers import AutoFeatureExtractor, FlaxASTForAudioClassification
>>> from datasets import load_dataset
>>> import jax.numpy as jnp
>>> import numpy as np

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
>>> model = FlaxASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

>>> # Audio preprocessing
>>> inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="np")
>>> input_values = inputs.input_values

>>> outputs = model(input_values)
>>> logits = outputs.logits

>>> predicted_class_idx = jnp.argmax(logits, axis=-1)[0]
>>> print("Predicted class:", model.config.id2label[predicted_class_idx.item()])
```
"""

overwrite_call_docstring(FlaxASTForAudioClassification, FLAX_AST_AUDIO_CLASSIFICATION_DOCSTRING)
append_replace_return_docstrings(
    FlaxASTForAudioClassification, output_type=FlaxSequenceClassifierOutput, config_class=ASTConfig
)


__all__ = ["FlaxASTForAudioClassification", "FlaxASTModel", "FlaxASTPreTrainedModel"]
