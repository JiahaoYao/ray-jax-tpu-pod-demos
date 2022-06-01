# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MNIST example.

Library file which executes the training and evaluation loop for MNIST.
The data is loaded using tensorflow_datasets.
"""

# See issue #620.
# pytype: disable=wrong-keyword-args

import functools
import time

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow_datasets as tfds
from absl import logging
from flax import jax_utils
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
from flax.training.common_utils import (get_metrics, onehot, shard,
                                        shard_prng_key)
from jax import lax


class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=512)(x)
        x = nn.relu(x)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


@functools.partial(jax.pmap, axis_name="ensemble")
def apply_model(state, images, labels):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        logits = CNN().apply({"params": params}, images)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


@jax.pmap
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


@functools.partial(jax.pmap, axis_name="ensemble")
def train_step(state, images, labels):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        logits = CNN().apply({"params": params}, images)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    grads = lax.pmean(grads, axis_name="ensemble")
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    state = state.apply_gradients(grads=grads)
    loss = lax.pmean(loss, axis_name="ensemble")
    accuracy = lax.pmean(accuracy, axis_name="ensemble")
    return state, loss, accuracy


def train_epoch(state, train_ds, batch_size, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds["image"])
    steps_per_epoch = train_ds_size // batch_size

    epoch_loss = []
    epoch_accuracy = []

    for i in range(steps_per_epoch):
        batch_images = train_ds["image"][i * batch_size : (i + 1) * batch_size]
        batch_labels = train_ds["label"][i * batch_size : (i + 1) * batch_size]
        batch_images = shard(batch_images)
        batch_labels = shard(batch_labels)
        state, loss, accuracy = train_step(state, batch_images, batch_labels)
        epoch_loss.append(jax_utils.unreplicate(loss))
        epoch_accuracy.append(jax_utils.unreplicate(accuracy))
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy


# shard the dataset
import einops


def shard_fn(x):
    return einops.rearrange(x, "(d l) ... -> d l ...", d=jax.process_count())[
        jax.process_index()
    ]


def get_datasets():
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder("mnist")
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split="train", batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split="test", batch_size=-1))
    train_ds["image"] = np.float32(shard_fn(train_ds["image"])) / 255.0
    test_ds["image"] = np.float32(shard_fn(test_ds["image"])) / 255.0
    train_ds["label"] = np.int32(shard_fn(train_ds["label"]))
    test_ds["label"] = np.int32(shard_fn(test_ds["label"]))
    return train_ds, test_ds


@functools.partial(jax.pmap, static_broadcasted_argnums=(1, 2))
def create_train_state(rng, learning_rate, momentum):
    """Creates initial `TrainState`."""
    cnn = CNN()
    params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))["params"]
    tx = optax.sgd(learning_rate, momentum)
    return train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)


def train_and_evaluate(
    config: ml_collections.ConfigDict, workdir: str
) -> train_state.TrainState:
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the tensorboard summaries are written to.

    Returns:
      The train state (which includes the `.params`).
    """
    train_ds, test_ds = get_datasets()
    rng = jax.random.PRNGKey(0)

    rng, init_rng = jax.random.split(rng)
    init_rng = jax_utils.replicate(rng)
    state = create_train_state(init_rng, config.learning_rate, config.momentum)

    for epoch in range(1, config.num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        tic = time.time()
        state, train_loss, train_accuracy = train_epoch(
            state, train_ds, config.batch_size, input_rng
        )
        epoch_time = time.time() - tic
        test_loss = test_accuracy = 0.0
        logging.info(
            "epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, epoch_time: %.3f"
            % (epoch, train_loss, train_accuracy * 100, epoch_time)
        )
        print(
            "epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, epoch_time: %.3f"
            % (epoch, train_loss, train_accuracy * 100, epoch_time)
        )
    return state
