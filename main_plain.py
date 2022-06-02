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

"""Main file for running the MNIST example.

This file is intentionally kept short. The majority of logic is in libraries
than can be easily tested and imported in Colab.
"""

import socket
import time
from pprint import pprint

def release_tpu_lock():
    import subprocess
    subprocess.run("sudo lsof -w /dev/accel0", shell=True)
    subprocess.run("sudo rm -f /tmp/libtpu_lockfile", shell=True)
release_tpu_lock()

import jax
import jax.numpy as jnp
import numpy as np
import ray
import tensorflow as tf
from absl import app, flags, logging
from clu import platform
from jax import pmap
from jax.experimental import maps
from jax.experimental.pjit import PartitionSpec as P
from jax.experimental.pjit import pjit
from jax.lax import pmean
from ml_collections import config_flags

import train


def _sync_devices(x):
    return jax.lax.psum(x, "i")


def sync_devices():
    """Creates a barrier across all hosts/devices."""
    print(
        jax.pmap(_sync_devices, "i")(
            np.ones(jax.local_device_count())
        ).block_until_ready()
    )
    jax.pmap(_sync_devices, "i")(np.ones(jax.local_device_count())).block_until_ready()


workdir = "/tmp/mnist"


import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.learning_rate = 0.1
    config.momentum = 0.9
    config.batch_size = 20000
    config.num_epochs = 10
    return config


config = get_config()


def main():
    print("JAX process: %d / %d" % (jax.process_index(), jax.process_count()))
    print("JAX local devices: %r" % (jax.local_devices()))

    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], "GPU")

    logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    # adjust the batchsize
    config.batch_size //= jax.process_count()

    # Add a note so that we can tell which task is which JAX host.
    # (Depending on the platform task 0 is not guaranteed to be host 0)
    platform.work_unit().set_task_status(
        f"process_index: {jax.process_index()}, "
        f"process_count: {jax.process_count()}"
    )
    platform.work_unit().create_artifact(
        platform.ArtifactType.DIRECTORY, workdir, "workdir"
    )

    train.train_and_evaluate(config, workdir)
    sync_devices()


if __name__ == "__main__":
    main()
