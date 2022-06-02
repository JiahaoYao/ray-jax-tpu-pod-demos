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

import numpy as np
import ray
from absl import app, flags, logging
from clu import platform
from ml_collections import config_flags

ray.init("auto")

host_ip = socket.gethostbyname(socket.gethostname())

ip_resources = [x for x in ray.cluster_resources() if "node:" in x]
ip_resources_ = [
    x.replace("node:", "") for x in ray.cluster_resources() if "node:" in x
]

# host node should be zero
ip_resources_ = sorted(ip_resources_, key=lambda x: host_ip not in x)
ip2hostid_dict = dict(zip(ip_resources_, range(len(ip_resources_))))

print(ip2hostid_dict)
print(host_ip)

# utils
def run_job_on_ray(func):
    results = [
        func.options(resources={ip_resource: 0.01}).remote()
        for ip_resource in ip_resources
    ]
    ray.get(results)


workdir = "/tmp/mnist"


import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.learning_rate = 0.1
    config.momentum = 0.9
    config.batch_size = 8192
    config.num_epochs = 10
    return config


config = get_config()


@ray.remote(resources={"TPU": 1})
def main():

    import jax
    import tensorflow as tf

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
        jax.pmap(_sync_devices, "i")(
            np.ones(jax.local_device_count())
        ).block_until_ready()

    print("JAX process: %d / %d" % (jax.process_index(), jax.process_count()))
    print("JAX local devices: %r" % (jax.local_devices()))

    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], "GPU")

    logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    # adjust the batchsize
    config.batch_size //= jax.device_count()

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
    run_job_on_ray(main)
