# ray-jax-tpu-pod-demos
Demos starting ray cluster on tpu pod




## MNIST classification

Trains a simple fully connected network on the MNIST dataset.

(Adopted from https://github.com/google/flax/tree/main/examples/mnist with the following modifications to better demonstrate the training speeedup on this small network.)


### setup tpu pod 

```
bash tpu_launcher.sh
```

### test the jax tpu

```python 
gcloud alpha compute tpus tpu-vm ssh jax-trainer-mnist-tpu-pod --zone=us-central1-a --command "sudo python3 -c \"import jax; print(jax.device_count(), jax.local_device_count())\"" --worker all
```




<details>
<summary>Expected Output</summary>

  
> ```
> SSH key found in project metadata; not updating instance.
> SSH: Attempting to connect to worker 0...
> SSH: Attempting to connect to worker 1...
> SSH: Attempting to connect to worker 2...
> SSH: Attempting to connect to worker 3...
> Warning: Permanently added 'tpu.1660999404634765869-2-4zwxvo' (ECDSA) to the list of known hosts.
> Warning: Permanently added 'tpu.1660999404634765869-0-eawa07' (ECDSA) to the list of known hosts.
> Warning: Permanently added 'tpu.1660999404634765869-3-6spoaq' (ECDSA) to the list of known hosts.
> Warning: Permanently added 'tpu.1660999404634765869-1-snwxxp' (ECDSA) to the list of known hosts.
> 32 8
> 32 8
> 32 8
> 32 8
> ```

</details>
  
#### Warning ⚠️

When one sees the following error messages, 
```python
I0000 00:00:1654201087.092808   11652 tpu_initializer_helper.cc:116] libtpu.so is already in use by process with pid 9665. Not attempting to load libtpu.so in this process.
WARNING: Logging before InitGoogle() is written to STDERR
I0000 00:00:1654201087.087846   11709 tpu_initializer_helper.cc:116] libtpu.so is already in use by process with pid 9708. Not attempting to load libtpu.so in this process.
WARNING: Logging before InitGoogle() is written to STDERR
I0000 00:00:1654201087.096611   11906 tpu_initializer_helper.cc:116] libtpu.so is already in use by process with pid 9904. Not attempting to load libtpu.so in this process.
WARNING: Logging before InitGoogle() is written to STDERR
I0000 00:00:1654201087.307681   24041 tpu_initializer_helper.cc:116] libtpu.so is already in use by process with pid 18494. Not attempting to load libtpu.so in this process.
```

the tpu might be locked (see: https://github.com/google/jax/issues/10192), run the following commands to remove the lock. 

```python
gcloud alpha compute tpus tpu-vm ssh jax-trainer-mnist-tpu-pod --zone=us-central1-a --command "sudo lsof -w /dev/accel0" --worker all
gcloud alpha compute tpus tpu-vm ssh jax-trainer-mnist-tpu-pod --zone=us-central1-a --command "sudo rm -f /tmp/libtpu_lockfile" --worker=all
```

  

### upload the files onto the tpu 

```python
gcloud alpha compute tpus tpu-vm  scp --recurse [CHANGE_HERE_YOUR_PATH]/ray-jax-tpu-pod-demos jax-trainer-mnist-tpu-pod: --zone=us-central1-a --worker all
```

### How to run


- run without using the ray cluster 

```python
gcloud alpha compute tpus tpu-vm ssh jax-trainer-mnist-tpu-pod --zone=us-central1-a --command "cd ~/ray-jax-tpu-pod-demos ; sudo python3 main_plain.py" --worker all
```


<details>
<summary>Failed example</summary>

```python
(base) ~/workspace/Github/ray-jax-tpu-pod-demos gcloud alpha compute tpus tpu-vm ssh jax-trainer-mnist-tpu-pod --zone=us-central1-a --command "cd ~/ray-jax-tpu-pod-demos ; sudo python3 main_plain.py" --worker all
SSH key found in project metadata; not updating instance.
SSH: Attempting to connect to worker 0...
SSH: Attempting to connect to worker 1...
SSH: Attempting to connect to worker 2...
SSH: Attempting to connect to worker 3...
2022-06-02 05:25:54.754331: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2022-06-02 05:25:54.920657: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2022-06-02 05:25:55.003941: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2022-06-02 05:25:55.003305: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
2022-06-02 05:25:56.573269: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
2022-06-02 05:25:56.573307: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
JAX process: 0 / 1
JAX local devices: [CpuDevice(id=0)]
Downloading and preparing dataset 11.06 MiB (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /root/tensorflow_datasets/mnist/3.0.1...
Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  8.72 file/s]2022-06-02 05:26:39.302923: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
2022-06-02 05:26:39.302962: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2022-06-02 05:26:39.348620: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
2022-06-02 05:26:39.348660: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2022-06-02 05:26:39.375824: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
2022-06-02 05:26:39.375863: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
JAX process: 3 / 4
JAX local devices: [TpuDevice(id=20, process_index=3, coords=(2,2,0), core_on_chip=0), TpuDevice(id=21, process_index=3, coords=(2,2,0), core_on_chip=1), TpuDevice(id=22, process_index=3, coords=(3,2,0), core_on_chip=0), TpuDevice(id=23, process_index=3, coords=(3,2,0), core_on_chip=1), TpuDevice(id=28, process_index=3, coords=(2,3,0), core_on_chip=0), TpuDevice(id=29, process_index=3, coords=(2,3,0), core_on_chip=1), TpuDevice(id=30, process_index=3, coords=(3,3,0), core_on_chip=0), TpuDevice(id=31, process_index=3, coords=(3,3,0), core_on_chip=1)]
Downloading and preparing dataset 11.06 MiB (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /root/tensorflow_datasets/mnist/3.0.1...
Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 15.92 file/s]JAX process: 2 / 4
JAX local devices: [TpuDevice(id=16, process_index=2, coords=(0,2,0), core_on_chip=0), TpuDevice(id=17, process_index=2, coords=(0,2,0), core_on_chip=1), TpuDevice(id=18, process_index=2, coords=(1,2,0), core_on_chip=0), TpuDevice(id=19, process_index=2, coords=(1,2,0), core_on_chip=1), TpuDevice(id=24, process_index=2, coords=(0,3,0), core_on_chip=0), TpuDevice(id=25, process_index=2, coords=(0,3,0), core_on_chip=1), TpuDevice(id=26, process_index=2, coords=(1,3,0), core_on_chip=0), TpuDevice(id=27, process_index=2, coords=(1,3,0), core_on_chip=1)]
Downloading and preparing dataset 11.06 MiB (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /root/tensorflow_datasets/mnist/3.0.1...
Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 20.34 file/s]JAX process: 1 / 4
JAX local devices: [TpuDevice(id=4, process_index=1, coords=(2,0,0), core_on_chip=0), TpuDevice(id=5, process_index=1, coords=(2,0,0), core_on_chip=1), TpuDevice(id=6, process_index=1, coords=(3,0,0), core_on_chip=0), TpuDevice(id=7, process_index=1, coords=(3,0,0), core_on_chip=1), TpuDevice(id=12, process_index=1, coords=(2,1,0), core_on_chip=0), TpuDevice(id=13, process_index=1, coords=(2,1,0), core_on_chip=1), TpuDevice(id=14, process_index=1, coords=(3,1,0), core_on_chip=0), TpuDevice(id=15, process_index=1, coords=(3,1,0), core_on_chip=1)]
Downloading and preparing dataset 11.06 MiB (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /root/tensorflow_datasets/mnist/3.0.1...
Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  9.27 file/s]Dataset mnist downloaded and prepared to /root/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.
epoch:  1, train_loss: 2.2453, train_accuracy: 17.77, epoch_time: 0.978
epoch:  2, train_loss: 1.9814, train_accuracy: 58.84, epoch_time: 0.252
epoch:  3, train_loss: 1.4973, train_accuracy: 73.63, epoch_time: 0.247
epoch:  4, train_loss: 0.9374, train_accuracy: 79.29, epoch_time: 0.259
epoch:  5, train_loss: 0.6238, train_accuracy: 82.46, epoch_time: 0.246
epoch:  6, train_loss: 0.5096, train_accuracy: 84.82, epoch_time: 0.250
epoch:  7, train_loss: 0.4630, train_accuracy: 86.73, epoch_time: 0.251
epoch:  8, train_loss: 0.4369, train_accuracy: 87.92, epoch_time: 0.257
epoch:  9, train_loss: 0.4143, train_accuracy: 88.67, epoch_time: 0.247
epoch: 10, train_loss: 0.3918, train_accuracy: 89.22, epoch_time: 0.243
[1.]

/usr/local/lib/python3.8/dist-packages/jax/_src/tree_util.py:188: FutureWarning: jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() instead as a drop-in replacement.
  warnings.warn('jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() '
  
/usr/local/lib/python3.8/dist-packages/jax/_src/tree_util.py:188: FutureWarning: jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() instead as a drop-in replacement.
  warnings.warn('jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() '
Dataset mnist downloaded and prepared to /root/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.
epoch:  1, train_loss: 2.2455, train_accuracy: 18.00, epoch_time: 0.716
epoch:  2, train_loss: 1.9817, train_accuracy: 58.91, epoch_time: 0.023
epoch:  3, train_loss: 1.4962, train_accuracy: 73.79, epoch_time: 0.013
epoch:  4, train_loss: 0.9364, train_accuracy: 79.30, epoch_time: 0.013
epoch:  5, train_loss: 0.6218, train_accuracy: 82.56, epoch_time: 0.013
epoch:  6, train_loss: 0.5105, train_accuracy: 84.74, epoch_time: 0.013
epoch:  7, train_loss: 0.4633, train_accuracy: 86.75, epoch_time: 0.013
epoch:  8, train_loss: 0.4391, train_accuracy: 87.88, epoch_time: 0.012
epoch:  9, train_loss: 0.4181, train_accuracy: 88.52, epoch_time: 0.013
epoch: 10, train_loss: 0.3930, train_accuracy: 89.16, epoch_time: 0.013
[32. 32. 32. 32. 32. 32. 32. 32.]

/usr/local/lib/python3.8/dist-packages/jax/_src/tree_util.py:188: FutureWarning: jax.tree_util.tree_multimap() is deprecated. PleDataset mnist downloaded and prepared to /root/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.
epoch:  1, train_loss: 2.2455, train_accuracy: 18.00, epoch_time: 0.732
epoch:  2, train_loss: 1.9817, train_accuracy: 58.91, epoch_time: 0.024
epoch:  3, train_loss: 1.4962, train_accuracy: 73.79, epoch_time: 0.013
epoch:  4, train_loss: 0.9364, train_accuracy: 79.30, epoch_time: 0.013
epoch:  5, train_loss: 0.6218, train_accuracy: 82.56, epoch_time: 0.013
epoch:  6, train_loss: 0.5105, train_accuracy: 84.74, epoch_time: 0.013
epoch:  7, train_loss: 0.4633, train_accuracy: 86.75, epoch_time: 0.013
epoch:  8, train_loss: 0.4391, train_accuracy: 87.88, epoch_time: 0.013
epoch:  9, train_loss: 0.4181, train_accuracy: 88.52, epoch_time: 0.013
epoch: 10, train_loss: 0.3930, train_accuracy: 89.16, epoch_time: 0.013
[32. 32. 32. 32. 32. 32. 32. 32.]

/usr/local/lib/python3.8/dist-packages/jax/_src/tree_util.py:188: FutureWarning: jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() instead as a drop-in replacement.
  warnings.warn('jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() '
ase use jax.tree_util.tree_map() instead as a drop-in replacement.
  warnings.warn('jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() '
Dataset mnist downloaded and prepared to /root/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.
epoch:  1, train_loss: 2.2455, train_accuracy: 18.00, epoch_time: 0.631
epoch:  2, train_loss: 1.9817, train_accuracy: 58.91, epoch_time: 0.024
epoch:  3, train_loss: 1.4962, train_accuracy: 73.79, epoch_time: 0.014
epoch:  4, train_loss: 0.9364, train_accuracy: 79.30, epoch_time: 0.012
epoch:  5, train_loss: 0.6218, train_accuracy: 82.56, epoch_time: 0.012
epoch:  6, train_loss: 0.5105, train_accuracy: 84.74, epoch_time: 0.013
epoch:  7, train_loss: 0.4633, train_accuracy: 86.75, epoch_time: 0.013
epoch:  8, train_loss: 0.4391, train_accuracy: 87.88, epoch_time: 0.012
epoch:  9, train_loss: 0.4181, train_accuracy: 88.52, epoch_time: 0.013
epoch: 10, train_loss: 0.3930, train_accuracy: 89.16, epoch_time: 0.013
[32. 32. 32. 32. 32. 32. 32. 32.]

/usr/local/lib/python3.8/dist-packages/jax/_src/tree_util.py:188: FutureWarning: jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() instead as a drop-in replacement.
  warnings.warn('jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() '
```  
One vm does not start properly, I would run the following code and rerun the job again. (<span style="color:red">the reason might be some process blocked.</span>)

```python
gcloud alpha compute tpus tpu-vm ssh jax-trainer-mnist-tpu-pod --zone=us-central1-a --command "sudo lsof -w /dev/accel0" --worker all
gcloud alpha compute tpus tpu-vm ssh jax-trainer-mnist-tpu-pod --zone=us-central1-a --command "sudo rm -f /tmp/libtpu_lockfile" --worker=all
```
  
</details>


<details>
<summary>Success example</summary>

```python
(base) ~/workspace/Github/ray-jax-tpu-pod-demos gcloud alpha compute tpus tpu-vm ssh jax-trainer-mnist-tpu-pod --zone=us-central1-a --command "cd ~/ray-jax-tpu-pod-demos ; sudo python3 main_plain.py" --worker all
SSH key found in project metadata; not updating instance.
SSH: Attempting to connect to worker 0...
SSH: Attempting to connect to worker 1...
SSH: Attempting to connect to worker 2...
SSH: Attempting to connect to worker 3...
2022-06-02 05:30:52.045017: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2022-06-02 05:30:52.072523: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2022-06-02 05:30:52.071341: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2022-06-02 05:30:52.074232: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2022-06-02 05:31:10.241255: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
2022-06-02 05:31:10.241296: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2022-06-02 05:31:10.243680: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
2022-06-02 05:31:10.243712: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2022-06-02 05:31:10.259084: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
2022-06-02 05:31:10.259122: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2022-06-02 05:31:10.323089: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
2022-06-02 05:31:10.323132: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
JAX process: 1 / 4
JAX local devices: [TpuDevice(id=4, process_index=1, coords=(2,0,0), core_on_chip=0), TpuDevice(id=5, process_index=1, coords=(2,0,0), core_on_chip=1), TpuDevice(id=6, process_index=1, coords=(3,0,0), core_on_chip=0), TpuDevice(id=7, process_index=1, coords=(3,0,0), core_on_chip=1), TpuDevice(id=12, process_index=1, coords=(2,1,0), core_on_chip=0), TpuDevice(id=13, process_index=1, coords=(2,1,0), core_on_chip=1), TpuDevice(id=14, process_index=1, coords=(3,1,0), core_on_chip=0), TpuDevice(id=15, process_index=1, coords=(3,1,0), core_on_chip=1)]
epoch:  1, train_loss: 2.2455, train_accuracy: 18.00, epoch_time: 0.476
epoch:  2, train_loss: 1.9817, train_accuracy: 58.91, epoch_time: 0.014
epoch:  3, train_loss: 1.4962, train_accuracy: 73.79, epoch_time: 0.014
epoch:  4, train_loss: 0.9364, train_accuracy: 79.30, epoch_time: 0.014
epoch:  5, train_loss: 0.6218, train_accuracy: 82.56, epoch_time: 0.014
epoch:  6, train_loss: 0.5105, train_accuracy: 84.74, epoch_time: 0.015
epoch:  7, trJAX process: 0 / 4
JAX local devices: [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0), TpuDevice(id=1, process_index=0, coords=(0,0,0), core_on_chip=1), TpuDevice(id=2, process_index=0, coords=(1,0,0), core_on_chip=0), TpuDevice(id=3, process_index=0, coords=(1,0,0), core_on_chip=1), TpuDevice(id=8, process_index=0, coords=(0,1,0), core_on_chip=0), TpuDevice(id=9, process_index=0, coords=(0,1,0), core_on_chip=1), TpuDevice(id=10, process_index=0, coords=(1,1,0), core_on_chip=0), TpuDevice(id=11, process_index=0, coords=(1,1,0), core_on_chip=1)]
epoch:  1, train_loss: 2.2455, train_accuracy: 18.00, epoch_time: 0.705
epoch:  2, train_loss: 1.9817, train_accuracy: 58.91, epoch_time: 0.014
epoch:  3, train_loss: 1.4962, train_accuracy: 73.79, epoch_time: 0.014
epoch:  4, train_loss: 0.9364, train_accuracy: 79.30, epoch_time: 0.014
epoch:  5, train_loss: 0.6218, train_accuracy: 82.56, epoch_time: 0.014
epoch:  6, train_loss: 0.5105, train_accuracy: 84.74, epoch_time: 0.015
epoch:  7, train_loss: 0.4633, train_accuracy: 86.75, epoch_time: 0.015
epoch:  8, train_loss: 0.4391, train_accuracy: 87.88, epoch_time: 0.015
epoch:  9, train_loss: 0.4181, train_accuracy: 88.52, epoch_time: 0.016
epoch: 10, train_loss: 0.3930, train_accuracy: 89.16, epoch_time: 0.016
[32. 32. 32. 32. 32. 32. 32. 32.]
ain_loss: 0.4633, train_accuracy: 86.75, epoch_time: 0.015
epoch:  8, train_loss: 0.4391, train_accuracy: 87.88, epoch_time: 0.016
epoch:  9, train_loss: 0.4181, train_accuracy: 88.52, epoch_time: 0.016
epoch: 10, train_loss: 0.3930, train_accuracy: 89.16, epoch_time: 0.016
[32. 32. 32. 32. 32. 32. 32. 32.]
/usr/local/lib/python3.8/dist-packages/jax/_src/tree_util.py:188: FutureWarning: jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() instead as a drop-in replacement.
  warnings.warn('jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() '
/usr/local/lib/python3.8/dist-packages/jax/_src/tree_util.py:188: FutureWarning: jax.tree_util.tree_JAX process: 3 / 4
JAX local devices: [TpuDevice(id=20, process_index=3, coords=(2,2,0), core_on_chip=0), TpuDevice(id=21, process_index=3, coords=(2,2,0), core_on_chip=1), TpuDevice(id=22, process_index=3, coords=(3,2,0), core_on_chip=0), TpuDevice(id=23, process_index=3, coords=(3,2,0), core_on_chip=1), TpuDevice(id=28, process_index=3, coords=(2,3,0), core_on_chip=0), TpuDevice(id=29, process_index=3, coords=(2,3,0), core_on_chip=1), TpuDevice(id=30, process_index=3, coords=(3,3,0), core_on_chip=0), TpuDevice(id=31, process_index=3, coords=(3,3,0), core_on_chip=1)]
epoch:  1, train_loss: 2.2455, train_accuracy: 18.00, epoch_time: 0.604
epoch:  2, train_loss: 1.9817, train_accuracy: 58.91, epoch_time: 0.014
epoch:  3, train_loss: 1.4962, train_accuracy: 73.79, epoch_time: 0.014
epoch:  4, train_loss: 0.9364, train_accuracy: 79.30, epoch_time: 0.014
epoch:  5, train_loss: 0.6218, train_accuracy: 82.56, epoch_time: 0.014
epoch:  6, train_loss: 0.5105, train_accuracy: 84.74, epoch_time: 0.015
epoch:  7, train_loss: 0.4633, train_accuracy: 86.75, epoch_time: 0.015
epoch:  8, train_loss: 0.4391, train_accuracy: 87.88, epoch_time: 0.015
epoch:  9, train_loss: 0.4181, train_accuracy: 88.52, epoch_time: 0.016
epoch: 10, train_loss: 0.3930, train_accuracy: 89.16, epoch_time: 0.016
[32. 32. 32. 32. 32. 32. 32. 32.]
multimap() is deprecated. Please use jax.tree_util.tree_map() instead as a drop-in replacement.
  warnings.warn('jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() '
/usr/local/lib/python3.8/dist-packages/jax/_src/tree_util.py:188: FutureWarning: jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() instead as a drop-in replacement.
  warnings.warn('jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() '
/usr/local/lib/python3.8/dist-packages/jax/_src/tree_util.py:188: FutureWarning: jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() instead as a drop-in replacement.
  warnings.waJAX process: 2 / 4
JAX local devices: [TpuDevice(id=16, process_index=2, coords=(0,2,0), core_on_chip=0), TpuDevice(id=17, process_index=2, coords=(0,2,0), core_on_chip=1), TpuDevice(id=18, process_index=2, coords=(1,2,0), core_on_chip=0), TpuDevice(id=19, process_index=2, coords=(1,2,0), core_on_chip=1), TpuDevice(id=24, process_index=2, coords=(0,3,0), core_on_chip=0), TpuDevice(id=25, process_index=2, coords=(0,3,0), core_on_chip=1), TpuDevice(id=26, process_index=2, coords=(1,3,0), core_on_chip=0), TpuDevice(id=27, process_index=2, coords=(1,3,0), core_on_chip=1)]
epoch:  1, train_loss: 2.2455, train_accuracy: 18.00, epoch_time: 0.661
epoch:  2, train_loss: 1.9817, train_accuracy: 58.91, epoch_time: 0.014
epoch:  3, train_loss: 1.4962, train_accuracy: 73.79, epoch_time: 0.014
epoch:  4, train_loss: 0.9364, train_accuracy: 79.30, epoch_time: 0.013
epoch:  5, train_loss: 0.6218, train_accuracy: 82.56, epoch_time: 0.014
epoch:  6, train_loss: 0.5105, train_accuracy: 84.74, epoch_time: 0.015
epoch:  7, train_loss: 0.4633, train_accuracy: 86.75, epoch_time: 0.015
epoch:  8, train_loss: 0.4391, train_accuracy: 87.88, epoch_time: 0.015
epoch:  9, train_loss: 0.4181, train_accuracy: 88.52, epoch_time: 0.016
epoch: 10, train_loss: 0.3930, train_accuracy: 89.16, epoch_time: 0.016
[32. 32. 32. 32. 32. 32. 32. 32.]
rn('jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() '  
```
  
  
</details>

  
  
- run with ray cluster 


#### launch the ray cluster :star:

```python
bash tpu_ray_cluster.sh
```


<details>
<summary>Expected Output</summary>


```python  
(base) ~/workspace/Github/ray-jax-tpu-pod-demos bash tpu_ray_cluster.sh
SSH key found in project metadata; not updating instance.
SSH: Attempting to connect to worker 0...
head node ip: 10.128.0.46
SSH key found in project metadata; not updating instance.
SSH: Attempting to connect to worker 0...
2022-06-02 05:44:02,285	VINFO scripts.py:1007 -- Send termination request to `/usr/local/lib/python3.8/dist-packages/ray/core/src/ray/raylet/raylet --raylet_socket_name=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/sockets/raylet --store_socket_name=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/sockets/plasma_store --object_manager_port=0 --min_worker_port=10002 --max_worker_port=19999 --node_manager_port=0 --node_ip_address=10.128.0.46 --maximum_startup_concurrency=96 --static_resource_list=TPU,1,node:10.128.0.46,1.0,CPU,96,memory,240226651136,object_store_memory,107239993344 "--python_worker_command=/usr/bin/python3 /usr/local/lib/python3.8/dist-packages/ray/workers/setup_worker.py /usr/local/lib/python3.8/dist-packages/ray/workers/default_worker.py --node-ip-address=10.128.0.46 --node-manager-port=RAY_NODE_MANAGER_PORT_PLACEHOLDER --object-store-name=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/sockets/plasma_store --raylet-name=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/sockets/raylet --redis-address=None --storage=None --temp-dir=/tmp/ray --metrics-agent-port=44278 --logging-rotate-bytes=536870912 --logging-rotate-backup-count=5 --gcs-address=10.128.0.46:6379 RAY_WORKER_DYNAMIC_OPTION_PLACEHOLDER --redis-password=5241590000000000" --java_worker_command= --cpp_worker_command= --native_library_path=/usr/local/lib/python3.8/dist-packages/ray/cpp/lib --redis_password=5241590000000000 --temp_dir=/tmp/ray --session_dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680 --log_dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/logs --resource_dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/runtime_resources --metrics-agent-port=44278 --metrics_export_port=64810 --object_store_memory=107239993344 --plasma_directory=/dev/shm --ray-debugger-external=0 --gcs-address=10.128.0.46:6379 "--agent_command=/usr/bin/python3 -u /usr/local/lib/python3.8/dist-packages/ray/dashboard/agent.py --node-ip-address=10.128.0.46 --metrics-export-port=64810 --dashboard-agent-port=44278 --listen-port=0 --node-manager-port=RAY_NODE_MANAGER_PORT_PLACEHOLDER --object-store-name=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/sockets/plasma_store --raylet-name=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/sockets/raylet --temp-dir=/tmp/ray --session-dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680 --runtime-env-dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/runtime_resources --log-dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/logs --logging-rotate-bytes=536870912 --logging-rotate-backup-count=5 --gcs-address=10.128.0.46:6379 --minimal"` (via SIGTERM)
2022-06-02 05:44:02,285	VINFO scripts.py:1007 -- Send termination request to `/usr/local/lib/python3.8/dist-packages/ray/core/src/ray/gcs/gcs_server --log_dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/logs --config_list=eyJvYmplY3Rfc3BpbGxpbmdfY29uZmlnIjogIntcInR5cGVcIjogXCJmaWxlc3lzdGVtXCIsIFwicGFyYW1zXCI6IHtcImRpcmVjdG9yeV9wYXRoXCI6IFwiL3RtcC9yYXkvc2Vzc2lvbl8yMDIyLTA2LTAyXzA1LTM3LTUxXzUxNjIzN180MDY4MFwifX0iLCAiaXNfZXh0ZXJuYWxfc3RvcmFnZV90eXBlX2ZzIjogdHJ1ZX0= --gcs_server_port=6379 --metrics-agent-port=44278 --node-ip-address=10.128.0.46 --redis_password=5241590000000000` (via SIGTERM)
2022-06-02 05:44:02,287	VINFO scripts.py:1007 -- Send termination request to `/usr/bin/python3 -u /usr/local/lib/python3.8/dist-packages/ray/autoscaler/_private/monitor.py --logs-dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/logs --logging-rotate-bytes=536870912 --logging-rotate-backup-count=5 --gcs-address=10.128.0.46:6379 --redis-password=5241590000000000 --monitor-ip=10.128.0.46` (via SIGTERM)
2022-06-02 05:44:02,288	VINFO scripts.py:1007 -- Send termination request to `/usr/bin/python3 -u /usr/local/lib/python3.8/dist-packages/ray/_private/log_monitor.py --logs-dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/logs --gcs-address=10.128.0.46:6379 --logging-rotate-bytes=536870912 --logging-rotate-backup-count=5` (via SIGTERM)
2022-06-02 05:44:02,290	VINFO scripts.py:1007 -- Send termination request to `/usr/bin/python3 -m ray.util.client.server --address=10.128.0.46:6379 --host=0.0.0.0 --port=10001 --mode=proxy --redis-password=5241590000000000 --metrics-agent-port=44278` (via SIGTERM)
2022-06-02 05:44:02,294	VINFO scripts.py:1007 -- Send termination request to `/usr/local/lib/python3.8/dist-packages/ray/core/src/ray/raylet/raylet --raylet_socket_name=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/sockets/raylet --store_socket_name=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/sockets/plasma_store --object_manager_port=0 --min_worker_port=10002 --max_worker_port=19999 --node_manager_port=0 --node_ip_address=10.128.0.46 --maximum_startup_concurrency=96 --static_resource_list=TPU,1,node:10.128.0.46,1.0,CPU,96,memory,240226651136,object_store_memory,107239993344 "--python_worker_command=/usr/bin/python3 /usr/local/lib/python3.8/dist-packages/ray/workers/setup_worker.py /usr/local/lib/python3.8/dist-packages/ray/workers/default_worker.py --node-ip-address=10.128.0.46 --node-manager-port=RAY_NODE_MANAGER_PORT_PLACEHOLDER --object-store-name=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/sockets/plasma_store --raylet-name=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/sockets/raylet --redis-address=None --storage=None --temp-dir=/tmp/ray --metrics-agent-port=44278 --logging-rotate-bytes=536870912 --logging-rotate-backup-count=5 --gcs-address=10.128.0.46:6379 RAY_WORKER_DYNAMIC_OPTION_PLACEHOLDER --redis-password=5241590000000000" --java_worker_command= --cpp_worker_command= --native_library_path=/usr/local/lib/python3.8/dist-packages/ray/cpp/lib --redis_password=5241590000000000 --temp_dir=/tmp/ray --session_dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680 --log_dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/logs --resource_dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/runtime_resources --metrics-agent-port=44278 --metrics_export_port=64810 --object_store_memory=107239993344 --plasma_directory=/dev/shm --ray-debugger-external=0 --gcs-address=10.128.0.46:6379 "--agent_command=/usr/bin/python3 -u /usr/local/lib/python3.8/dist-packages/ray/dashboard/agent.py --node-ip-address=10.128.0.46 --metrics-export-port=64810 --dashboard-agent-port=44278 --listen-port=0 --node-manager-port=RAY_NODE_MANAGER_PORT_PLACEHOLDER --object-store-name=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/sockets/plasma_store --raylet-name=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/sockets/raylet --temp-dir=/tmp/ray --session-dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680 --runtime-env-dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/runtime_resources --log-dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/logs --logging-rotate-bytes=536870912 --logging-rotate-backup-count=5 --gcs-address=10.128.0.46:6379 --minimal"` (via SIGTERM)
2022-06-02 05:44:02,296	VINFO scripts.py:1007 -- Send termination request to `/usr/local/lib/python3.8/dist-packages/ray/core/src/ray/raylet/raylet --raylet_socket_name=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/sockets/raylet --store_socket_name=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/sockets/plasma_store --object_manager_port=0 --min_worker_port=10002 --max_worker_port=19999 --node_manager_port=0 --node_ip_address=10.128.0.46 --maximum_startup_concurrency=96 --static_resource_list=TPU,1,node:10.128.0.46,1.0,CPU,96,memory,240226651136,object_store_memory,107239993344 "--python_worker_command=/usr/bin/python3 /usr/local/lib/python3.8/dist-packages/ray/workers/setup_worker.py /usr/local/lib/python3.8/dist-packages/ray/workers/default_worker.py --node-ip-address=10.128.0.46 --node-manager-port=RAY_NODE_MANAGER_PORT_PLACEHOLDER --object-store-name=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/sockets/plasma_store --raylet-name=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/sockets/raylet --redis-address=None --storage=None --temp-dir=/tmp/ray --metrics-agent-port=44278 --logging-rotate-bytes=536870912 --logging-rotate-backup-count=5 --gcs-address=10.128.0.46:6379 RAY_WORKER_DYNAMIC_OPTION_PLACEHOLDER --redis-password=5241590000000000" --java_worker_command= --cpp_worker_command= --native_library_path=/usr/local/lib/python3.8/dist-packages/ray/cpp/lib --redis_password=5241590000000000 --temp_dir=/tmp/ray --session_dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680 --log_dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/logs --resource_dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/runtime_resources --metrics-agent-port=44278 --metrics_export_port=64810 --object_store_memory=107239993344 --plasma_directory=/dev/shm --ray-debugger-external=0 --gcs-address=10.128.0.46:6379 "--agent_command=/usr/bin/python3 -u /usr/local/lib/python3.8/dist-packages/ray/dashboard/agent.py --node-ip-address=10.128.0.46 --metrics-export-port=64810 --dashboard-agent-port=44278 --listen-port=0 --node-manager-port=RAY_NODE_MANAGER_PORT_PLACEHOLDER --object-store-name=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/sockets/plasma_store --raylet-name=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/sockets/raylet --temp-dir=/tmp/ray --session-dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680 --runtime-env-dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/runtime_resources --log-dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/logs --logging-rotate-bytes=536870912 --logging-rotate-backup-count=5 --gcs-address=10.128.0.46:6379 --minimal"` (via SIGTERM)
2022-06-02 05:44:02,301	VINFO scripts.py:1007 -- Send termination request to `/usr/bin/python3 -u /usr/local/lib/python3.8/dist-packages/ray/_private/log_monitor.py --logs-dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/logs --gcs-address=10.128.0.46:6379 --logging-rotate-bytes=536870912 --logging-rotate-backup-count=5` (via SIGTERM)
2022-06-02 05:44:02,305	VINFO scripts.py:1007 -- Send termination request to `/usr/bin/python3 -u /usr/local/lib/python3.8/dist-packages/ray/dashboard/dashboard.py --host=localhost --port=8265 --port-retries=0 --temp-dir=/tmp/ray --log-dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/logs --session-dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680 --logging-rotate-bytes=536870912 --logging-rotate-backup-count=5 --gcs-address=10.128.0.46:6379 --minimal` (via SIGTERM)
2022-06-02 05:44:02,307	VINFO scripts.py:1007 -- Send termination request to `/usr/local/lib/python3.8/dist-packages/ray/core/src/ray/raylet/raylet --raylet_socket_name=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/sockets/raylet --store_socket_name=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/sockets/plasma_store --object_manager_port=0 --min_worker_port=10002 --max_worker_port=19999 --node_manager_port=0 --node_ip_address=10.128.0.46 --maximum_startup_concurrency=96 --static_resource_list=TPU,1,node:10.128.0.46,1.0,CPU,96,memory,240226651136,object_store_memory,107239993344 "--python_worker_command=/usr/bin/python3 /usr/local/lib/python3.8/dist-packages/ray/workers/setup_worker.py /usr/local/lib/python3.8/dist-packages/ray/workers/default_worker.py --node-ip-address=10.128.0.46 --node-manager-port=RAY_NODE_MANAGER_PORT_PLACEHOLDER --object-store-name=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/sockets/plasma_store --raylet-name=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/sockets/raylet --redis-address=None --storage=None --temp-dir=/tmp/ray --metrics-agent-port=44278 --logging-rotate-bytes=536870912 --logging-rotate-backup-count=5 --gcs-address=10.128.0.46:6379 RAY_WORKER_DYNAMIC_OPTION_PLACEHOLDER --redis-password=5241590000000000" --java_worker_command= --cpp_worker_command= --native_library_path=/usr/local/lib/python3.8/dist-packages/ray/cpp/lib --redis_password=5241590000000000 --temp_dir=/tmp/ray --session_dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680 --log_dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/logs --resource_dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/runtime_resources --metrics-agent-port=44278 --metrics_export_port=64810 --object_store_memory=107239993344 --plasma_directory=/dev/shm --ray-debugger-external=0 --gcs-address=10.128.0.46:6379 "--agent_command=/usr/bin/python3 -u /usr/local/lib/python3.8/dist-packages/ray/dashboard/agent.py --node-ip-address=10.128.0.46 --metrics-export-port=64810 --dashboard-agent-port=44278 --listen-port=0 --node-manager-port=RAY_NODE_MANAGER_PORT_PLACEHOLDER --object-store-name=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/sockets/plasma_store --raylet-name=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/sockets/raylet --temp-dir=/tmp/ray --session-dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680 --runtime-env-dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/runtime_resources --log-dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/logs --logging-rotate-bytes=536870912 --logging-rotate-backup-count=5 --gcs-address=10.128.0.46:6379 --minimal"` (via SIGTERM)
2022-06-02 05:44:02,307	VINFO scripts.py:1007 -- Send termination request to `/usr/bin/python3 -u /usr/local/lib/python3.8/dist-packages/ray/dashboard/agent.py --node-ip-address=10.128.0.46 --metrics-export-port=64810 --dashboard-agent-port=44278 --listen-port=0 --node-manager-port=46797 --object-store-name=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/sockets/plasma_store --raylet-name=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/sockets/raylet --temp-dir=/tmp/ray --session-dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680 --runtime-env-dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/runtime_resources --log-dir=/tmp/ray/session_2022-06-02_05-37-51_516237_40680/logs --logging-rotate-bytes=536870912 --logging-rotate-backup-count=5 --gcs-address=10.128.0.46:6379 --minimal` (via SIGTERM)
2022-06-02 05:44:02,954	SUCC scripts.py:1055 -- Stopped all 7 Ray processes.
2022-06-02 05:44:03,519	INFO usage_lib.py:320 -- Usage stats collection is enabled by default without user confirmation because this stdin is detected to be non-interactively. To disable this, add `--disable-usage-stats` to the command that starts the cluster, or run the following command: `ray disable-usage-stats` before starting the cluster. See https://docs.ray.io/en/master/cluster/usage-stats.html for more details.
2022-06-02 05:44:03,519	INFO scripts.py:719 -- Local node IP: 10.128.0.46
2022-06-02 05:44:05,217	SUCC scripts.py:761 -- --------------------
2022-06-02 05:44:05,217	SUCC scripts.py:762 -- Ray runtime started.
2022-06-02 05:44:05,217	SUCC scripts.py:763 -- --------------------
2022-06-02 05:44:05,217	INFO scripts.py:765 -- Next steps
2022-06-02 05:44:05,217	INFO scripts.py:766 -- To connect to this Ray runtime from another node, run
2022-06-02 05:44:05,217	INFO scripts.py:769 --   ray start --address='10.128.0.46:6379'
2022-06-02 05:44:05,217	INFO scripts.py:774 -- Alternatively, use the following Python code:
2022-06-02 05:44:05,217	INFO scripts.py:776 -- import ray
2022-06-02 05:44:05,218	INFO scripts.py:780 -- ray.init(address='auto')
2022-06-02 05:44:05,218	INFO scripts.py:792 -- To connect to this Ray runtime from outside of the cluster, for example to
2022-06-02 05:44:05,218	INFO scripts.py:796 -- connect to a remote cluster from your laptop directly, use the following
2022-06-02 05:44:05,218	INFO scripts.py:800 -- Python code:
2022-06-02 05:44:05,218	INFO scripts.py:802 -- import ray
2022-06-02 05:44:05,218	INFO scripts.py:803 -- ray.init(address='ray://<head_node_ip_address>:10001')
2022-06-02 05:44:05,218	INFO scripts.py:812 -- If connection fails, check your firewall settings and network configuration.
2022-06-02 05:44:05,218	INFO scripts.py:820 -- To terminate the Ray runtime, run
2022-06-02 05:44:05,218	INFO scripts.py:821 --   ray stop
SSH key found in project metadata; not updating instance.
SSH: Attempting to connect to worker 1...
2022-06-02 05:44:09,276	INFO scripts.py:1052 -- Did not find any active Ray processes.
2022-06-02 05:44:09,832	INFO scripts.py:874 -- Local node IP: 10.128.0.49
2022-06-02 05:44:10,798	SUCC scripts.py:886 -- --------------------
2022-06-02 05:44:10,798	SUCC scripts.py:887 -- Ray runtime started.
2022-06-02 05:44:10,798	SUCC scripts.py:888 -- --------------------
2022-06-02 05:44:10,798	INFO scripts.py:890 -- To terminate the Ray runtime, run
2022-06-02 05:44:10,798	INFO scripts.py:891 --   ray stop
[2022-06-02 05:44:10,797 I 13239 13239] global_state_accessor.cc:357: This node has an IP address of 10.128.0.49, while we can not found the matched Raylet address. This maybe come from when you connect the Ray cluster with a different IP address or connect a container.
SSH key found in project metadata; not updating instance.
SSH: Attempting to connect to worker 2...
2022-06-02 05:44:14,819	INFO scripts.py:1052 -- Did not find any active Ray processes.
[2022-06-02 05:44:16,198 I 13316 13316] global_state_accessor.cc:357: This node has an IP address of 10.128.0.47, while we can not found the matched Raylet address. This maybe come from when you connect the Ray cluster with a different IP address or connect a container.
2022-06-02 05:44:15,383	INFO scripts.py:874 -- Local node IP: 10.128.0.47
2022-06-02 05:44:16,199	SUCC scripts.py:886 -- --------------------
2022-06-02 05:44:16,199	SUCC scripts.py:887 -- Ray runtime started.
2022-06-02 05:44:16,199	SUCC scripts.py:888 -- --------------------
2022-06-02 05:44:16,199	INFO scripts.py:890 -- To terminate the Ray runtime, run
2022-06-02 05:44:16,199	INFO scripts.py:891 --   ray stop
SSH key found in project metadata; not updating instance.
SSH: Attempting to connect to worker 3...
2022-06-02 05:44:20,285	INFO scripts.py:1052 -- Did not find any active Ray processes.
[2022-06-02 05:44:21,436 I 13605 13605] global_state_accessor.cc:357: This node has an IP address of 10.128.0.48, while we can not found the matched Raylet address. This maybe come from when you connect the Ray cluster with a different IP address or connect a container.
2022-06-02 05:44:20,833	INFO scripts.py:874 -- Local node IP: 10.128.0.48
2022-06-02 05:44:21,438	SUCC scripts.py:886 -- --------------------
2022-06-02 05:44:21,438	SUCC scripts.py:887 -- Ray runtime started.
2022-06-02 05:44:21,438	SUCC scripts.py:888 -- --------------------
2022-06-02 05:44:21,438	INFO scripts.py:890 -- To terminate the Ray runtime, run
2022-06-02 05:44:21,438	INFO scripts.py:891 --   ray stop
```
  
</details>


#### run the file on the tpu (only need to run on head node) :star:

```python
gcloud alpha compute tpus tpu-vm ssh jax-trainer-mnist-tpu-pod --zone=us-central1-a --command "cd ~/ray-jax-tpu-pod-demos ; sudo python3 main.py" --worker 0
```
  

<details>
<summary>Failed example</summary>
  
```python
(base) ~/workspace/Github/ray-jax-tpu-pod-demos gcloud alpha compute tpus tpu-vm ssh jax-trainer-mnist-tpu-pod --zone=us-central1-a --command "cd ~/ray-jax-tpu-pod-demos ; sudo python3 main.py" --worker 0
SSH key found in project metadata; not updating instance.
SSH: Attempting to connect to worker 0...
2022-06-02 05:45:11.676578: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
(pid=41948) 2022-06-02 05:45:14.107022: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
(pid=41948) 2022-06-02 05:45:14.107064: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
(pid=13458, ip=10.128.0.47) 2022-06-02 05:45:14.145632: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
(pid=13458, ip=10.128.0.47) 2022-06-02 05:45:14.145675: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
(pid=13377, ip=10.128.0.49) 2022-06-02 05:45:14.225858: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
(pid=13377, ip=10.128.0.49) 2022-06-02 05:45:14.225906: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
(pid=13742, ip=10.128.0.48) 2022-06-02 05:45:14.224012: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
(pid=13742, ip=10.128.0.48) 2022-06-02 05:45:14.224065: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
(pid=41948) WARNING: Logging before InitGoogle() is written to STDERR
(pid=41948) I0000 00:00:1654148715.502791   41948 tpu_initializer_helper.cc:165] libtpu.so already in use by another process probably owned by another user. Run "$ sudo lsof -w /dev/accel0" to figure out which process is using the TPU. Not attempting to load libtpu.so in this process.
(pid=13742, ip=10.128.0.48) WARNING: Logging before InitGoogle() is written to STDERR
(pid=13742, ip=10.128.0.48) I0000 00:00:1654148715.567386   13742 tpu_initializer_helper.cc:165] libtpu.so already in use by another process probably owned by another user. Run "$ sudo lsof -w /dev/accel0" to figure out which process is using the TPU. Not attempting to load libtpu.so in this process.
(pid=13458, ip=10.128.0.47) WARNING: Logging before InitGoogle() is written to STDERR
(pid=13458, ip=10.128.0.47) I0000 00:00:1654148715.568145   13458 tpu_initializer_helper.cc:165] libtpu.so already in use by another process probably owned by another user. Run "$ sudo lsof -w /dev/accel0" to figure out which process is using the TPU. Not attempting to load libtpu.so in this process.
(pid=13377, ip=10.128.0.49) WARNING: Logging before InitGoogle() is written to STDERR
(pid=13377, ip=10.128.0.49) I0000 00:00:1654148715.640015   13377 tpu_initializer_helper.cc:165] libtpu.so already in use by another process probably owned by another user. Run "$ sudo lsof -w /dev/accel0" to figure out which process is using the TPU. Not attempting to load libtpu.so in this process.
(main pid=13742, ip=10.128.0.48) WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
(main pid=13742, ip=10.128.0.48) 2022-06-02 05:45:16.697753: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
(main pid=13742, ip=10.128.0.48) 2022-06-02 05:45:16.697778: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
(main pid=13742, ip=10.128.0.48) 2022-06-02 05:45:16.697809: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (t1v-n-6875f656-w-3): /proc/driver/nvidia/version does not exist
(main pid=13458, ip=10.128.0.47) WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
(main pid=13458, ip=10.128.0.47) 2022-06-02 05:45:16.734632: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
(main pid=13458, ip=10.128.0.47) 2022-06-02 05:45:16.734654: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
(main pid=13458, ip=10.128.0.47) 2022-06-02 05:45:16.734673: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (t1v-n-6875f656-w-2): /proc/driver/nvidia/version does not exist
(main pid=41948) WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
(main pid=41948) 2022-06-02 05:45:16.759850: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
(main pid=41948) 2022-06-02 05:45:16.759880: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
(main pid=41948) 2022-06-02 05:45:16.759901: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (t1v-n-6875f656-w-0): /proc/driver/nvidia/version does not exist
(main pid=13377, ip=10.128.0.49) WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
(main pid=13377, ip=10.128.0.49) 2022-06-02 05:45:16.811677: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
(main pid=13377, ip=10.128.0.49) 2022-06-02 05:45:16.811708: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
(main pid=13377, ip=10.128.0.49) 2022-06-02 05:45:16.811728: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (t1v-n-6875f656-w-1): /proc/driver/nvidia/version does not exist
Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]
Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 11.94 file/s]
Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]
Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]
Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 18.98 file/s]
Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 22.53 file/s]
{'10.128.0.46': 0, '10.128.0.47': 1, '10.128.0.48': 2, '10.128.0.49': 3}
10.128.0.46
(main pid=13742, ip=10.128.0.48) JAX process: 0 / 1
(main pid=13742, ip=10.128.0.48) JAX local devices: [CpuDevice(id=0)]
(main pid=13458, ip=10.128.0.47) JAX process: 0 / 1
(main pid=13458, ip=10.128.0.47) JAX local devices: [CpuDevice(id=0)]
(main pid=41948) JAX process: 0 / 1
(main pid=41948) JAX local devices: [CpuDevice(id=0)]
(main pid=13377, ip=10.128.0.49) JAX process: 0 / 1
(main pid=13377, ip=10.128.0.49) JAX local devices: [CpuDevice(id=0)]
(main pid=41948) Downloading and preparing dataset 11.06 MiB (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/jimmy/tensorflow_datasets/mnist/3.0.1...
(main pid=13742, ip=10.128.0.48) Downloading and preparing dataset 11.06 MiB (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/jimmy/tensorflow_datasets/mnist/3.0.1...
(main pid=13458, ip=10.128.0.47) Downloading and preparing dataset 11.06 MiB (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/jimmy/tensorflow_datasets/mnist/3.0.1...
(main pid=13377, ip=10.128.0.49) Downloading and preparing dataset 11.06 MiB (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/jimmy/tensorflow_datasets/mnist/3.0.1...
(main pid=41948) Dataset mnist downloaded and prepared to /home/jimmy/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.
(main pid=13458, ip=10.128.0.47) Dataset mnist downloaded and prepared to /home/jimmy/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.
(main pid=13377, ip=10.128.0.49) Dataset mnist downloaded and prepared to /home/jimmy/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.
(main pid=13742, ip=10.128.0.48) Dataset mnist downloaded and prepared to /home/jimmy/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.
(main pid=41948) epoch:  1, train_loss: 2.0537, train_accuracy: 42.87, epoch_time: 0.811
(main pid=41948) epoch:  2, train_loss: 0.9891, train_accuracy: 78.64, epoch_time: 0.282
(main pid=41948) epoch:  3, train_loss: 0.5043, train_accuracy: 85.14, epoch_time: 0.287
(main pid=41948) epoch:  4, train_loss: 0.4320, train_accuracy: 87.95, epoch_time: 0.288
(main pid=41948) epoch:  5, train_loss: 0.3870, train_accuracy: 89.12, epoch_time: 0.283
(main pid=41948) epoch:  6, train_loss: 0.3420, train_accuracy: 90.06, epoch_time: 0.297
(main pid=41948) epoch:  7, train_loss: 0.3106, train_accuracy: 90.84, epoch_time: 0.284
(main pid=13377, ip=10.128.0.49) epoch:  1, train_loss: 2.0537, train_accuracy: 42.87, epoch_time: 1.022
(main pid=41948) epoch:  8, train_loss: 0.2783, train_accuracy: 91.82, epoch_time: 0.285
(main pid=41948) epoch:  9, train_loss: 0.2504, train_accuracy: 92.72, epoch_time: 0.290
(main pid=13458, ip=10.128.0.47) epoch:  1, train_loss: 2.0537, train_accuracy: 42.87, epoch_time: 0.958
(main pid=13377, ip=10.128.0.49) epoch:  2, train_loss: 0.9891, train_accuracy: 78.64, epoch_time: 0.453
(main pid=13458, ip=10.128.0.47) epoch:  2, train_loss: 0.9891, train_accuracy: 78.64, epoch_time: 0.293
(main pid=41948) epoch: 10, train_loss: 0.2332, train_accuracy: 93.30, epoch_time: 0.288
(main pid=41948) [1.]
(main pid=13377, ip=10.128.0.49) epoch:  3, train_loss: 0.5043, train_accuracy: 85.14, epoch_time: 0.289
(main pid=13458, ip=10.128.0.47) epoch:  3, train_loss: 0.5043, train_accuracy: 85.14, epoch_time: 0.284
(main pid=13377, ip=10.128.0.49) epoch:  4, train_loss: 0.4320, train_accuracy: 87.95, epoch_time: 0.285
(main pid=13458, ip=10.128.0.47) epoch:  4, train_loss: 0.4320, train_accuracy: 87.95, epoch_time: 0.285
(main pid=13742, ip=10.128.0.48) epoch:  1, train_loss: 2.0537, train_accuracy: 42.87, epoch_time: 0.967
(main pid=13377, ip=10.128.0.49) epoch:  5, train_loss: 0.3870, train_accuracy: 89.12, epoch_time: 0.290
(main pid=13458, ip=10.128.0.47) epoch:  5, train_loss: 0.3870, train_accuracy: 89.12, epoch_time: 0.289
(main pid=13742, ip=10.128.0.48) epoch:  2, train_loss: 0.9891, train_accuracy: 78.64, epoch_time: 0.298
(main pid=13377, ip=10.128.0.49) epoch:  6, train_loss: 0.3420, train_accuracy: 90.06, epoch_time: 0.291
(main pid=13742, ip=10.128.0.48) epoch:  3, train_loss: 0.5043, train_accuracy: 85.14, epoch_time: 0.284
(main pid=13458, ip=10.128.0.47) epoch:  6, train_loss: 0.3420, train_accuracy: 90.06, epoch_time: 0.290
(main pid=13377, ip=10.128.0.49) epoch:  7, train_loss: 0.3106, train_accuracy: 90.84, epoch_time: 0.302
(main pid=13742, ip=10.128.0.48) epoch:  4, train_loss: 0.4320, train_accuracy: 87.95, epoch_time: 0.279
(main pid=13458, ip=10.128.0.47) epoch:  7, train_loss: 0.3106, train_accuracy: 90.84, epoch_time: 0.289
(main pid=13377, ip=10.128.0.49) epoch:  8, train_loss: 0.2783, train_accuracy: 91.82, epoch_time: 0.277
(main pid=13742, ip=10.128.0.48) epoch:  5, train_loss: 0.3870, train_accuracy: 89.12, epoch_time: 0.282
(main pid=13458, ip=10.128.0.47) epoch:  8, train_loss: 0.2783, train_accuracy: 91.82, epoch_time: 0.290
(main pid=13377, ip=10.128.0.49) epoch:  9, train_loss: 0.2504, train_accuracy: 92.72, epoch_time: 0.277
(main pid=13458, ip=10.128.0.47) epoch:  9, train_loss: 0.2504, train_accuracy: 92.72, epoch_time: 0.290
(main pid=13742, ip=10.128.0.48) epoch:  6, train_loss: 0.3420, train_accuracy: 90.06, epoch_time: 0.282
(main pid=13377, ip=10.128.0.49) epoch: 10, train_loss: 0.2332, train_accuracy: 93.30, epoch_time: 0.284
(main pid=13377, ip=10.128.0.49) [1.]
(main pid=13458, ip=10.128.0.47) epoch: 10, train_loss: 0.2332, train_accuracy: 93.30, epoch_time: 0.285
(main pid=13458, ip=10.128.0.47) [1.]
(main pid=13742, ip=10.128.0.48) epoch:  7, train_loss: 0.3106, train_accuracy: 90.84, epoch_time: 0.277
(main pid=13742, ip=10.128.0.48) epoch:  8, train_loss: 0.2783, train_accuracy: 91.82, epoch_time: 0.284
(main pid=13742, ip=10.128.0.48) epoch:  9, train_loss: 0.2504, train_accuracy: 92.72, epoch_time: 0.282
Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 25.92 file/s]
Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 17.82 file/s]
Dl Completed...: 100%|██████████| 4/4 [00:00<00:00, 14.74 file/s]
(main pid=41948) 2022-06-02 05:45:17.635914: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
(main pid=41948) To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  8.12 file/s]
(main pid=13458, ip=10.128.0.47) 2022-06-02 05:45:17.855961: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
(main pid=13458, ip=10.128.0.47) To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  8.88 file/s]
(main pid=13377, ip=10.128.0.49) 2022-06-02 05:45:17.863194: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
(main pid=13377, ip=10.128.0.49) To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 17.82 file/s]
Dl Completed...: 100%|██████████| 4/4 [00:01<00:00,  2.58 file/s]
(main pid=13742, ip=10.128.0.48) 2022-06-02 05:45:19.004834: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
(main pid=13742, ip=10.128.0.48) To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
(main pid=41948) /usr/local/lib/python3.8/dist-packages/jax/_src/tree_util.py:188: FutureWarning: jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() instead as a drop-in replacement.
(main pid=41948)   warnings.warn('jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() '
(main pid=13377, ip=10.128.0.49) /usr/local/lib/python3.8/dist-packages/jax/_src/tree_util.py:188: FutureWarning: jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() instead as a drop-in replacement.
(main pid=13377, ip=10.128.0.49)   warnings.warn('jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() '
(main pid=13458, ip=10.128.0.47) /usr/local/lib/python3.8/dist-packages/jax/_src/tree_util.py:188: FutureWarning: jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() instead as a drop-in replacement.
(main pid=13458, ip=10.128.0.47)   warnings.warn('jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() '
(main pid=13742, ip=10.128.0.48) /usr/local/lib/python3.8/dist-packages/jax/_src/tree_util.py:188: FutureWarning: jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() instead as a drop-in replacement.
(main pid=13742, ip=10.128.0.48)   warnings.warn('jax.tree_util.tree_multimap() is deprecated. Please use jax.tree_util.tree_map() '
(main pid=13742, ip=10.128.0.48) epoch: 10, train_loss: 0.2332, train_accuracy: 93.30, epoch_time: 0.286
(main pid=13742, ip=10.128.0.48) [1.]
```

vms does not use tpu properly, I would run the following code and rerun the job again. (<span style="color:red">the reason might be some process blocked.</span>)

```python
gcloud alpha compute tpus tpu-vm ssh jax-trainer-mnist-tpu-pod --zone=us-central1-a --command "sudo lsof -w /dev/accel0" --worker all
gcloud alpha compute tpus tpu-vm ssh jax-trainer-mnist-tpu-pod --zone=us-central1-a --command "sudo rm -f /tmp/libtpu_lockfile" --worker=all
```
</details>


<details>
<summary>Expected Output</summary>

```python
{'10.128.0.46': 0, '10.128.0.47': 1, '10.128.0.48': 2, '10.128.0.49': 3}
10.128.0.46
(main pid=20809, ip=10.128.0.47) JAX process: 1 / 4
(main pid=20809, ip=10.128.0.47) JAX local devices: [TpuDevice(id=4, process_index=1, coords=(2,0,0), core_on_chip=0), TpuDevice(id=5, process_index=1, coords=(2,0,0), core_on_chip=1), TpuDevice(id=6, process_index=1, coords=(3,0,0), core_on_chip=0), TpuDevice(id=7, process_index=1, coords=(3,0,0), core_on_chip=1), TpuDevice(id=12, process_index=1, coords=(2,1,0), core_on_chip=0), TpuDevice(id=13, process_index=1, coords=(2,1,0), core_on_chip=1), TpuDevice(id=14, process_index=1, coords=(3,1,0), core_on_chip=0), TpuDevice(id=15, process_index=1, coords=(3,1,0), core_on_chip=1)]
(main pid=22252, ip=10.128.0.48) JAX process: 3 / 4
(main pid=22252, ip=10.128.0.48) JAX local devices: [TpuDevice(id=20, process_index=3, coords=(2,2,0), core_on_chip=0), TpuDevice(id=21, process_index=3, coords=(2,2,0), core_on_chip=1), TpuDevice(id=22, process_index=3, coords=(3,2,0), core_on_chip=0), TpuDevice(id=23, process_index=3, coords=(3,2,0), core_on_chip=1), TpuDevice(id=28, process_index=3, coords=(2,3,0), core_on_chip=0), TpuDevice(id=29, process_index=3, coords=(2,3,0), core_on_chip=1), TpuDevice(id=30, process_index=3, coords=(3,3,0), core_on_chip=0), TpuDevice(id=31, process_index=3, coords=(3,3,0), core_on_chip=1)]
(main pid=20729, ip=10.128.0.49) JAX process: 2 / 4
(main pid=20729, ip=10.128.0.49) JAX local devices: [TpuDevice(id=16, process_index=2, coords=(0,2,0), core_on_chip=0), TpuDevice(id=17, process_index=2, coords=(0,2,0), core_on_chip=1), TpuDevice(id=18, process_index=2, coords=(1,2,0), core_on_chip=0), TpuDevice(id=19, process_index=2, coords=(1,2,0), core_on_chip=1), TpuDevice(id=24, process_index=2, coords=(0,3,0), core_on_chip=0), TpuDevice(id=25, process_index=2, coords=(0,3,0), core_on_chip=1), TpuDevice(id=26, process_index=2, coords=(1,3,0), core_on_chip=0), TpuDevice(id=27, process_index=2, coords=(1,3,0), core_on_chip=1)]
(main pid=51639) JAX process: 0 / 4
(main pid=51639) JAX local devices: [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0), TpuDevice(id=1, process_index=0, coords=(0,0,0), core_on_chip=1), TpuDevice(id=2, process_index=0, coords=(1,0,0), core_on_chip=0), TpuDevice(id=3, process_index=0, coords=(1,0,0), core_on_chip=1), TpuDevice(id=8, process_index=0, coords=(0,1,0), core_on_chip=0), TpuDevice(id=9, process_index=0, coords=(0,1,0), core_on_chip=1), TpuDevice(id=10, process_index=0, coords=(1,1,0), core_on_chip=0), TpuDevice(id=11, process_index=0, coords=(1,1,0), core_on_chip=1)]
(main pid=20809, ip=10.128.0.47) epoch:  1, train_loss: 0.6796, train_accuracy: 81.45, epoch_time: 0.663
(main pid=22252, ip=10.128.0.48) epoch:  1, train_loss: 0.6796, train_accuracy: 81.45, epoch_time: 0.529
(main pid=20729, ip=10.128.0.49) epoch:  1, train_loss: 0.6796, train_accuracy: 81.45, epoch_time: 0.595
(main pid=51639) epoch:  1, train_loss: 0.6796, train_accuracy: 81.45, epoch_time: 0.494
(main pid=20809, ip=10.128.0.47) epoch:  2, train_loss: 0.2089, train_accuracy: 93.81, epoch_time: 0.119
(main pid=22252, ip=10.128.0.48) epoch:  2, train_loss: 0.2089, train_accuracy: 93.81, epoch_time: 0.116
(main pid=20729, ip=10.128.0.49) epoch:  2, train_loss: 0.2089, train_accuracy: 93.81, epoch_time: 0.113
(main pid=51639) epoch:  2, train_loss: 0.2089, train_accuracy: 93.81, epoch_time: 0.110
(main pid=20729, ip=10.128.0.49) epoch:  3, train_loss: 0.1486, train_accuracy: 95.59, epoch_time: 0.112
(main pid=51639) epoch:  3, train_loss: 0.1486, train_accuracy: 95.59, epoch_time: 0.105
(main pid=20809, ip=10.128.0.47) epoch:  3, train_loss: 0.1486, train_accuracy: 95.59, epoch_time: 0.110
(main pid=22252, ip=10.128.0.48) epoch:  3, train_loss: 0.1486, train_accuracy: 95.59, epoch_time: 0.118
(main pid=20809, ip=10.128.0.47) epoch:  4, train_loss: 0.1137, train_accuracy: 96.70, epoch_time: 0.154
(main pid=22252, ip=10.128.0.48) epoch:  4, train_loss: 0.1137, train_accuracy: 96.70, epoch_time: 0.150
(main pid=20729, ip=10.128.0.49) epoch:  4, train_loss: 0.1137, train_accuracy: 96.70, epoch_time: 0.157
(main pid=51639) epoch:  4, train_loss: 0.1137, train_accuracy: 96.70, epoch_time: 0.166
(main pid=20809, ip=10.128.0.47) epoch:  5, train_loss: 0.0910, train_accuracy: 97.36, epoch_time: 0.120
(main pid=20729, ip=10.128.0.49) epoch:  5, train_loss: 0.0910, train_accuracy: 97.36, epoch_time: 0.129
(main pid=22252, ip=10.128.0.48) epoch:  5, train_loss: 0.0910, train_accuracy: 97.36, epoch_time: 0.119
(main pid=51639) epoch:  5, train_loss: 0.0910, train_accuracy: 97.36, epoch_time: 0.114
(main pid=22252, ip=10.128.0.48) epoch:  6, train_loss: 0.0755, train_accuracy: 97.88, epoch_time: 0.129
(main pid=51639) epoch:  6, train_loss: 0.0755, train_accuracy: 97.88, epoch_time: 0.129
(main pid=20809, ip=10.128.0.47) epoch:  6, train_loss: 0.0755, train_accuracy: 97.88, epoch_time: 0.129
(main pid=20729, ip=10.128.0.49) epoch:  6, train_loss: 0.0755, train_accuracy: 97.88, epoch_time: 0.129
(main pid=51639) epoch:  7, train_loss: 0.0636, train_accuracy: 98.19, epoch_time: 0.126
(main pid=20809, ip=10.128.0.47) epoch:  7, train_loss: 0.0636, train_accuracy: 98.19, epoch_time: 0.128
(main pid=20729, ip=10.128.0.49) epoch:  7, train_loss: 0.0636, train_accuracy: 98.19, epoch_time: 0.127
(main pid=22252, ip=10.128.0.48) epoch:  7, train_loss: 0.0636, train_accuracy: 98.19, epoch_time: 0.127
(main pid=51639) epoch:  8, train_loss: 0.0534, train_accuracy: 98.52, epoch_time: 0.112
(main pid=20809, ip=10.128.0.47) epoch:  8, train_loss: 0.0534, train_accuracy: 98.52, epoch_time: 0.114
(main pid=20729, ip=10.128.0.49) epoch:  8, train_loss: 0.0534, train_accuracy: 98.52, epoch_time: 0.113
(main pid=22252, ip=10.128.0.48) epoch:  8, train_loss: 0.0534, train_accuracy: 98.52, epoch_time: 0.117
(main pid=20809, ip=10.128.0.47) epoch:  9, train_loss: 0.0452, train_accuracy: 98.76, epoch_time: 0.114
(main pid=20729, ip=10.128.0.49) epoch:  9, train_loss: 0.0452, train_accuracy: 98.76, epoch_time: 0.109
(main pid=22252, ip=10.128.0.48) epoch:  9, train_loss: 0.0452, train_accuracy: 98.76, epoch_time: 0.114
(main pid=51639) epoch:  9, train_loss: 0.0452, train_accuracy: 98.76, epoch_time: 0.120
(main pid=20809, ip=10.128.0.47) epoch: 10, train_loss: 0.0389, train_accuracy: 98.98, epoch_time: 0.116
(main pid=22252, ip=10.128.0.48) epoch: 10, train_loss: 0.0389, train_accuracy: 98.98, epoch_time: 0.114
(main pid=20729, ip=10.128.0.49) epoch: 10, train_loss: 0.0389, train_accuracy: 98.98, epoch_time: 0.114
(main pid=51639) epoch: 10, train_loss: 0.0389, train_accuracy: 98.98, epoch_time: 0.120
(main pid=20809, ip=10.128.0.47) [32. 32. 32. 32. 32. 32. 32. 32.]
(main pid=20729, ip=10.128.0.49) [32. 32. 32. 32. 32. 32. 32. 32.]
(main pid=22252, ip=10.128.0.48) [32. 32. 32. 32. 32. 32. 32. 32.]
(main pid=51639) [32. 32. 32. 32. 32. 32. 32. 32.]
```


```python
(main pid=14529, ip=10.128.0.47) Could not open any log file.
(main pid=14529, ip=10.128.0.47) Could not open the log file '/tmp/tpu_logs/tpu_driver.t1v-n-6875f656-w-2.jimmy.log.INFO.20220602-054725.14529': Permission denied
```
this should be good (writing permission!)
</details>


### trouble-shooting

if not using tpu, the tpu might be locked (see: https://github.com/google/jax/issues/10192)

```python
gcloud alpha compute tpus tpu-vm ssh jax-trainer-mnist-tpu-pod --zone=us-central1-a --command "sudo lsof -w /dev/accel0" --worker all
gcloud alpha compute tpus tpu-vm ssh jax-trainer-mnist-tpu-pod --zone=us-central1-a --command "sudo rm -f /tmp/libtpu_lockfile" --worker=all
```

### stop the instance 

```python
gcloud alpha compute tpus tpu-vm delete jax-trainer-mnist-tpu-pod \
  --zone=us-central1-a 
```

#### Tips

- [gcp cheatsheet](https://gist.github.com/pydevops/cffbd3c694d599c6ca18342d3625af97)
- authetication
```python
gcloud auth login
```
- log into the tpu (head node)
```python
gcloud alpha compute tpus tpu-vm ssh jax-trainer-mnist-tpu-pod --zone=us-central1-a --worker 0
```



## Acknoledgements

Thanks to the help from [Antoni](https://github.com/Yard1) and [Matt](https://github.com/matthewdeng) and their demos [ray-train-demos](https://github.com/matthewdeng/ray-train-demos) and [swarm-jax](https://github.com/Yard1/swarm-jax)!
