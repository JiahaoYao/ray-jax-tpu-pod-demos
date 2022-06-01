# ray-jax-tpu-pod-demos
Demos starting ray cluster on tpu pod

## MNIST classification

Trains a simple fully connected network on the MNIST dataset.

(Adopted from https://github.com/google/flax/tree/main/examples/mnist with the following modifications to better demonstrate the training speeedup on this small network.)


### setup tpu pod 

```
bash tpu_launcher.py
```

### test the jax tpu

```python 
gcloud alpha compute tpus tpu-vm ssh jax-trainer-mnist-tpu-pod --zone=us-central1-a --command "sudo python3 -c \"import jax; print(jax.device_count(), jax.local_device_count())\"" --worker all
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

- run with ray cluster 


#### launch the ray cluster :star:

**NOTE**: Remember to change the ip address of `10.128.0.38` to your head node ip address!!!
**NOTE**: Remember to change the ip address of `10.128.0.38` to your head node ip address!!!
**NOTE**: Remember to change the ip address of `10.128.0.38` to your head node ip address!!!

```python 
gcloud alpha compute tpus tpu-vm ssh jax-trainer-mnist-tpu-pod --zone=us-central1-a --command "ray stop && ray start --head --port=6379 --resources='{\"TPU\":1}'" --worker=0

gcloud alpha compute tpus tpu-vm ssh jax-trainer-mnist-tpu-pod --zone=us-central1-a --command "ray stop && ray start --address='10.128.0.38:6379' --resources='{\"TPU\":1}'" --worker=1

gcloud alpha compute tpus tpu-vm ssh jax-trainer-mnist-tpu-pod --zone=us-central1-a --command "ray stop && ray start --address='10.128.0.38:6379' --resources='{\"TPU\":1}'" --worker=2

gcloud alpha compute tpus tpu-vm ssh jax-trainer-mnist-tpu-pod --zone=us-central1-a --command "ray stop && ray start --address='10.128.0.38:6379' --resources='{\"TPU\":1}'" --worker=3
```


#### run the file on the tpu (only need to run on head node) :star:

```python
gcloud alpha compute tpus tpu-vm ssh jax-trainer-mnist-tpu-pod --zone=us-central1-a --command "cd ~/ray-jax-tpu-pod-demos ; sudo python3 main.py" --worker 0
```

### trouble-shooting

if not using tpu, the tpu might be locked 

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

- log into the tpu (head node)
```python
gcloud alpha compute tpus tpu-vm ssh jax-trainer-mnist-tpu-pod --zone=us-central1-a --worker 0
```


## Acknoledgements

Thanks to the help from [Antoni](https://github.com/Yard1) and [Matt](https://github.com/matthewdeng) and their demos [ray-train-demos](https://github.com/matthewdeng/ray-train-demos) and [swarm-jax](https://github.com/Yard1/swarm-jax)!