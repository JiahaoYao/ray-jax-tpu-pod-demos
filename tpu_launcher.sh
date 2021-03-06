gcloud alpha compute tpus tpu-vm create jax-trainer-mnist-tpu-pod \
  --zone=us-central1-a \
  --accelerator-type=v2-32 \
  --version=tpu-vm-tf-2.8.0 \
  --metadata-from-file=startup-script=tpu_setup.sh
