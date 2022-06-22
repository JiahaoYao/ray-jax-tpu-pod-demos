import functools
import os
import subprocess
import time

import glob
import requests
from fabric import Connection


@functools.lru_cache()
def get_bearer():
    return subprocess.check_output("gcloud auth print-access-token", shell=True).decode("utf-8").strip()


@functools.lru_cache()
def get_project():
    return subprocess.check_output("gcloud config list --format 'value(core.project)'", shell=True).decode(
        "utf-8").strip()


def create_tpu(
        name,
        zone,
        type,
        version, 
        preemptible,
):
    headers = {
        'Authorization': f'Bearer {get_bearer()}',
        'Content-Type': 'application/json',
    }

    params = (
        ('node_id', name),
    )

    data = {"accelerator_type":
                type,
            "runtime_version":
                version,
            "network_config":
                {"enable_external_ips": True},
            }

    if preemptible:
        data["schedulingConfig"] = {"preemptible": True}

    response = requests.post(f'https://tpu.googleapis.com/v2alpha1/projects/{get_project()}/locations/{zone}/nodes',
                             headers=headers, params=params, json=data)

    print(response.json())

    return response.status_code == 200


def check_tpu(name, zone):
    headers = {
        'Authorization': f'Bearer {get_bearer()}',
    }

    response = requests.get(
        f'https://tpu.googleapis.com/v2alpha1/projects/{get_project()}/locations/{zone}/nodes/{name}',
        headers=headers)

    return response.json()



def list_tpu_avail(zone):
    headers = {
        'Authorization': f'Bearer {get_bearer()}',
    }

    response = requests.get(
        f'https://tpu.googleapis.com/v2alpha1/projects/{get_project()}/locations/{zone}/acceleratorTypes',
        headers=headers)

    return response.json()




# GET https://tpu.googleapis.com/v1/{parent=projects/*/locations/*}/acceleratorTypes

def delete_tpu(name, zone):
    headers = {
        'Authorization': f'Bearer {get_bearer()}',
    }

    response = requests.delete(
        f'https://tpu.googleapis.com/v2alpha1/projects/{get_project()}/locations/{zone}/nodes/{name}',
        headers=headers)

    return response.json()


def wait_til(name, zone, state):
    while True:
        ret = check_tpu(name, zone)

        print(ret)

        matches = True
        for k, expected_v in state.items():
            if k not in ret:
                matches = False
                continue
            if ret[k] != expected_v:
                matches = False

        if "error" in ret:
            return False

        if ret["state"] == "TERMINATED":
            return False

        if matches:
            return True

        time.sleep(1)


def get_connection(
        name,
        zone,
):
    info = check_tpu(name, zone)
    outputs = []
    for i in info["networkEndpoints"]:
        outputs.append(Connection(i["ipAddress"],
                                  connect_kwargs={
                                      "key_filename": os.path.expanduser('~/.ssh/google_compute_engine'), }))
    return outputs


def start_ray(conn, address):
    conn.sudo('rm -rf *.py')
    conn.sudo('rm -rf swarm_jax')

    for i in glob.glob("*.py"):
        print(i)
        conn.put(i, "")

    conn.run("mkdir swarm_jax -p")

    for i in glob.glob("swarm_jax/*.py"):
        print(i)
        conn.put(i, "swarm_jax/")

    conn.sudo('python3 setup.py install')

    conn.put("scripts/init_ray.sh", "/tmp/ray-tpu.sh")
    print(conn.sudo('chmod +x /tmp/ray-tpu.sh'))
    print(conn.sudo('/tmp/ray-tpu.sh'))
    try:
        print(conn.run('ray stop -f'))
    except:
        pass
    print(conn.run(f"ray start --address={address} --load-code-from-local --resources='" + '{"tpu": 1}\''))
    
    
    


# from icecream import ic 
# list all the available resources 
# ic(list_tpu_avail(zone='us-central1-b'))
# ic(list_tpu_avail(zone='us-central1-a'))

# start the tpu
# create_tpu(name='ray_tpu_simon',zone='us-central1-a',type='v2-32', version='v2-alpha', preemptible=False)
# create_tpu(name='ray_tpu_simon',zone='us-central1-a',type='v2-32', version='tpu-vm-tf-2.8.0', preemptible=False)

# shut down tpu
# delete_tpu(name='ray_tpu_test',zone='us-central1-a')
# delete_tpu(name='ray_tpu_simon',zone='us-central1-a')

# print out the ips 
# ic(list_ips(name='ray_tpu_simon',zone='us-central1-a', external=False))
# ic(list_ips(name='ray_tpu_simon',zone='us-central1-a', external=True))