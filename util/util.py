import subprocess


def get_gpu_names():
    gpus = subprocess.check_output(
        "nvidia-smi --query-gpu=name --format=csv,noheader", shell=True)
    return gpus.decode("utf-8").split("\n")[:-1]


def are_all_A100():
    gpus = get_gpu_names()
    return all("A100" in gpu for gpu in gpus)
