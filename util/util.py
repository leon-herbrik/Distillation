import subprocess


def get_gpu_names():
    gpus = subprocess.check_output(
        "nvidia-smi --query-gpu=name --format=csv,noheader", shell=True)
    return gpus.decode("utf-8").split("\n")[:-1]


def are_all_A100():
    gpus = get_gpu_names()
    return all("A100" in gpu for gpu in gpus)


def get_model_name(config):
    base = "GT"
    depth = config['model']['depth']
    num_frames = config['dataset']['num_frames']
    context_before, context_after = config['dataset'][
        'context_before'], config['dataset']['context_after']
    return f"{base}-D{depth}-F{num_frames}-CB{context_before}-CA{context_after}"
