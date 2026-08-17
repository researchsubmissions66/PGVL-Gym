# Environment setup

PGVL-Gym uses its own `pgvl-gym` environment. It does not depend on a
server-wide `trident` environment or any packages installed into one.

## Create the full benchmark environment

Clone the repository and run the environment command from its root so the
editable package path in `environment.yml` resolves correctly:

```bash
git clone https://github.com/researchsubmissions66/PGVL-Gym.git
cd PGVL-Gym

conda env create --file environment.yml
conda activate pgvl-gym
```

The full profile uses Python 3.10, PyTorch 2.5.1, torchvision 0.20.1, CUDA
12.4, OpenSlide, and every dependency available from the normal package
indexes. It is intended for the two-GPU benchmark server and keeps PGVL-Gym
isolated from feature-extraction or unrelated research environments.

Confirm that the environment is internally consistent:

```bash
python -m pip check
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
PY
```

PGVL-Gym exposes modules as `common`, `methods`, and `clip`; it does not
provide a top-level `pgvl_gym` import. To verify the editable installation
without importing a foundation model:

```bash
python -c "import common, methods, clip; print('PGVL-Gym imports: OK')"
python scripts/list_backbone_compatibility.py
```

!!! note "Driver compatibility"

    `environment.yml` installs the CUDA 12.4 PyTorch runtime, not an NVIDIA
    driver. The host driver must support that runtime. Check it with
    `nvidia-smi` before launching a GPU job.

## Install on a shared cluster

On a multi-user cluster the environment usually cannot live in `$HOME`: a full
PyTorch and CUDA stack is 15-25 GB, which most home quotas will not hold. Build
it under a project filesystem with `--prefix` instead of `--name`:

```bash
export CONDA_PKGS_DIRS=/path/to/project/conda_pkgs   # keep the package cache off $HOME too
conda env create --file environment.yml --prefix /path/to/project/envs/pgvl-gym
```

A prefix environment is activated by path, and needs no `~/.condarc` change:

```bash
conda activate /path/to/project/envs/pgvl-gym
```

Point the job wrapper at it once and every submitted run picks it up:

```bash
export PGVL_CONDA_ENV=/path/to/project/envs/pgvl-gym
```

`scripts/pgvl_job.sh` activates `$PGVL_CONDA_ENV` on the compute node and exits
with status 78 if activation fails, so a broken environment is reported as an
environment problem rather than being mistaken for a modelling failure.

!!! warning "A cluster PyTorch module is not enough"

    A site-provided module such as `pytorch-conda` supplies torch, numpy,
    pandas and scikit-learn, but not `h5py`, `ftfy`, `torch_geometric`, or the
    gated encoder packages. Benchmark generation and several methods fail
    against it. Build the project environment described above.

### Offline compute nodes

Compute nodes on most clusters have no outbound network, while login nodes do.
Download every weight from a login node, then point jobs at that cache and make
a cache miss fail immediately instead of hanging on a connection attempt:

```bash
export HF_HOME=/path/to/project/.cache_huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

`scripts/pgvl_job.sh` sets all three, defaulting `HF_HOME` to the shared project
cache. Without them, `transformers` looks in `~/.cache/huggingface`, misses, and
every job that loads an encoder dies with

```
OSError: We couldn't connect to 'https://huggingface.co' to load the files,
and couldn't find them in the cached files.
```

!!! note "First import from a parallel filesystem is slow"

    An environment on Lustre or GPFS pages in thousands of shared objects on its
    first use from a given node, so a cold `import torch` can take minutes and
    look like a hung job. The cost is per node, not per job, and disappears once
    the file cache is warm. If it becomes a problem, stage the environment to
    node-local storage in the job script.

### Verified versions

The environment resolved from `environment.yml` on Python 3.10:

| Package | Version |
| --- | --- |
| torch | 2.5.1 |
| numpy | 1.26.4 |
| pandas | 2.3.3 |
| h5py | 3.16.0 |
| scikit-learn | 1.7.2 |
| transformers | 4.57.6 |
| timm | 1.0.28 |
| torch-geometric | 2.8.0.post1 |

`conda` stages a NumPy 2.x build while solving; the pip stage then downgrades it
to satisfy the `numpy<2` pin in `pyproject.toml`. That downgrade is expected, and
`python -m pip check` should still report no conflicts afterwards.

## Install only selected method families

For development on one method, create a smaller isolated environment and add
only that method's extra. Install PyTorch first so pip does not choose a
different CUDA build:

```bash
conda create --name pgvl-gym python=3.10 pip openslide \
  --channel conda-forge --yes
conda activate pgvl-gym

conda install pytorch=2.5.1 torchvision=0.20.1 pytorch-cuda=12.4 \
  --channel pytorch --channel nvidia --yes

python -m pip install -e .
python -m pip install -e ".[maple]"
```

Replace `maple` with one or more profiles:

| Profile | Additional runtime |
| --- | --- |
| `preprocessing` | OpenSlide Python bindings, OpenCV, Pillow, Matplotlib |
| `cod-mil` | configuration helpers and PyTorch Geometric |
| `maple` | graph construction, nmslib, and PyTorch Geometric |
| `mscpt` | PyTorch Geometric and PyTorch Lightning |
| `pathpt` | Nyström attention |
| `pathpt-musk` | PathPT plus timm and fairscale |
| `slip` | Nyström attention, OpenCLIP, and timm |
| `wsi-five` | report/vision training, LoRA, augmentation, and distributed loss |
| `convlm` | OpenCLIP attribute encoder support |
| `sldpc` | TITAN's einops extensions |
| `all` | every pip-installable profile above |

Several profiles can be combined in one installation:

```bash
python -m pip install -e ".[maple,mscpt,pathpt-musk]"
```

FOCUS, TOP, and MUSE do not require an additional PyPI profile beyond the
core package, but the selected encoder may require one of the gated installs
below.

## Gated model packages

Gated model code and weights are intentionally not downloaded by
`environment.yml`. Install only the encoders you are licensed and approved
to use.

### CONCH

After accepting the upstream model terms:

```bash
python -m pip install "git+https://github.com/mahmoodlab/CONCH.git"
hf auth login
```

A local CONCH checkpoint can be configured instead of downloading weights at
runtime.

### MUSK

After accepting the upstream terms:

```bash
python -m pip install fairscale
python -m pip install "git+https://github.com/lilab-stanford/MUSK.git"
hf auth login
```

The `pathpt-musk` and `all` profiles already install `fairscale`; the
MUSK repository and gated weights remain explicit steps.

KEEP, PLIP, and TITAN load through their registered Hugging Face interfaces.
They require the corresponding model access and cache, but not a second
project environment.

## Documentation-only environment

Documentation work does not need the benchmark stack:

```bash
conda create --name pgvl-gym-docs python=3.10 pip --yes
conda activate pgvl-gym-docs
python -m pip install -r requirements-docs.txt
python -m mkdocs serve
```

## Update or remove the environment

Apply future dependency changes from the repository root:

```bash
conda env update --name pgvl-gym --file environment.yml --prune
python -m pip check
```

Remove only this project environment when it is no longer needed:

```bash
conda deactivate
conda env remove --name pgvl-gym
```
