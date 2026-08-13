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
