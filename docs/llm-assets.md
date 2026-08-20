# Prompting language models for text assets

Several methods embed text a paper published for one cohort and not for others.
WSI-FiVE answers six clinical questions per slide, but those answers exist for
TCGA-Lung alone — and the questions themselves are lung-specific. MUSE publishes
description banks for three tasks. CoD-MIL publishes a normal-tissue bank for
kidney only.

In native WSI-FiVE, generated or published per-slide answers are training
supervision only. They form a fold-local candidate bank and are never supplied
to fusion or classification at validation/test time. The fixed diagnostic
evaluation bank is a separate asset.

This framework can generate the missing text with a served language model. The
generated asset is an experimental input like any other, so it records exactly
what produced it and never passes as the published condition.

## The rule

**A generated asset carries its own provenance, and a result computed from it is
reported separately from one computed on published text.** Every file this
pipeline writes begins with a record like:

```json
{"_provenance": "generated",
 "_model": "hf:WenchuanZhang/Patho-R1-7B@7a69eb299bde72a4c3b8ec26fe0b17515346ef73",
 "_sampling": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512, "seed": 1},
 "_prompt_template": "wsi_five_answers/v1",
 "_generated_at": "2026-08-18T09:14:00+00:00"}
```

That is enough to regenerate the asset rather than trust it. Decoding is
deterministic by default (`temperature=0`, fixed seed).

## Registered models

`common/llm/registry.py` pins a repository **and a revision** for each model.

| Name | Model | Modality | Cached |
| --- | --- | --- | --- |
| `patho-r1-7b` | `WenchuanZhang/Patho-R1-7B` | vision-language (Qwen2.5-VL) | yes |
| `qwen2.5-7b-instruct` | `Qwen/Qwen2.5-7B-Instruct` | text | no |
| `quilt-llava-7b` | `wisdomik/Quilt-Llava-v1.5-7b` | vision-language | yes |
| `pathgen-llava` | `jamessyx/PathGen-LLaVA` | vision-language | no |

`patho-r1-7b` is the default for pathology text: it is tuned on diagnostic
language a general model has not seen. `qwen2.5-7b-instruct` is registered as a
**control at the same parameter count** — generating an asset with both is how
you show the pathology tuning contributed something, instead of assuming it did.

Models marked *not cached* must be downloaded from a login node; compute nodes
have no outbound network.

## vLLM lives in its own environment

`pyproject.toml` pins `torch>=2.4,<2.7` because the methods' numerics depend on
it, and vLLM pins torch hard in its own right. Installing vLLM into `pgvl-gym`
risks forcing a torch change that would break the `timm` pin and both
safetensors workarounds.

So vLLM is **never imported by this project**. It runs in a separate environment
and is reached over HTTP, since it speaks the OpenAI API.

```bash
# conda is not on PATH in a non-interactive shell; load it or call it by path
module load miniforge3-python          # or: /sw/rh9.4/python/miniforge3/bin/conda
conda create --yes --prefix /path/to/envs/pgvl-vllm python=3.11
/path/to/envs/pgvl-vllm/bin/python -m pip install vllm
export PGVL_VLLM_ENV=/path/to/envs/pgvl-vllm
```

Install the package with the environment's **own interpreter** rather than
activating it, and note that `scripts/llm_server.sh` invokes
`${PGVL_VLLM_ENV}/bin/python` directly for the same reason: a batch shell has no
conda hook, so `conda activate` fails there.

The launcher also exports `PYTHONNOUSERSITE=1`. Python otherwise places
`~/.local/lib/pythonX.Y/site-packages` ahead of the environment, and a stray
user-installed package can shadow one of vLLM's pinned dependencies — which
defeats the point of a separate environment.

Verified on this cluster:

```
vllm 0.27.1 | torch 2.13.0+cu130 | cuda 13.0
```

That torch is far ahead of the benchmark's pinned `2.5.1`, which is exactly why
the two environments are separate. Check that the GPU partition's driver
supports the CUDA build vLLM pulled in — `cudatoolkit/26.5_13.2` being available
here indicates a CUDA 13-capable driver.

## Serving a model

```bash
scripts/llm_server.sh patho-r1-7b                # sbatch on a GPU node
scripts/llm_server.sh patho-r1-7b --foreground   # run on an interactive node
```

The launcher resolves the repository and revision **from the registry**, so the
served model is the one the client later verifies against rather than a
hand-typed path. It prints the endpoint into `logs/vllm-<model>-<jobid>.out`;
wait for it with:

```bash
until grep -m1 -o 'http://[^ ]*' logs/vllm-patho-r1-7b-<jobid>.out; do sleep 10; done
```

Tunable through the environment: `PGVL_VLLM_PORT`, `PGVL_VLLM_GPUS`,
`PGVL_VLLM_WALLTIME`, `PGVL_VLLM_ARGS`, `PGVL_SLURM_ACCOUNT`,
`PGVL_SLURM_PARTITION`.

## Generating an asset

```bash
python scripts/generate_llm_assets.py \
    --task wsi_five_answers --cohort rcc \
    --model patho-r1-7b --endpoint http://gpu042:8000/v1 \
    --limit 5          # smoke run first
```

The cohort's question set and report CSV come from its `protocol.yaml`, so the
generated answers respond to the questions that cohort actually declares.

### Attribution is verified, not asserted

Before the first completion the client queries `/v1/models` and compares the
served identity against the requested spec. If they disagree it **refuses**:

```
http://gpu042:8000/v1 serves 'qwen2.5-7b-instruct', which does not match
requested model 'WenchuanZhang/Patho-R1-7B'. Refusing to attribute generated
text to a model that did not produce it.
```

Recording one model's provenance on another's output would be the same class of
error as mislabelling a feature space, and is caught the same way.

### Cohorts that already ship published text

Running the generator against a cohort whose report CSV already holds answers
fails deliberately:

```
text_prompts/wsi_five/nsclc_report_answers.csv has no free-text report column
for cohort nsclc. If this cohort already ships answers, it does not need
generating -- generated text would displace published text.
```

TCGA-Lung is the reference cohort. Its checked-in complete bank preserves 912
author answers and fills 27 blank upstream cells with the conservative local
policy disclosed in `text_prompts/PROVENANCE.json`; an external model must not
silently replace either portion.

## What this does not do

Methods do **not** call a language model at train or eval time. Generation is an
offline step producing a reviewable file, so a run does not depend on a live
server, and a result stays reproducible from committed assets alone.
