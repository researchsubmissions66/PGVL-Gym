"""Generate a complete matrix of `configs/<method>/<dataset>[_<backbone>].yaml`.

Run from repo root:
    python scripts/generate_configs.py [--force]

For each method we keep ONE canonical training recipe (the paper's
default) and only the *dataset-dependent* fields change between
configs (n_classes, classnames, label_dict, dataset_csv, etc).

For PathPT (and any method that benchmarks several backbones with
identical hyperparameters), one config per (dataset × backbone) is
emitted; the only difference between them is `backbone` and
`feature_root` -- this is the locked-recipe design from the paper.
"""
from __future__ import annotations
import argparse
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CFG_DIR = REPO_ROOT / "configs"


# ---------------------------------------------------------------------------
# Per-dataset metadata. Used by every method.
# ---------------------------------------------------------------------------
DATASETS = {
    "ubc": {
        "n_classes": 5,
        "classnames": ["HGSC", "LGSC", "EC", "CC", "MC"],
        "label_dict": {"HGSC": 0, "LGSC": 1, "EC": 2, "CC": 3, "MC": 4},
        "task": "task_UBC-OCEAN_subtyping",
        "split_dirname": "UBC-OCEAN",
        "text_prompt": "text_prompts/focus/UBC_OCEAN_two_scale_text_prompt.csv",
        "focus_file_classnames": ["CC", "HGSC", "LGSC", "EC", "MC"],
        "focus_file_sha256": "d8e93638bca7a12b0282710de8f540ab33187e15b594a2d783752f51ec31f1ca",
        "focus_bank_sha256": "6f9c80918319220f0582fc591de4a27400f9bf34a50ce0f57c63db1e6092b04c",
        "focus_provenance": "upstream",
        "vila_prompt": "text_prompts/vila_mil/UBC_OCEAN_two_scale_text_prompt.csv",
        "vila_file_classnames": ["CC", "EC", "HGSC", "LGSC", "MC"],
        "vila_file_sha256": "f2a8e2c2ab45f0aece457029666fcce2f0a467a9c96ed7b866415ef70f9ba788",
        "vila_bank_sha256": "a693ac313d96ed3dd588507cca47de413cee4dc570a22ac35ff2639d70e7b418",
        "vila_provenance": "generated",
    },
    "lung": {
        "n_classes": 2,
        "classnames": ["lung adenocarcinoma", "lung squamous cell carcinoma"],
        "label_dict": {"LUAD": 0, "LUSC": 1},
        "task": "task_tcga_lung_subtyping",
        "split_dirname": "TCGA_Lung",
        "text_prompt": "text_prompts/focus/TCGA_NSCLC_two_scale_text_prompt.csv",
        "focus_file_classnames": ["LUAD", "LUSC"],
        "focus_file_sha256": "a8df24b0adbfa15369b51fb31f17adef4a9a6b955d25cd9ab0e09f3284c76c50",
        "focus_bank_sha256": "a966d04324b31fe04b81874d5bbace791fd44347eac19bef0fa73d7a4c156b3c",
        "focus_provenance": "upstream",
        "vila_prompt": "text_prompts/vila_mil/TCGA_Lung_two_scale_text_prompt.csv",
        "vila_file_classnames": ["LUAD", "LUSC"],
        "vila_file_sha256": "4fcea3a2016b2b0c600421da19a9b40441faef5a6b491d40ed80aa80e58319f1",
        "vila_bank_sha256": "61c001097a14a153dd8723cf5d85b41b8b2b0f61bb09b84b218280f1818eedfa",
        "vila_provenance": "upstream",
    },
    "rcc": {
        "n_classes": 3,
        "classnames": ["clear cell renal cell carcinoma",
                       "papillary renal cell carcinoma",
                       "chromophobe renal cell carcinoma"],
        "label_dict": {"CCRCC": 0, "PRCC": 1, "CHRCC": 2},
        "task": "task_tcga_rcc_subtyping",
        "split_dirname": "TCGA_RCC",
        "text_prompt": "text_prompts/focus/TCGA_RCC_two_scale_text_prompt.csv",
        "focus_file_classnames": ["CCRCC", "PRCC", "CHRCC"],
        "focus_file_sha256": "24c83ca73f7d8de17c6206189f9c8024a60b83336468cb976608fb1eb91aefb1",
        "focus_bank_sha256": "f311087ebf290b280746040886195a4731ad90c75b614f7a2886f27d17c3f422",
        "focus_provenance": "generated",
        "vila_prompt": "text_prompts/vila_mil/TCGA_RCC_two_scale_text_prompt.csv",
        "vila_file_classnames": ["CCRCC", "PRCC", "CHRCC"],
        "vila_file_sha256": "af95c33eaaa2ab47eeba139a98fcdf7e5137fe797748c8a7caff189c4340886a",
        "vila_bank_sha256": "288fd6b82c6d608ffb2e5d57ff191df47638171ff876d116a928edf4bf5c3077",
        "vila_provenance": "upstream",
    },
}

# MSCPT opens ``<gpt_dir>/description/<dataset_name>.json`` verbatim.  The
# official assets use title case for Lung and the full UBC-OCEAN dataset name.
MSCPT_DESCRIPTION_NAMES = {"lung": "Lung", "rcc": "RCC", "ubc": "UBC-OCEAN"}


# ---------------------------------------------------------------------------
# Per-method *recipe*: the bits that DON'T change between datasets.
# Each entry is a callable (dataset_meta) -> YAML body string.
# ---------------------------------------------------------------------------
def _yml_classnames(meta):
    return "[" + ", ".join(f'"{c}"' for c in meta["classnames"]) + "]"


def _yml_label_dict(meta):
    items = ", ".join(f'"{k}": {v}' for k, v in meta["label_dict"].items())
    return "{" + items + "}"


def _yml_string_list(values):
    return "[" + ", ".join(f'"{value}"' for value in values) + "]"


# ---- FOCUS ----------------------------------------------------------------
def cfg_focus(meta, dset):
    prompt_source = (
        "focus_upstream_native_two_scale_csv"
        if meta["focus_provenance"] == "upstream"
        else "focus_generated_native_two_scale_csv"
    )
    return textwrap.dedent(f"""\
        # FOCUS / {dset.upper()} / 16-shot / 10-fold
        # Source: official `{dset.upper()}.sh` style script from dddavid4real/FOCUS.
        # FOCUS uses CONCH only.

        method: "focus"
        backbone: "conch"
        feature_dim: 512
        n_classes: {meta['n_classes']}
        classnames: {_yml_classnames(meta)}
        label_dict: {_yml_label_dict(meta)}

        # --- few-shot setup ---
        shots: 16
        k: 10
        k_start: 0
        k_end: 10

        # --- training ---
        seed: 1
        epochs: 200
        lr: 1.0e-4
        weight_decay: 1.0e-5
        bag_loss: "ce"
        drop_out: true
        early_stopping: true
        es_patience: 20
        es_stop_epoch: 40

        # --- model ---
        prototype_number: 16
        mode: "transformer"
        loader_mode: "transformer"

        # --- paths (fill in your own) ---
        dataset_csv:    "splits/{meta['split_dirname']}/dataset.csv"
        data_folder_s:  "path/to/{meta['split_dirname']}/features_5x_conch"
        data_folder_l:  "path/to/{meta['split_dirname']}/features_20x_conch"
        split_dir:      "splits/{meta['split_dirname']}_16shots_10folds"
        text_prompt_path: "{meta['text_prompt']}"
        focus_prompt_format: "headerless_low_then_high"
        focus_prompt_file_classnames: {_yml_string_list(meta['focus_file_classnames'])}
        focus_prompt_file_sha256: "{meta['focus_file_sha256']}"
        focus_prompt_bank_sha256: "{meta['focus_bank_sha256']}"
        prompt_provenance: "{meta['focus_provenance']}"
        prompt_source: "{prompt_source}"
        conch_ckpt:     "ckpts/conch.pth"
        results_dir:    "results/focus_{dset}16"
        """)


# ---- ViLa-MIL ------------------------------------------------------------
def cfg_vila_mil(meta, dset):
    prompt_source = (
        "vila_mil_upstream_native_two_scale_csv"
        if meta["vila_provenance"] == "upstream"
        else "vila_mil_generated_native_two_scale_csv"
    )
    return textwrap.dedent(f"""\
        # ViLa-MIL / {dset.upper()} / 16-shot / 5-fold
        # Source: README of Jiangbo-Shi/ViLa-MIL.

        method: "vila_mil"
        backbone: "clip-rn50"
        feature_dim: 1024
        n_classes: {meta['n_classes']}
        classnames: {_yml_classnames(meta)}
        label_dict: {_yml_label_dict(meta)}

        shots: 16
        k: 5
        k_start: 0
        k_end: 5

        seed: 1
        epochs: 200
        lr: 1.0e-4
        weight_decay: 1.0e-5
        bag_loss: "ce"
        drop_out: true
        early_stopping: true
        es_patience: 20
        es_stop_epoch: 80          # ViLa-MIL uses a longer warm-up than FOCUS

        prototype_number: 16
        mode: "transformer"
        loader_mode: "transformer"

        dataset_csv:    "splits/{meta['split_dirname']}/dataset.csv"
        data_folder_s:  "path/to/{meta['split_dirname']}/feats_5x"
        data_folder_l:  "path/to/{meta['split_dirname']}/feats_20x"
        split_dir:      "splits/{meta['task']}_16shots_5folds"
        text_prompt_path: "{meta['vila_prompt']}"
        vila_prompt_format: "headerless_low_then_high"
        vila_prompt_file_classnames: {_yml_string_list(meta['vila_file_classnames'])}
        vila_prompt_file_sha256: "{meta['vila_file_sha256']}"
        vila_prompt_bank_sha256: "{meta['vila_bank_sha256']}"
        prompt_provenance: "{meta['vila_provenance']}"
        prompt_source: "{prompt_source}"
        results_dir:    "results/vila_mil_{dset}16"
        """)


# ---- CoD-MIL -------------------------------------------------------------
def cfg_cod_mil(meta, dset):
    if dset == "rcc":
        prompt_assets = textwrap.dedent("""\
            # The released 30-row RN50 tensor is audit-only because it is not
            # aligned with the published 27-row source CSV.
            text_prompt_bank_csv: "text_prompts/cod_mil/rcc_chain_of_diagnosis.csv"
            text_prompt_features: "text_prompts/cod_mil/rcc_text_prompt_features_clip_rn50_verified.pt"
            prompt_encoding: "precomputed"
            """)
    else:
        prompt_assets = textwrap.dedent("""\
            # No task-matched upstream tensor exists; encode the declared chain
            # and normal-tissue bank once with the matching RN50 text tower.
            text_prompt_features: null
            prompt_encoding: "runtime_cached"
            backbone_weights: "${PGVL_USER_ROOT}/.cache/clip/RN50.pt"
            """)
    return textwrap.dedent(f"""\
        # CoD-MIL / {dset.upper()} / full data / 5-fold
        # Source: README of Jiangbo-Shi/CoD-MIL.

        method: "cod_mil"
        backbone: "clip-rn50"
        feature_dim: 1024
        feature_space_id: "openai/clip-rn50@official"
        text_feature_space_id: "openai/clip-rn50@official"
        n_classes: {meta['n_classes']}
        classnames: {_yml_classnames(meta)}
        label_dict: {_yml_label_dict(meta)}

        shots: -1                    # CoD-MIL trains on full data by default
        k: 5
        k_start: 0
        k_end: 5

        seed: 1
        epochs: 200
        lr: 1.0e-4
        weight_decay: 1.0e-5
        early_stopping: true
        es_patience: 20
        es_stop_epoch: 50

        dataset_csv:    "splits/{meta['split_dirname']}/dataset.csv"
        data_folder_s:  "path/to/{meta['split_dirname']}/feats_5x"
        data_folder_l:  "path/to/{meta['split_dirname']}/feats_20x"
        split_dir:      "splits/{meta['task']}"
        text_prompt_path: "text_prompts/cod_mil/{dset}_chain_of_diagnosis.json"
{textwrap.indent(prompt_assets.rstrip(), '        ')}
        results_dir:    "results/cod_mil_{dset}"
        """)


# ---- MAPLE ---------------------------------------------------------------
def cfg_maple(meta, dset):
    provenance = "generated" if dset == "ubc" else "upstream"
    prompt_source = (
        "maple_task_extension_attribute_json" if dset == "ubc"
        else "maple_upstream_attribute_json"
    )
    return textwrap.dedent(f"""\
        # MAPLE / {dset.upper()} / 16-shot / 5-fold
        # Source: run.sh from JJ-ZHOU-Code/MAPLE.
        # MAPLE uses CLIP or PLIP (default PLIP).

        method: "maple"
        backbone: "plip"
        n_classes: {meta['n_classes']}
        classnames: {_yml_classnames(meta)}
        label_dict: {_yml_label_dict(meta)}

        shots: 16
        k: 5
        k_start: 0
        k_end: 5

        seed: 1
        epochs: 200
        lr: 2.0e-4                  # MAPLE uses 2e-4, not 1e-4
        weight_decay: 1.0e-5
        early_stopping: true
        es_patience: 20
        es_stop_epoch: 50

        prompt_mode: "attribute"
        attr_edge_topk: 7
        entity_weight: 0.3
        pos_ratio: 0.8
        n_ctx: 0
        csc: false
        all_ctx_trainable: false
        p_drop_out: 0.0
        p_bag_drop_out: 0.0

        dataset_csv:    "splits/{meta['split_dirname']}/dataset.csv"
        data_folder_s:  "data_dir/TCGA/{dset.upper()}/feats-l0-s2048-PLIP/pt_files"
        data_folder_l:  "data_dir/TCGA/{dset.upper()}/feats-l0-s1024-PLIP/pt_files"
        split_dir:      "splits/{meta['task']}_16shots_5folds"
        text_prompt_path: "text_prompts/maple/{dset.upper()}_attributes.json"
        prompt_provenance: "{provenance}"
        prompt_source: "{prompt_source}"
        results_dir:    "results/maple_{dset}16"
        """)


# ---- MSCPT ---------------------------------------------------------------
def cfg_mscpt(meta, dset, backbone):
    # MSCPT uses backbone-specific epoch counts: CLIP=100, PLIP=50, CONCH=50
    epochs = 100 if backbone == "clip" else 50
    return textwrap.dedent(f"""\
        # MSCPT / {dset.upper()} / 8-shot / 5-seed / {backbone.upper()} backbone
        # Source: scripts/mscpt/train_my_{dset}.sh from Hanminghao/MSCPT.
        # NOTE: MSCPT uses different epoch counts per backbone:
        #   CLIP=100, PLIP=50, CONCH=50.

        method: "mscpt"
        backbone: "{backbone}"
        n_classes: {meta['n_classes']}
        classnames: {_yml_classnames(meta)}
        label_dict: {_yml_label_dict(meta)}
        dataset_name: "{MSCPT_DESCRIPTION_NAMES[dset]}"

        shots: 8
        k: 5
        k_start: 0
        k_end: 5

        seed: 1
        epochs: {epochs}
        lr: 1.0e-4
        weight_decay: 1.0e-5
        batch_size: 1
        num_workers: 4

        n_tpro: 2
        n_vpro: 2
        n_set: 5
        num_k: 100

        dataset_csv:     "tcga_{dset}.csv"
        feat_data_dir:   "path/to/{meta['split_dirname']}/{backbone}/pt_files"
        selected_5x_dir: "path/to/{meta['split_dirname']}/selected_5x_patches"
        gpt_dir:         "./train_data/gpt"
        split_dir:       "numshots/{dset.upper()}_8shot_5fold"
        results_dir:     "results/mscpt_{dset}8_{backbone}"
        """)


# ---- PathPT --------------------------------------------------------------
def cfg_pathpt(meta, dset, backbone):
    # PathPT locks the recipe across backbones; only feature_root + MUSK cap differ
    extra = "patch_num: 100000            # OOM guard for MUSK" \
            if backbone == "musk" else "patch_num: null"
    return textwrap.dedent(f"""\
        # PathPT / {dset.upper()} / 10-shot / {backbone.upper()} backbone
        # Source: official train.py defaults from MAGIC-AI4Med/PathPT.
        # IMPORTANT: hyperparameters are intentionally identical across
        # all four backbones (PLIP / CONCH / KEEP / MUSK) -- only
        # `backbone` and `feature_root` change.  This is the paper's
        # fair-comparison recipe.

        method: "pathpt"
        backbone: "{backbone}"
        n_classes: {meta['n_classes']}
        classnames: {_yml_classnames(meta)}
        label_dict: {_yml_label_dict(meta)}

        shots: 10
        k: 10
        k_start: 0
        k_end: 10

        # --- training (locked across backbones) ---
        seed: 42
        epochs: 20
        lr: 1.0e-4
        batch_size: 1
        n_ctx: 32
        prompt_init: "template"
        aux_weight: 0.5
        learnable: "token"
        vision_only: false
        vision_grad: true
        use_aug: false
        loss_weight: [1.0, 0.5, 0.1]

        # --- features ---
        feature_root: "features/{backbone}/{dset}/h5_files"
        {extra}
        loader_mode: "h5"

        # --- paths ---
        dataset_csv:  "multifold/dataset_csv_10shot/{dset.upper()}/fold0.csv"
        split_dir:    "multifold/dataset_csv_10shot/{dset.upper()}"
        results_dir:  "results/pathpt_{dset}_{backbone}_10shot"
        """)


# ---- TOP -----------------------------------------------------------------
def cfg_top(meta, dset):
    return textwrap.dedent(f"""\
        # TOP / {dset.upper()} / 16-shot
        # Source: exp_TCGA.sh from miccaiif/TOP.
        # IMPORTANT: TOP uses VERY different hyper-parameters from the rest:
        #   LR ~ 0.02 (vs 1e-4),  epochs = 8000 (vs 20-200).

        method: "top"
        clip_arch: "RN50"
        n_classes: {meta['n_classes']}
        classnames: {_yml_classnames(meta)}
        label_dict: {_yml_label_dict(meta)}

        shots: 16
        k: 5

        seed: 0
        epochs: 8000
        lr: 0.02
        lr_TB: 0.02
        lr_IB: 0.02
        weight_decay: 0.0
        early_stopping: false

        n_ctx_bag: 4
        n_ctx_inst: 4
        ctx_init_bag: ""
        ctx_init_inst: ""
        csc: true
        p_drop_out: 0.2
        p_bag_drop_out: 0.2
        weight_lossA: 25
        pooling_strategy: "learnablePrompt_multi"

        dataset_csv: "splits/{meta['split_dirname']}/dataset.csv"
        data_folder_s: "path/to/{meta['split_dirname']}/clip_features"
        split_dir:   "splits/TOP_{meta['split_dirname']}_16shot"
        results_dir: "results/top_{dset}16"
        """)


# ---- SLIP ----------------------------------------------------------------
def cfg_slip(meta, dset):
    prompt_bank = {
        "lung": "TCGA_prompt_bank.json",
        "rcc": "tcga_rcc_tissues.json",
        "ubc": "ubc_ocean_tissues.json",
    }[dset]
    provenance = "upstream" if dset == "lung" else "generated"
    prompt_source = (
        "slip_upstream_complete_prompt_bank" if dset == "lung"
        else "slip_generated_task_extension_prompt_bank"
    )
    return textwrap.dedent(f"""\
        # SLIP / {dset.upper()} / 1-shot
        # Source: main.py defaults from LTS5/SLIP.
        # Backbones: CLIP / BiomedCLIP / PLIP / CLIP-RN50.

        method: "slip"
        backbone: "CLIP"
        n_classes: {meta['n_classes']}
        classnames: {_yml_classnames(meta)}
        label_dict: {_yml_label_dict(meta)}
        tissue_classnames_path: "${{PGVL_REPO_ROOT}}/text_prompts/slip/{prompt_bank}"
        prompt_provenance: "{provenance}"
        prompt_source: "{prompt_source}"

        shots: 1
        k: 5

        seed: 0
        epochs: 10                  # SLIP runs only a handful of epochs
        lr: 2.0e-3
        weight_decay: 0.0
        batch_size: 1
        early_stopping: false

        context_size: 1
        context_gain: 0.01
        topk: 50
        temp: 0.01
        image_size: 224

        dataroot: "./data"
        dataset_csv: "splits/{meta['split_dirname']}/dataset.csv"
        data_folder_s: "path/to/{meta['split_dirname']}/clip_features"
        split_dir:   "splits/SLIP_{meta['split_dirname']}_1shot"
        results_dir: "results/slip_{dset}_1shot"
        """)


# ---- WSI-FiVE ------------------------------------------------------------
WSI_FIVE_QUESTION_METADATA = {
    "lung": {
        "file_sha256":
            "642274c61e3388acda81dc461c2f0be4e8c6c3be8c8cf6285eb550d825335da8",
        "bank_sha256":
            "66689686c7d2305341dc42e71b27438412fc327275a80860e7f98499935f026f",
        "provenance": "derived",
    },
    "rcc": {
        "file_sha256":
            "c104cf4f2a9f1bba1daa6158041de62d8370ce64ffa148eb2a1151b4035ccb65",
        "bank_sha256":
            "96ca33c1413503c7552b4bd3c8d958ae82f3664e07f8759a650a89470b025b2b",
        "provenance": "generated",
    },
    "ubc": {
        "file_sha256":
            "ff302aaa9ef8d36dcb5ddc0cdd1d6df323fd8e2a7b37eb6f5b356c9e86fd75e6",
        "bank_sha256":
            "3ae81a381f60f53b2c9b217c193b3e14aef4c1ec8b3c511200de16d6222a2a70",
        "provenance": "generated",
    },
}


def cfg_wsi_five(meta, dset):
    native = dset == "lung"
    training_mode = ("upstream_answer_bank" if native
                     else "simplified_classnames")
    question_name = {"lung": "nsclc", "rcc": "rcc", "ubc": "ubc_ocean"}[dset]
    question_meta = WSI_FIVE_QUESTION_METADATA[dset]
    prompt_provenance = (
        "derived_questions_with_generated_answer_and_derived_evaluation_banks"
        if native else "generated_questions_with_classname_comparison")
    prompt_source = (
        "wsi_five_derived_upstream_text_assets"
        if native else "wsi_five_simplified_classname_baseline")
    native_assets = (
        '        report_csv: "text_prompts/wsi_five/'
        'nsclc_report_answers.csv"\n'
        '        evaluation_prompt_path: "text_prompts/wsi_five/'
        'nsclc_evaluation_prompts.json"\n'
        '        wsi_answer_file_sha256: '
        '"dbe2ebb91eb57e391f2f9fff807246815943a4a85036575375c2ec62d8b9d360"\n'
        '        wsi_answer_bank_sha256: '
        '"a853601f0cb18324627956e1a9304aa554e499e8164e798306cd80f7f611ac83"\n'
        '        wsi_answer_provenance: "generated"\n'
        '        wsi_evaluation_file_sha256: '
        '"3cdcdc0fe542e01ae8f416897e71159b6f55b555d9788b1d440653af9ab9a0b2"\n'
        '        wsi_evaluation_bank_sha256: '
        '"cc1647ebeb30b20f588da39992b5990a360506734d7633f50ee246c8f753b3da"\n'
        '        wsi_evaluation_provenance: "derived"\n'
        '        require_report: true\n'
        if native else
        '        require_report: false\n'
    )
    return textwrap.dedent(f"""\
        # WSI-FiVE / {dset.upper()}
        # Source: configs/wsi/fix_pth.yaml from ls1rius/WSI_FiVE.
        # Per-slide answers are native training targets, never inference input.

        method: "wsi_five"
        backbone: "wsi-five-vit"  # fixed precomputed 512-d feature boundary
        n_classes: {meta['n_classes']}
        classnames: {_yml_classnames(meta)}
        label_dict: {_yml_label_dict(meta)}

        seed: 0
        epochs: 30
        lr: 8.0e-6
        weight_decay: 0.05
        batch_size: 1
        num_frames: 2048
        T_mit: 8
        is_img_pth: true
        training_mode: "{training_mode}"
        clinical_questions: "text_prompts/wsi_five/clinical_questions/{question_name}.json"
        wsi_prompt_format: "six_questions_structured_answers_and_class_descriptions"
        wsi_question_file_sha256: "{question_meta['file_sha256']}"
        wsi_question_bank_sha256: "{question_meta['bank_sha256']}"
        wsi_question_provenance: "{question_meta['provenance']}"
        prompt_provenance: "{prompt_provenance}"
        prompt_source: "{prompt_source}"
{native_assets}
        dataset:    "tcga"
        data_path:  "path/to/{meta['split_dirname']}"

        dataset_csv: "splits/{meta['split_dirname']}/dataset.csv"
        data_folder_s: "path/to/{meta['split_dirname']}/features"
        split_dir:    "splits/WSI_FiVE_{meta['split_dirname']}"
        results_dir:  "results/wsi_five_{dset}"
        """)


# ---------------------------------------------------------------------------
# Build the matrix of (method, dataset[, backbone]) → file
# ---------------------------------------------------------------------------
PATHPT_BACKBONES = ("plip", "conch", "keep", "musk")
MSCPT_BACKBONES = ("clip", "plip", "conch")


def build_matrix():
    items = []
    for dset in DATASETS:
        meta = DATASETS[dset]
        items.append(("focus",     f"{dset}.yaml",            cfg_focus(meta, dset)))
        items.append(("vila_mil",  f"{dset}.yaml",            cfg_vila_mil(meta, dset)))
        items.append(("cod_mil",   f"{dset}.yaml",            cfg_cod_mil(meta, dset)))
        items.append(("maple",     f"{dset}.yaml",            cfg_maple(meta, dset)))
        items.append(("top",       f"{dset}.yaml",            cfg_top(meta, dset)))
        items.append(("slip",      f"{dset}.yaml",            cfg_slip(meta, dset)))
        items.append(("wsi_five",  f"{dset}.yaml",            cfg_wsi_five(meta, dset)))

        for bb in MSCPT_BACKBONES:
            items.append(("mscpt",  f"{dset}_{bb}.yaml",       cfg_mscpt(meta, dset, bb)))
        for bb in PATHPT_BACKBONES:
            items.append(("pathpt", f"{dset}_{bb}.yaml",       cfg_pathpt(meta, dset, bb)))
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing config files.")
    args = p.parse_args()

    written = 0
    skipped = 0
    for method, name, body in build_matrix():
        out = CFG_DIR / method / name
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and not args.force:
            skipped += 1
            continue
        out.write_text(body)
        written += 1

    total = written + skipped
    print(f"Wrote {written} / {total} configs (skipped {skipped} existing).")
    print("Tree:")
    for p in sorted(CFG_DIR.rglob("*.yaml")):
        print(f"  {p.relative_to(CFG_DIR)}")


if __name__ == "__main__":
    main()
