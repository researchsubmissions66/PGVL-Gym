"""WSI-FiVE patch bags and text supervision.

Native WSI-FiVE uses the authors' six GPT-derived answers as *training
targets*. They are returned to the adapter but never supplied to the visual
fusion transformer at validation or test time.
"""
from __future__ import annotations
import os
import torch
import pandas as pd
from torch.utils.data import Dataset

from common.datasets.dataset_generic import _load_feature_tensor
from .prompts import ANSWER_FIELD_COUNT, normalize_answer_fields


class WSI_FiVE_Dataset(Dataset):
    NATIVE_MODE = "upstream_answer_bank"

    def __init__(self, csv_path: str | pd.DataFrame, feature_root: str,
                 report_csv: str | None, label_dict: dict,
                 max_patches: int = 2048,
                 tokenizer=None, max_text_len: int = 77,
                 default_report: str | None = None,
                 random_subsampling: bool = True,
                 feature_dim: int | None = None,
                 supervision_mode: str = "simplified_classnames"):
        self.df = (csv_path.copy() if isinstance(csv_path, pd.DataFrame)
                   else pd.read_csv(csv_path))
        self.feature_root = feature_root
        self.label_dict = label_dict
        self.max_patches = max_patches
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.feature_path_column = None
        self.feature_key = "features"
        self.include_metadata = False
        self.require_report = True
        self.default_report = str(default_report or "").strip()
        self.random_subsampling = bool(random_subsampling)
        self.feature_dim = (
            int(feature_dim) if feature_dim is not None else None)
        self.supervision_mode = str(supervision_mode).strip().lower()

        self.reports = {}
        self.answer_fields: dict[str, tuple[str, ...]] = {}
        if report_csv is not None and os.path.isfile(report_csv):
            rdf = pd.read_csv(report_csv)
            id_col = next((column for column in (
                "slide_id", "case_id", "patient_id", "patient_filename")
                           if column in rdf.columns), None)
            txt_col = next((column for column in ("answer", "report", "text")
                            if column in rdf.columns), None)
            question_columns = tuple(
                f"q{index}" for index in range(1, ANSWER_FIELD_COUNT + 1))
            has_structured_answers = all(
                column in rdf.columns for column in question_columns)
            if id_col is None or (
                    txt_col is None and not has_structured_answers):
                raise ValueError(
                    f"{report_csv} needs an ID column (slide_id/case_id/patient_id) "
                    "and an answer/report/text column or q1..q6")
            if (self.supervision_mode == self.NATIVE_MODE
                    and not has_structured_answers):
                raise ValueError(
                    f"{report_csv} needs q1..q6 for WSI-FiVE native "
                    "answer-bank supervision")
            for row_number, r in rdf.iterrows():
                raw_identifier = r[id_col]
                if pd.isna(raw_identifier) or not str(raw_identifier).strip():
                    raise ValueError(
                        f"{report_csv} row {row_number + 2} has a blank report ID")
                identifier = str(raw_identifier).strip()
                keys = tuple(dict.fromkeys((identifier, identifier[:12])))
                text = ""
                if txt_col is not None:
                    raw_text = r[txt_col]
                    if pd.isna(raw_text) or not str(raw_text).strip():
                        raise ValueError(
                            f"{report_csv} row {row_number + 2} has blank "
                            "report text")
                    text = str(raw_text).strip()
                fields = None
                if has_structured_answers:
                    try:
                        fields = normalize_answer_fields(
                            tuple(r[column] for column in question_columns))
                    except ValueError as error:
                        raise ValueError(
                            f"{report_csv} row {row_number + 2}: {error}") from error
                for key in keys:
                    existing = self.reports.get(key)
                    if text and existing is not None and existing != text:
                        raise ValueError(
                            f"{report_csv} has conflicting reports for ID {key!r}")
                    if text:
                        self.reports[key] = text
                    existing_fields = self.answer_fields.get(key)
                    if (fields is not None and existing_fields is not None
                            and existing_fields != fields):
                        raise ValueError(
                            f"{report_csv} has conflicting answers for ID {key!r}")
                    if fields is not None:
                        self.answer_fields[key] = fields

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _case_id(row) -> str:
        slide_id = str(row["slide_id"]).strip()
        raw_case_id = row.get("case_id")
        return (str(raw_case_id).strip()
                if raw_case_id is not None and not pd.isna(raw_case_id)
                else slide_id[:12])

    @classmethod
    def _row_keys(cls, row) -> tuple[str, ...]:
        slide_id = str(row["slide_id"]).strip()
        return tuple(dict.fromkeys(
            (slide_id, cls._case_id(row), slide_id[:12])))

    def supervision_for_row(self, row):
        """Resolve native answer fields or legacy report text for one row."""
        keys = self._row_keys(row)
        if self.supervision_mode == self.NATIVE_MODE:
            for key in keys:
                if key in self.answer_fields:
                    return self.answer_fields[key]
            return ()
        for key in keys:
            if key in self.reports:
                return self.reports[key]
        return self.default_report

    def native_answer_bank(self) -> tuple[tuple[str, ...], ...]:
        """Return unique training-fold answers without loading feature bags."""
        if self.supervision_mode != self.NATIVE_MODE:
            raise RuntimeError(
                "WSI-FiVE native_answer_bank requires upstream_answer_bank mode")
        unique: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()
        for row_number, (_, row) in enumerate(self.df.iterrows(), start=2):
            fields = self.supervision_for_row(row)
            if not fields:
                raise FileNotFoundError(
                    "WSI-FiVE has no structured training answers for "
                    f"{row['slide_id']} (split row {row_number})")
            normalized = normalize_answer_fields(fields)
            if normalized not in seen:
                seen.add(normalized)
                unique.append(normalized)
        if not unique:
            raise ValueError("WSI-FiVE training fold has no answer candidates")
        return tuple(unique)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = str(row["slide_id"])
        label_key = row["label"]
        label = self.label_dict[label_key] \
                if isinstance(label_key, str) else int(label_key)

        if self.feature_path_column:
            feature_path = row[self.feature_path_column]
        else:
            feature_path = os.path.join(self.feature_root, f"{slide_id}.pt")
        feats = _load_feature_tensor(feature_path, self.feature_key)
        if self.feature_dim is not None and feats.shape[1] != self.feature_dim:
            raise ValueError(
                f"WSI-FiVE features for {slide_id!r} have width "
                f"{feats.shape[1]}, expected {self.feature_dim}")
        # The fusion transformer positions patches by their index within the
        # slide and masks padding per slide, so the sampled indices and the
        # slide's full patch count are part of the input, not bookkeeping.
        patch_pub_cnt = int(feats.shape[0])
        if feats.shape[0] > self.max_patches:
            if self.random_subsampling:
                keep = torch.randperm(
                    feats.shape[0])[: self.max_patches].sort().values
            else:
                keep = torch.linspace(
                    0, feats.shape[0] - 1,
                    self.max_patches).round().long()
            feats = feats[keep]
            patch_inds = keep
        else:
            patch_inds = torch.arange(feats.shape[0])

        # TCGA answer assets are patient-level (TCGA-XX-XXXX), whereas patch
        # bags normally use a longer slide identifier. Prefer the exact match.
        case_id = self._case_id(row)
        supervision = self.supervision_for_row(row)
        if self.require_report and not supervision:
            raise FileNotFoundError(
                f"WSI-FiVE has no text supervision for slide {slide_id}")
        if self.tokenizer is not None:
            if not isinstance(supervision, str):
                raise TypeError(
                    "WSI-FiVE tokenizer mode accepts report strings only")
            text_ids = self.tokenizer(supervision, return_tensors="pt",
                                      padding="max_length",
                                      truncation=True,
                                      max_length=self.max_text_len)["input_ids"][0]
            text_output = text_ids
        else:
            text_output = supervision

        patch_info = {
            "patch_inds": patch_inds.float(),
            "patch_pub_cnt": torch.tensor(float(max(patch_pub_cnt, 1))),
            "sample_range": int(feats.shape[0]),
        }
        if self.include_metadata:
            return feats, text_output, patch_info, {
                "slide_id": slide_id, "case_id": case_id,
            }, label
        return feats, text_output, patch_info, label


def _collate_one(batch):
    if len(batch) != 1:
        raise ValueError("WSI-FiVE uses batch_size=1 for variable-length slides")
    item = batch[0]
    metadata = None
    if len(item) == 5:
        feats, report, patch_info, metadata, label = item
    else:
        feats, report, patch_info, label = item
    collated_info = {
        "patch_inds": patch_info["patch_inds"].unsqueeze(0),
        "patch_pub_cnt": patch_info["patch_pub_cnt"].reshape(1),
        "sample_range": [patch_info["sample_range"]],
    }
    if metadata is not None:
        return (feats.unsqueeze(0), [report], collated_info,
                {"slide_id": [metadata["slide_id"]],
                 "case_id": [metadata["case_id"]]}, torch.tensor([label]))
    return feats.unsqueeze(0), [report], collated_info, torch.tensor([label])


def build_wsi_five_loader(cfg, split: str = "train", shuffle: bool = True,
                          fold: int | None = None):
    from torch.utils.data import DataLoader
    from common.datasets.split_tables import load_phase_table

    fold = cfg.get("_fold_index", 0) if fold is None else fold
    phase_table = load_phase_table(cfg, split, fold)
    supervision_mode = cfg.get("training_mode", "simplified_classnames")
    # Do not even load the patient-answer asset for held-out native phases.
    # The eval contract is patch bag + fixed diagnostic descriptions only.
    report_csv = cfg.get("report_csv")
    if supervision_mode == WSI_FiVE_Dataset.NATIVE_MODE and split != "train":
        report_csv = None
    ds = WSI_FiVE_Dataset(
        csv_path=phase_table,
        feature_root=cfg.get("feature_root", cfg.get("data_folder_s", cfg.get("data_path", "."))),
        report_csv=report_csv,
        label_dict=cfg["label_dict"],
        max_patches=cfg.get("num_frames", 2048),
        default_report=cfg.get("default_report"),
        random_subsampling=split == "train",
        feature_dim=cfg.get("feature_dim"),
        supervision_mode=supervision_mode)
    ds.feature_path_column = cfg.get("feature_path_column")
    ds.feature_key = cfg.get("feature_key", "features")
    ds.include_metadata = cfg.get("include_metadata", False)
    if ds.supervision_mode == ds.NATIVE_MODE:
        # Native answers are training supervision, never validation/test input.
        ds.require_report = split == "train"
    else:
        ds.require_report = cfg.get("require_report", False)
    if cfg.get("batch_size", 1) != 1:
        raise ValueError("WSI-FiVE unified loader requires batch_size=1")
    return DataLoader(ds, batch_size=1,
                      shuffle=shuffle and split == "train",
                      num_workers=cfg.get("num_workers", 2),
                      collate_fn=_collate_one)
