"""WSI-FiVE-style dataset.

WSI-FiVE loads patch features + GPT-summarised pathology report
embeddings.  CSV format:
    slide_id, label, report_text
"""
from __future__ import annotations
import os
import torch
import pandas as pd
from torch.utils.data import Dataset

from common.datasets.dataset_generic import _load_feature_tensor


class WSI_FiVE_Dataset(Dataset):
    def __init__(self, csv_path: str, feature_root: str,
                 report_csv: str | None, label_dict: dict,
                 max_patches: int = 2048,
                 tokenizer=None, max_text_len: int = 77,
                 default_report: str | None = None):
        self.df = pd.read_csv(csv_path)
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

        self.reports = {}
        if report_csv is not None and os.path.isfile(report_csv):
            rdf = pd.read_csv(report_csv)
            id_col = next((column for column in (
                "slide_id", "case_id", "patient_id", "patient_filename")
                           if column in rdf.columns), None)
            txt_col = next((column for column in ("report", "text")
                            if column in rdf.columns), None)
            if id_col is None or txt_col is None:
                raise ValueError(
                    f"{report_csv} needs an ID column (slide_id/case_id/patient_id) "
                    "and a report/text column")
            for _, r in rdf.iterrows():
                identifier = str(r[id_col])
                self.reports[identifier] = str(r[txt_col])
                self.reports.setdefault(identifier[:12], str(r[txt_col]))

    def __len__(self):
        return len(self.df)

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
        if feats.shape[0] > self.max_patches:
            idx_perm = torch.randperm(feats.shape[0])[: self.max_patches]
            feats = feats[idx_perm]

        # TCGA report assets are patient-level (TCGA-XX-XXXX), whereas patch
        # bags normally use a longer slide identifier.  Prefer the exact match.
        case_id = str(row.get("case_id", slide_id[:12]))
        text = self.reports.get(
            slide_id, self.reports.get(
                case_id, self.reports.get(slide_id[:12], self.default_report)))
        if self.require_report and not text.strip():
            raise FileNotFoundError(
                f"WSI-FiVE has no pathology report for slide {slide_id}")
        if self.tokenizer is not None:
            text_ids = self.tokenizer(text, return_tensors="pt",
                                      padding="max_length",
                                      truncation=True,
                                      max_length=self.max_text_len)["input_ids"][0]
            text_output = text_ids
        else:
            text_output = text

        if self.include_metadata:
            return feats, text_output, {
                "slide_id": slide_id, "case_id": case_id,
            }, label
        return feats, text_output, label


def _collate_one(batch):
    if len(batch) != 1:
        raise ValueError("WSI-FiVE uses batch_size=1 for variable-length slides")
    item = batch[0]
    if len(item) == 4:
        feats, report, metadata, label = item
        collated_metadata = {
            "slide_id": [metadata["slide_id"]],
            "case_id": [metadata["case_id"]],
        }
        return (feats.unsqueeze(0), [report], collated_metadata,
                torch.tensor([label]))
    feats, report, label = item
    return feats.unsqueeze(0), [report], torch.tensor([label])


def build_wsi_five_loader(cfg, split: str = "train", shuffle: bool = True):
    from torch.utils.data import DataLoader
    csv_path = os.path.join(cfg["split_dir"], f"{split}.csv")
    ds = WSI_FiVE_Dataset(
        csv_path=csv_path,
        feature_root=cfg.get("feature_root", cfg.get("data_folder_s", cfg.get("data_path", "."))),
        report_csv=cfg.get("report_csv"),
        label_dict=cfg["label_dict"],
        max_patches=cfg.get("num_frames", 2048),
        default_report=cfg.get("default_report"))
    ds.feature_path_column = cfg.get("feature_path_column")
    ds.feature_key = cfg.get("feature_key", "features")
    ds.include_metadata = cfg.get("include_metadata", False)
    ds.require_report = cfg.get("require_report", True)
    if cfg.get("batch_size", 1) != 1:
        raise ValueError("WSI-FiVE unified loader requires batch_size=1")
    return DataLoader(ds, batch_size=1,
                      shuffle=shuffle and split == "train",
                      num_workers=cfg.get("num_workers", 2),
                      collate_fn=_collate_one)
