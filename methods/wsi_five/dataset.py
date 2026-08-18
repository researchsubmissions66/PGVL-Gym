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
            # `answer` first: WSI-FiVE's own supervision text is the six
            # semicolon-separated responses to its clinical questions, which is
            # what the release's GPT sheets carry. A raw multi-page `text`
            # report is a fallback, not an equivalent.
            txt_col = next((column for column in ("answer", "report", "text")
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
        # The fusion transformer positions patches by their index within the
        # slide and masks padding per slide, so the sampled indices and the
        # slide's full patch count are part of the input, not bookkeeping.
        patch_pub_cnt = int(feats.shape[0])
        if feats.shape[0] > self.max_patches:
            keep = torch.randperm(feats.shape[0])[: self.max_patches].sort().values
            feats = feats[keep]
            patch_inds = keep
        else:
            patch_inds = torch.arange(feats.shape[0])

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
