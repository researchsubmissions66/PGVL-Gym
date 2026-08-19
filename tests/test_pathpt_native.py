from __future__ import annotations

import torch
import torch.nn as nn

from methods.pathpt.native import (
    generate_patch_targets,
    select_prompt_embedding,
    vote_slide_probabilities,
)
from methods.pathpt.prompts import resolve_prompt_bank


def test_pathpt_implementation_provenance_tracks_the_active_mode():
    from common.method_provenance import method_provenance

    assert method_provenance("pathpt").upstream_fidelity == "partial"
    native = method_provenance("pathpt", _brca_cfg())
    assert native.implementation == "vendored_native_objective"
    assert native.upstream_fidelity == "upstream"
    camelyon = method_provenance(
        "pathpt", {**_brca_cfg(), "task": "camelyon16"})
    assert camelyon.upstream_fidelity == "partial"


def _brca_cfg(**updates):
    cfg = {
        "method": "pathpt",
        "training_mode": "upstream_patch_ssl",
        "task": "brca",
        "backbone": "conch",
        "feature_dim": 512,
        "n_classes": 2,
        "classnames": [
            "invasive ductal carcinoma", "invasive lobular carcinoma"],
        "label_dict": {"IDC": 0, "ILC": 1},
        "epochs": 20,
    }
    cfg.update(updates)
    return cfg


def test_pathpt_prompt_banks_distinguish_upstream_and_generated_content():
    brca = resolve_prompt_bank(_brca_cfg())
    assert brca.provenance == "upstream"
    assert brca.synthetic_normal is True
    assert len(brca.class_synonyms) == 3
    assert [len(rows) for rows in brca.prompts] == [66, 44, 44]

    nsclc = resolve_prompt_bank({
        **_brca_cfg(),
        "task": "nsclc",
        "classnames": ["lung adenocarcinoma", "lung squamous cell carcinoma"],
        "label_dict": {"LUAD": 0, "LUSC": 1},
    })
    assert nsclc.provenance == "generated"
    assert nsclc.synthetic_normal is True


def test_pathpt_preserves_upstream_camelyon_prompt_defect_and_discloses_binary_mode():
    bank = resolve_prompt_bank({
        **_brca_cfg(),
        "task": "camelyon16",
        "classnames": ["normal lymph node", "metastatic lymph node"],
        "label_dict": {"normal": 0, "tumor": 1},
    })
    assert bank.synthetic_normal is False
    assert bank.provenance == "upstream"
    assert "non-cancerous tissuenormal breast tissue" in bank.class_synonyms[0]
    assert "binary supervision is a local adaptation" in bank.note


def test_pathpt_patch_targets_encode_normal_subtype_and_candidate_rows():
    scores = torch.tensor([
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.2],
        [0.1, 0.2, 0.7],
    ])
    targets = generate_patch_targets(
        scores, torch.tensor([0]), synthetic_normal=True)
    assert targets.tolist() == [0, 1, -1]


def test_pathpt_vote_excludes_normal_and_uses_tumour_fallback_on_ties():
    scores = torch.tensor([
        [0.9, 0.08, 0.02],
        [0.9, 0.03, 0.07],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
    ])
    probabilities = vote_slide_probabilities(
        scores, n_classes=2, synthetic_normal=True)
    torch.testing.assert_close(probabilities, torch.tensor([[0.5, 0.5]]))


def test_pathpt_binary_vote_keeps_normal_as_a_slide_class():
    scores = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8]])
    probabilities = vote_slide_probabilities(
        scores, n_classes=2, synthetic_normal=False)
    torch.testing.assert_close(probabilities, torch.tensor([[2 / 3, 1 / 3]]))


def test_pathpt_prompt_selection_is_deterministic_and_train_loader_scoped():
    vectors = {
        "n0": [1.0, 0.0, 0.0], "n1": [0.9, 0.1, 0.0],
        "a0": [0.0, 1.0, 0.0], "a1": [0.0, 0.9, 0.1],
        "b0": [0.0, 0.0, 1.0], "b1": [0.1, 0.0, 0.9],
    }

    def encode_text(texts):
        return torch.tensor([vectors[text] for text in texts])

    loader = [
        (torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.8, 0.2]]]),
         torch.zeros(1, 2), torch.tensor([0])),
        (torch.tensor([[[0.0, 0.0, 1.0], [0.2, 0.0, 0.8]]]),
         torch.zeros(1, 2), torch.tensor([1])),
    ]
    kwargs = dict(
        n_slide_classes=2, synthetic_normal=True,
        device=torch.device("cpu"),
        classifier_count=8, select_count=4, top_patches=2,
        classifier_batch_size=3, text_batch_size=2,
    )
    first = select_prompt_embedding(
        encode_text, loader, (("n0", "n1"), ("a0", "a1"), ("b0", "b1")),
        **kwargs)
    second = select_prompt_embedding(
        encode_text, loader, (("n0", "n1"), ("a0", "a1"), ("b0", "b1")),
        **kwargs)
    torch.testing.assert_close(first.embedding, second.embedding)
    assert first.top_classifier_indices == second.top_classifier_indices
    assert first.embedding.shape == (3, 3)


def test_pathpt_native_adapter_uses_patch_ssl_and_returns_slide_logits():
    from methods.pathpt.adapter import PathPTMethod

    method = PathPTMethod(_brca_cfg(), device="cpu")
    method.prompt_bank = resolve_prompt_bank(method.cfg)

    class DummyPathPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_logits = nn.Parameter(torch.tensor([
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
            ]))
            self.register_buffer(
                "pathpt_selected_prompt_embedding", torch.eye(3))

        def forward(self, _features):
            return None, self.patch_logits.softmax(dim=-1)

    model = DummyPathPT()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    features = torch.eye(3).unsqueeze(0)
    output = method.train_step(
        (features, torch.zeros(1, 3, 2), torch.tensor([0])),
        model, optimizer, nn.CrossEntropyLoss())
    assert output["loss"] > 0
    assert output["logits"].shape == (1, 2)
    assert torch.isfinite(output["logits"]).all()
