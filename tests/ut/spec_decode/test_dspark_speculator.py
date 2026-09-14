#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Unit tests for ``AscendDSparkSpeculator.load_draft_model`` fc rotation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

from vllm_ascend.worker.v2.aclgraph_utils import ModelWithContext
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import (
    AscendDSparkSpeculator,
)

_HIDDEN = 8
_FC_IN = 5 * _HIDDEN  # concatenated aux hidden states
# Patch where load_draft_model looks it up (the speculator module binding).
_ROT_MATRIX = "vllm_ascend.worker.v2.spec_decode.dspark.speculator.get_rotation_matrix"


def _spec(vllm_config: SimpleNamespace) -> AscendDSparkSpeculator:
    """Bypass the heavy ``__init__``; ``load_draft_model`` only reads
    ``self.vllm_config`` and the patched parent call."""
    spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    spec.vllm_config = vllm_config
    return spec


def _fake_draft() -> SimpleNamespace:
    fc = torch.nn.Linear(_FC_IN, _HIDDEN, bias=False)
    with torch.no_grad():
        fc.weight.copy_(torch.randn_like(fc.weight))
    return SimpleNamespace(model=SimpleNamespace(fc=fc))


def _quarot_config() -> SimpleNamespace:
    quarot = {"rotation_map": {"global_rotation": "x.safetensors"}}
    return SimpleNamespace(
        quant_config=SimpleNamespace(quant_description={"optional": {"quarot": quarot}}),
        model_config=SimpleNamespace(model="/fake"),
    )


def _bf16_config() -> SimpleNamespace:
    return SimpleNamespace(quant_config=None, model_config=SimpleNamespace())


def _no_call(*args, **kwargs):
    raise AssertionError("get_rotation_matrix must not be called without a rotation path")


class TestLoadDraftModel:
    """``load_draft_model`` rotates fc for a QuaRot target and is a no-op otherwise."""

    @pytest.fixture
    def captured(self, monkeypatch):
        """Stub the heavy parent ``load_draft_model`` to return a fake draft and
        snapshot its fc weight before the override mutates it in place."""
        out: dict = {}

        def _load(self, target_model, target_attn_layer_names):
            draft = _fake_draft()
            out["before"] = draft.model.fc.weight.data.clone()
            out["draft"] = draft
            return draft

        monkeypatch.setattr(DSparkSpeculator, "load_draft_model", _load)
        return out

    def test_rotates_fc_for_quarot_target(self, captured, monkeypatch):
        # R = 2*I -> W @ R == 2*W, an expectation independent of process_weight.
        monkeypatch.setattr(_ROT_MATRIX, lambda path: torch.eye(_HIDDEN) * 2.0)
        draft = _spec(_quarot_config()).load_draft_model(MagicMock(), set())
        before = captured["before"]
        assert draft is captured["draft"]
        assert torch.allclose(draft.model.fc.weight.data, 2.0 * before, atol=1e-6)
        assert not torch.allclose(draft.model.fc.weight.data, before)

    def test_noop_for_bf16_target(self, captured, monkeypatch):
        monkeypatch.setattr(_ROT_MATRIX, _no_call)
        draft = _spec(_bf16_config()).load_draft_model(MagicMock(), set())
        assert torch.equal(draft.model.fc.weight.data, captured["before"])


class _SparseProjectionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.markov_inputs: list[torch.Tensor] = []

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def compute_draft_topk(self, hidden_states, k):
        assert hidden_states.shape == (2, 1)
        assert k == 2
        return (
            torch.tensor([[2, 3], [1, 9]], dtype=torch.int64),
            torch.zeros(2, 2),
        )

    def markov_embed(self, token_ids):
        self.markov_inputs.append(token_ids.clone())
        return token_ids.float().unsqueeze(-1)

    def score_draft_candidates(self, markov_embed, values, token_ids):
        return values - (token_ids.float() - markov_embed).abs()

    def map_draft_to_target(self, draft_ids):
        return draft_ids


def _topk_speculator(model):
    spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    spec.model = model
    spec.draft_logits = None
    spec._draft_topk = 2
    spec.num_speculative_steps = 2
    spec.sample_indices = torch.tensor([0, 1], dtype=torch.int64)
    spec._anchor_idx = torch.tensor([0], dtype=torch.int64)
    spec.input_buffers = SimpleNamespace(input_ids=torch.tensor([8]))
    spec.draft_tokens = torch.zeros(1, 2, dtype=torch.int64)
    spec.enable_adaptive_verification = False
    return spec


def test_sparse_topk_uses_each_previous_sampled_token():
    model = _SparseProjectionModel()
    spec = _topk_speculator(model)

    spec._sample_sequential_topk(1, torch.zeros(2, 1))

    assert torch.equal(spec.draft_tokens, torch.tensor([[3, 1]]))
    assert len(model.markov_inputs) == 2
    assert torch.equal(model.markov_inputs[0], torch.tensor([8]))
    assert torch.equal(model.markov_inputs[1], torch.tensor([3]))


def test_aclgraph_wrapper_forwards_sparse_projection_methods():
    model = _SparseProjectionModel()
    wrapped = ModelWithContext(model, is_draft_model=True)
    hidden_states = torch.zeros(2, 1)
    token_ids, values = wrapped.compute_draft_topk(hidden_states, 2)

    assert token_ids.shape == values.shape == (2, 2)
    scores = wrapped.score_draft_candidates(
        torch.tensor([[3.0], [8.0]]),
        values,
        token_ids,
    )
    assert scores.shape == (2, 2)
