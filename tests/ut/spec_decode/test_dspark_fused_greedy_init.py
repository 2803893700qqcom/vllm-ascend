# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

from vllm_ascend.worker.v2.spec_decode.dspark.greedy import (
    sample_greedy_markov,
    scratch_shape,
)
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator


@pytest.mark.parametrize(
    ("enabled", "topk", "probabilistic", "has_scratch"),
    [
        (False, None, False, False),
        (True, None, False, True),
        (True, 32, False, False),
        (True, None, True, False),
    ],
)
def test_fused_greedy_initialization_paths(enabled, topk, probabilistic, has_scratch):
    config = SimpleNamespace(
        additional_config={"enable_dspark_fused_greedy": enabled},
    )
    draft_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="qwen3_dspark" if topk or probabilistic else "deepseek_v4",
            vocab_size=4097,
            draft_vocab_size=None,
        )
    )

    def init_parent(self, vllm_config, device):
        self.draft_model_config = draft_config
        self._draft_topk = topk
        self.draft_logits = torch.empty(1) if probabilistic else None
        self.max_num_reqs = 2

    with patch.object(DSparkSpeculator, "__init__", init_parent):
        speculator = AscendDSparkSpeculator(config, torch.device("cpu"))

    assert (speculator._greedy_partial_values is not None) == has_scratch
    assert (speculator._greedy_partial_indices is not None) == has_scratch
    if has_scratch:
        assert speculator._greedy_partial_values.shape == (2, 3)
        assert speculator._greedy_partial_indices.shape == (2, 3)


@pytest.mark.parametrize("legacy_upstream", [True, False])
def test_fused_greedy_sequential_sampling(legacy_upstream):
    speculator = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    speculator._enable_dspark_fused_greedy = True
    speculator.draft_logits = None
    speculator._draft_topk = None
    speculator.enable_adaptive_verification = True
    if not legacy_upstream:
        speculator.use_acceptance_estimator = False
        speculator.draft_watermarker = None
        speculator.use_confidence_head = True

    speculator.model = SimpleNamespace(
        draft_id_to_target_id=None,
        compute_draft_logits=MagicMock(
            return_value=torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        ),
        markov_embed=MagicMock(side_effect=lambda ids: ids.float().unsqueeze(-1)),
        markov_bias=MagicMock(side_effect=lambda embed: torch.zeros(1, 3)),
        compute_confidence=MagicMock(return_value=torch.tensor([0.7, 0.8])),
    )
    speculator.num_speculative_steps = 2
    speculator.sample_indices = torch.arange(2)
    speculator.sample_idx_mapping = torch.arange(2)
    speculator.sample_pos = torch.arange(2)
    speculator._anchor_idx = torch.tensor([0])
    speculator.input_buffers = SimpleNamespace(input_ids=torch.tensor([0]))
    speculator.draft_tokens = torch.full((1, 2), -1, dtype=torch.int64)
    speculator.draft_token_confidence_probs = torch.empty((1, 2))
    speculator._greedy_partial_values = torch.empty((1, 1))
    speculator._greedy_partial_indices = torch.empty((1, 1), dtype=torch.int32)

    def sample(base, bias, output, partial_values, partial_indices):
        output.copy_((base + bias).argmax(dim=-1))

    with patch(
        "vllm_ascend.worker.v2.spec_decode.dspark.speculator.sample_greedy_markov",
        side_effect=sample,
    ) as fused:
        speculator._sample_sequential(1, torch.zeros(2, 1))

    assert fused.call_count == 2
    assert speculator.draft_tokens.tolist() == [[1, 2]]
    assert speculator.model.markov_embed.call_args_list[1].args[0].tolist() == [1]
    assert torch.allclose(
        speculator.draft_token_confidence_probs, torch.tensor([[0.7, 0.8]])
    )


@pytest.mark.parametrize("reason", ["acceptance_estimator", "watermarker"])
def test_fused_greedy_preserves_new_upstream_logits_consumers(reason):
    speculator = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    speculator._enable_dspark_fused_greedy = True
    speculator.draft_logits = None
    speculator._draft_topk = None
    speculator.model = SimpleNamespace(draft_id_to_target_id=None)
    speculator.use_acceptance_estimator = reason == "acceptance_estimator"
    speculator.draft_watermarker = object() if reason == "watermarker" else None

    with patch.object(DSparkSpeculator, "_sample_sequential") as original:
        speculator._sample_sequential(1, torch.zeros(2, 1))

    original.assert_called_once()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vocab", [1, 2047, 2048, 2049, 129280])
@pytest.mark.parametrize("rows", [1, 4])
def test_fused_greedy_matches_materialized_add_on_npu(dtype, vocab, rows):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU is required")

    device = torch.device("npu")
    base = torch.randn((rows, vocab), dtype=dtype, device=device)
    bias = torch.randn((rows, vocab), dtype=dtype, device=device)
    output_storage = torch.full((rows, 3), -1, dtype=torch.int64, device=device)
    output = output_storage[:, 1]
    values = torch.empty(scratch_shape(rows, vocab), dtype=torch.float32, device=device)
    indices = torch.empty(scratch_shape(rows, vocab), dtype=torch.int32, device=device)

    sample_greedy_markov(base, bias, output, values, indices)

    assert torch.equal(output, (base + bias).argmax(dim=-1))
    assert torch.all(output_storage[:, [0, 2]] == -1)


def test_fused_greedy_special_values_on_npu():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU is required")

    device = torch.device("npu")
    base = torch.full((3, 2049), float("-inf"), device=device)
    bias = torch.zeros_like(base)
    base[0, 7] = float("nan")
    base[1, 1] = 3
    base[1, 3] = 3
    output = torch.empty(3, dtype=torch.int64, device=device)
    values = torch.empty(scratch_shape(3, 2049), dtype=torch.float32, device=device)
    indices = torch.empty(scratch_shape(3, 2049), dtype=torch.int32, device=device)

    sample_greedy_markov(base, bias, output, values, indices)

    assert torch.equal(output, (base + bias).argmax(dim=-1))
