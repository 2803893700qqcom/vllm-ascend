# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

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
