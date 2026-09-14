from types import SimpleNamespace

import torch
from torch import nn

from vllm_ascend.models.deepseek_v4 import dspark as deepseek_v4_dspark
from vllm_ascend.models.deepseek_v4 import model as deepseek_v4


class StubModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


def test_dspark_markov_topk_scores_match_dense_projection():
    batch_size, rank, vocab_size, topk = 3, 5, 17, 4
    head = deepseek_v4_dspark.DSparkMarkovHead.__new__(
        deepseek_v4_dspark.DSparkMarkovHead
    )
    nn.Module.__init__(head)
    head.markov_w2 = nn.Linear(rank, vocab_size, bias=False)

    markov_embed = torch.randn(batch_size, rank)
    base_logits = torch.randn(batch_size, vocab_size)
    token_ids = torch.randint(0, vocab_size, (batch_size, topk))
    base_values = base_logits.gather(1, token_ids)
    scale = 1.0

    dense_scores = base_logits + head.markov_w2(markov_embed)
    expected = dense_scores.gather(1, token_ids)
    actual = head.score_gathered(
        markov_embed,
        base_values,
        token_ids,
        scale,
    )

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_dspark_compute_draft_topk_merges_tp_candidates(monkeypatch):
    model_cls = deepseek_v4_dspark.DSparkDeepseekV4ForCausalLM
    model = model_cls.__new__(model_cls)
    nn.Module.__init__(model)

    local_logits = torch.tensor([[4.0, 2.0]])
    model.model = SimpleNamespace(norm=lambda hidden_states: hidden_states)
    model.lm_head = SimpleNamespace(
        tp_size=2,
        shard_indices=SimpleNamespace(
            num_org_vocab_padding=0,
            org_vocab_start_index=10,
        ),
    )
    model.logits_processor = SimpleNamespace(
        scale=1.0,
        soft_cap=None,
        _apply_head=lambda lm_head, hidden_states, embedding_bias: local_logits,
    )

    def fake_all_gather(tensor, dim):
        assert dim == -1
        remote = (
            torch.tensor([[3.0, 1.0]])
            if tensor.is_floating_point()
            else torch.tensor([[20, 21]], dtype=torch.int64)
        )
        return torch.cat((tensor, remote), dim=-1)

    monkeypatch.setattr(
        deepseek_v4_dspark,
        "tensor_model_parallel_all_gather",
        fake_all_gather,
    )

    token_ids, values = model.compute_draft_topk(torch.zeros(1, 2), k=3)

    assert torch.equal(token_ids, torch.tensor([[10, 20, 11]]))
    assert torch.equal(values, torch.tensor([[4.0, 3.0, 2.0]]))


def test_routed_moe_receives_configured_swiglu_limit(monkeypatch):
    fused_moe_kwargs = {}

    class StubFusedMoEFactory(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            fused_moe_kwargs.update(kwargs)

    monkeypatch.setattr(deepseek_v4, "FusedMoEFactory", StubFusedMoEFactory)
    monkeypatch.setattr(deepseek_v4, "ReplicatedLinear", StubModule)
    monkeypatch.setattr(deepseek_v4, "DeepseekV2MLP", StubModule)
    monkeypatch.setattr(deepseek_v4, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(deepseek_v4, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        deepseek_v4,
        "get_ep_group",
        lambda: SimpleNamespace(device_group=SimpleNamespace(size=lambda: 1), rank_in_group=0),
    )
    monkeypatch.setattr(deepseek_v4, "get_ascend_config", lambda: SimpleNamespace(mix_placement=False))
    monkeypatch.setattr(deepseek_v4.rocm_aiter_ops, "is_fused_moe_enabled", lambda: False)
    monkeypatch.setattr(deepseek_v4.rocm_aiter_ops, "is_fusion_moe_shared_experts_enabled", lambda: False)

    config = SimpleNamespace(
        hidden_act="silu",
        hidden_size=16,
        moe_intermediate_size=8,
        n_group=1,
        n_routed_experts=8,
        n_shared_experts=1,
        norm_topk_prob=True,
        num_experts_per_tok=2,
        num_hash_layers=0,
        routed_scaling_factor=2.5,
        scoring_func="sigmoid",
        swiglu_limit=10.0,
        topk_group=1,
    )
    parallel_config = SimpleNamespace(
        enable_eplb=False,
        eplb_config=SimpleNamespace(num_redundant_experts=0),
        use_sequence_parallel_moe=False,
    )

    deepseek_v4.DeepseekV4MoE(config, parallel_config, prefix="model.layers.1.mlp")

    assert fused_moe_kwargs["swiglu_limit"] == config.swiglu_limit
