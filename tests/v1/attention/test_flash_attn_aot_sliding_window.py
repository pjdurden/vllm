# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The FA3 AOT schedule is disabled when a builder's layers disagree on the
sliding window. Layers owned by *other* builders (e.g. a speculative drafter's
sliding-window layers alongside a full-attention target) must not count."""

from types import SimpleNamespace

from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionImpl,
    _get_sliding_window_configs,
)

TARGET_LAYERS = ["model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn"]
DRAFT_LAYERS = ["drafter.layers.0.self_attn.attn", "drafter.layers.1.self_attn.attn"]


def _attention_layer(sliding_window: tuple[int, int]) -> Attention:
    impl = object.__new__(FlashAttentionImpl)
    impl.sliding_window = sliding_window
    layer = object.__new__(Attention)
    layer.impl = impl
    return layer


def _vllm_config() -> SimpleNamespace:
    """Full-attention target plus an all-sliding-window DFlash drafter."""
    static_forward_context = {
        name: _attention_layer((-1, -1)) for name in TARGET_LAYERS
    }
    static_forward_context.update(
        {name: _attention_layer((2047, 0)) for name in DRAFT_LAYERS}
    )
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context=static_forward_context
        )
    )


def test_drafter_window_does_not_leak_into_target_builder():
    assert _get_sliding_window_configs(_vllm_config(), TARGET_LAYERS) == {(-1, -1)}


def test_draft_builder_sees_only_its_own_window():
    assert _get_sliding_window_configs(_vllm_config(), DRAFT_LAYERS) == {(2047, 0)}


def test_mixed_windows_within_one_builder_are_still_reported():
    # A builder serving both windows (e.g. hybrid KV cache manager disabled)
    # must still see both, so build() falls back to the dynamic FA3 schedule.
    assert _get_sliding_window_configs(
        _vllm_config(), TARGET_LAYERS + DRAFT_LAYERS
    ) == {(-1, -1), (2047, 0)}
