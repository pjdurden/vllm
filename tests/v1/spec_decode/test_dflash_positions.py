# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash input positions.

DFlash turns the positions it is handed into KV cache slots, so they have to be
each token's index within its sequence. Target models with multi-dimensional
RoPE (M-RoPE, XD-RoPE) hand the proposer rope positions instead, which diverge
from the sequence index once multimodal inputs are in the prompt.
"""

import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.dflash import sequence_positions


def _cad(num_computed: list[int], num_scheduled: list[int]) -> CommonAttentionMetadata:
    query_start_loc = torch.tensor([0, *num_scheduled], dtype=torch.int32).cumsum(0)
    seq_lens = torch.tensor(num_computed, dtype=torch.int32) + torch.tensor(
        num_scheduled, dtype=torch.int32
    )
    num_tokens = int(query_start_loc[-1])
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc.to(torch.int32),
        query_start_loc_cpu=query_start_loc.to(torch.int32),
        seq_lens=seq_lens,
        num_reqs=len(num_scheduled),
        num_actual_tokens=num_tokens,
        max_query_len=max(num_scheduled),
        max_seq_len=int(seq_lens.max()),
        block_table_tensor=torch.zeros(len(num_scheduled), 4, dtype=torch.int32),
        slot_mapping=torch.zeros(num_tokens, dtype=torch.int64),
    )


def test_sequence_positions_matches_computed_plus_offset():
    """Mixed decode + chunked prefill batch, including a padded request."""
    num_computed = [100, 0, 48, 0]
    num_scheduled = [16, 5, 16, 0]
    cad = _cad(num_computed, num_scheduled)

    positions = sequence_positions(
        cad, cad.num_actual_tokens, torch.arange(64, dtype=torch.int32)
    )

    expected = torch.tensor(
        [c + i for c, n in zip(num_computed, num_scheduled) for i in range(n)],
        dtype=torch.int64,
    )
    assert positions.dtype == torch.int64
    assert torch.equal(positions, expected)


def test_sequence_positions_ignores_rope_positions():
    """M-RoPE compresses image spans, so rope positions trail the token index.

    The proposer must not feed those to the slot-mapping kernel: two tokens
    sharing a rope position would map onto the same KV cache slot.
    """
    cad = _cad(num_computed=[8], num_scheduled=[4])
    # What an M-RoPE target would hand the proposer for the same batch.
    mrope_positions = torch.tensor([[3, 3, 3, 4]], dtype=torch.int64)

    positions = sequence_positions(
        cad, cad.num_actual_tokens, torch.arange(16, dtype=torch.int32)
    )

    assert torch.equal(positions, torch.tensor([8, 9, 10, 11], dtype=torch.int64))
    assert not torch.equal(positions, mrope_positions[0])
