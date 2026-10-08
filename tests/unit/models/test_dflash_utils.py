"""Unit tests for get_base_indices_for_anchored_blocks and select_anchors."""

import torch

from speculators.models.dflash.utils import (
    get_base_indices_for_anchored_blocks,
    select_anchors,
)


class TestGetBaseIndicesForAnchoredBlocks:
    def test_single_anchor(self):
        anchor_positions = torch.tensor([[3]])
        result = get_base_indices_for_anchored_blocks(anchor_positions, block_size=4)
        expected = torch.tensor([3, 4, 5, 6])
        assert torch.equal(result, expected)

    def test_multiple_anchors(self):
        anchor_positions = torch.tensor([[0, 5, 10]])
        result = get_base_indices_for_anchored_blocks(anchor_positions, block_size=3)
        expected = torch.tensor([0, 1, 2, 5, 6, 7, 10, 11, 12])
        assert torch.equal(result, expected)

    def test_block_size_one(self):
        anchor_positions = torch.tensor([[2, 7, 9]])
        result = get_base_indices_for_anchored_blocks(anchor_positions, block_size=1)
        expected = torch.tensor([2, 7, 9])
        assert torch.equal(result, expected)

    def test_1d_input(self):
        anchor_positions = torch.tensor([1, 4])
        result = get_base_indices_for_anchored_blocks(anchor_positions, block_size=2)
        expected = torch.tensor([1, 2, 4, 5])
        assert torch.equal(result, expected)

    def test_output_shape(self):
        num_anchors = 5
        block_size = 4
        anchor_positions = torch.tensor([[0, 3, 6, 9, 12]])
        result = get_base_indices_for_anchored_blocks(
            anchor_positions, block_size=block_size
        )
        assert result.shape == (num_anchors * block_size,)

    def test_output_dtype_is_long(self):
        anchor_positions = torch.tensor([[2.0, 5.0]])
        result = get_base_indices_for_anchored_blocks(anchor_positions, block_size=2)
        assert result.dtype == torch.long


class TestSelectAnchors:
    def test_sampled_anchors_are_sorted(self):
        # Anchors are returned sorted by position so the draft blocks form
        # contiguous flex-attention blocks (fast path) instead of scattered ones.
        torch.manual_seed(0)
        loss_mask = torch.ones(1, 64)
        anchors, anchor_valid = select_anchors(loss_mask, num_anchors=8, block_size=4)
        selected = anchors[anchor_valid]
        assert torch.equal(selected, torch.sort(selected).values)

    def test_padding_repeats_real_anchors_but_remains_loss_invalid(self):
        torch.manual_seed(0)
        loss_mask = torch.tensor([[0, 1, 0, 0, 0, 0]])

        anchors, anchor_valid = select_anchors(
            loss_mask, num_anchors=4, block_size=2
        )

        assert torch.equal(anchors, torch.tensor([1, 1, 1, 1]))
        assert torch.equal(
            anchor_valid, torch.tensor([True, False, False, False])
        )

    def test_no_valid_anchors_remain_safely_invalid(self):
        loss_mask = torch.zeros(1, 6)

        anchors, anchor_valid = select_anchors(
            loss_mask, num_anchors=4, block_size=2
        )

        assert torch.equal(anchors, torch.zeros(4, dtype=torch.long))
        assert not bool(anchor_valid.any())
