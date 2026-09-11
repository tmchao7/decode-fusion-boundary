import unittest

from nanovllm.backends.megakernel import PagedKVCacheSpec


class PagedKVCacheTests(unittest.TestCase):
    def setUp(self):
        self.spec = PagedKVCacheSpec(
            num_layers=2,
            num_blocks=16,
            block_size=4,
            num_kv_heads=2,
            head_dim=8,
            dtype_itemsize=2,
        )

    def test_describes_nanovllm_cache_shape_and_strides(self):
        self.assertEqual(
            self.spec.shape,
            (2, 2, 16, 4, 2, 8),
        )
        self.assertEqual(self.spec.token_stride, 2 * 8)
        self.assertEqual(self.spec.block_stride, 4 * 2 * 8)
        self.assertEqual(self.spec.layer_stride, 16 * 4 * 2 * 8)
        self.assertEqual(self.spec.total_bytes, 2 * 2 * 16 * 4 * 2 * 8 * 2)

    def test_maps_logical_position_to_physical_slot(self):
        block_table = [7, 3, 12]
        self.assertEqual(
            self.spec.physical_slot(block_table, 0), (7, 0)
        )
        self.assertEqual(
            self.spec.physical_slot(block_table, 5), (3, 1)
        )
        self.assertEqual(
            self.spec.physical_slot(block_table, 11), (12, 3)
        )

    def test_computes_flat_element_offset(self):
        block_table = [7, 3, 12]
        # layer=1, logical position=5 -> block 3, offset 1, kv_head=1, dim=2.
        expected = (
            1 * self.spec.layer_stride
            + 3 * self.spec.block_stride
            + 1 * self.spec.token_stride
            + 1 * self.spec.head_stride
            + 2
        )
        self.assertEqual(
            self.spec.element_offset(1, block_table, 5, 1, 2), expected
        )

    def test_rejects_invalid_block_table_and_positions(self):
        with self.assertRaisesRegex(ValueError, "block table"):
            self.spec.physical_slot([], 0)
        with self.assertRaisesRegex(IndexError, "position"):
            self.spec.physical_slot([7], 4)
        with self.assertRaisesRegex(ValueError, "physical block"):
            self.spec.physical_slot([-1], 0)

    def test_supports_cache_capacity_beyond_a_single_prompt(self):
        spec = PagedKVCacheSpec(
            num_layers=28,
            num_blocks=32,
            block_size=256,
            num_kv_heads=8,
            head_dim=128,
            dtype_itemsize=2,
        )
        self.assertEqual(spec.shape, (2, 28, 32, 256, 8, 128))
        self.assertEqual(spec.physical_slot([11, 4], 257), (4, 1))

    def test_rejects_invalid_dimensions(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            PagedKVCacheSpec(0, 1, 4, 1, 8, 2)
        with self.assertRaisesRegex(ValueError, "block_size"):
            PagedKVCacheSpec(1, 1, 0, 1, 8, 2)


if __name__ == "__main__":
    unittest.main()
