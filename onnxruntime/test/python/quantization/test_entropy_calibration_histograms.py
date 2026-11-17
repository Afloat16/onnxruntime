# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import unittest

import numpy as np

from onnxruntime.quantization.calibrate import HistogramCollector


class TestEntropyCalibrationHistograms(unittest.TestCase):
    @staticmethod
    def _collector(num_bins, num_quantized_bins):
        return HistogramCollector("entropy", False, num_bins, num_quantized_bins, 99.999, "same")

    def test_thresholds_for_sparse_and_remainder_bins(self):
        cases = (
            ([1] * 8, 4, (-4, 4)),
            ([1] * 9, 5, (-4.5, 4.5)),
            ([0] * 8, 4, (-2, 3)),
            ([0, 0, 0, 1, 3, 0, 0, 0], 4, (-2, 3)),
            ([10, 0, 0, 1, 3, 0, 0, 10], 4, (-4, 4)),
            ([2, 0, 4, 1, 10, 20, 30, 50, 20, 10, 5, 1, 0, 4], 4, (-2, 3)),
            ([2, 0, 4, 1, 10, 20, 30, 50, 20, 10, 5, 1, 0, 4], 5, (-3, 4)),
            ([2, 0, 4, 1, 10, 20, 30, 50, 20, 10, 5, 1, 0, 4], 1, (0, 1)),
        )
        for dtype in (np.float16, np.float32):
            for counts, num_quantized_bins, expected in cases:
                with self.subTest(dtype=dtype, counts=counts, num_quantized_bins=num_quantized_bins):
                    hist = np.array(counts, dtype=np.int64)
                    edges = np.linspace(-len(hist) / 2, len(hist) / 2, len(hist) + 1, dtype=dtype)
                    original_hist = hist.copy()
                    original_edges = edges.copy()
                    histogram = (hist, edges, edges[0], edges[-1], np.array(abs(edges[-1]), dtype=dtype))
                    actual = self._collector(len(hist), num_quantized_bins).get_entropy_threshold(
                        histogram, num_quantized_bins
                    )
                    np.testing.assert_array_equal(actual, np.array(expected, dtype=dtype))
                    self.assertTrue(all(value.dtype == dtype for value in actual))
                    np.testing.assert_array_equal(hist, original_hist)
                    np.testing.assert_array_equal(edges, original_edges)

    def test_clamps_thresholds_to_observed_range(self):
        hist = np.array([0, 0, 0, 1, 3, 0, 0, 0], dtype=np.int64)
        edges = np.linspace(-4, 4, 9, dtype=np.float32)
        histogram = (hist, edges, np.float32(-0.5), np.float32(0.5), np.float32(4))
        actual = self._collector(8, 4).get_entropy_threshold(histogram, 4)
        np.testing.assert_array_equal(actual, np.array([-0.5, 0.5], dtype=np.float32))

    def test_default_bin_count_preserves_full_range(self):
        collector = self._collector(128, 128)
        values = np.linspace(-1, 2, 1024, dtype=np.float32)
        collector.collect({"values": values})
        result = collector.compute_collection_result()["values"]
        np.testing.assert_array_equal(result[:2], np.array([-1, 2], dtype=np.float32))
        self.assertEqual(result[2].sum(), values.size)


if __name__ == "__main__":
    unittest.main()
