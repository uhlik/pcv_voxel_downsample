import sys
import os
import unittest
import threading
import time
import numpy as np
import pcv_voxel_downsample

class TestVoxelDownsample(unittest.TestCase):

    def test_empty_cloud(self):
        """1. Empty Input: Ensure an empty point cloud does not crash and returns empty arrays."""
        empty_cloud = np.empty((0, 3), dtype=np.float64)
        points, indices = pcv_voxel_downsample.voxel_downsample(empty_cloud, 0.1)
        self.assertEqual(points.shape[0], 0)
        self.assertEqual(indices.shape[0], 0)

    def test_single_point(self):
        """2. Single Point: Ensure a lone point passes through unchanged."""
        single_point = np.array([[10.0, -5.0, 3.2]], dtype=np.float64)
        points, indices = pcv_voxel_downsample.voxel_downsample(single_point, 0.05)
        
        self.assertEqual(points.shape[0], 1)
        self.assertEqual(indices.shape[0], 1)
        self.assertEqual(indices[0], 0)
        np.testing.assert_array_almost_equal(points, single_point)

    def test_all_points_in_one_voxel(self):
        """3. Complete Collapse: Multiple points inside a single small box resolve to exactly one point."""
        cluster = np.array([
            [0.01, 0.01, 0.01],
            [0.02, 0.02, 0.02],
            [0.03, 0.01, 0.02]
        ], dtype=np.float64)
        points, indices = pcv_voxel_downsample.voxel_downsample(cluster, 1.0)
        
        self.assertEqual(points.shape[0], 1)
        self.assertEqual(indices.shape[0], 1)
        np.testing.assert_array_almost_equal(points, cluster[indices])

    def test_nearest_to_centroid_logic(self):
        """4. Geometry Validation: Ensure the chosen point is strictly the closest to the true centroid."""
        line_points = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0]
        ], dtype=np.float64)
        points, indices = pcv_voxel_downsample.voxel_downsample(line_points, 10.0)
        
        self.assertEqual(points.shape[0], 1)
        self.assertEqual(indices[0], 1) # Must pick index 1 (closest to true centroid)
        np.testing.assert_array_equal(points, [[2.0, 0.0, 0.0]])

    def test_index_alignment_large(self):
        """5. Index Integrity: Verify that reconstructed slices perfectly match downsampled coordinates."""
        np.random.seed(42)
        large_cloud = np.random.uniform(-50, 50, (5000, 3))
        points, indices = pcv_voxel_downsample.voxel_downsample(large_cloud, 2.5)
        reconstructed_points = large_cloud[indices]
        np.testing.assert_array_almost_equal(points, reconstructed_points)

    def test_spatial_limit_overflow_exception(self):
        """6. Safety Constraints: Verify that the runtime exception fires when spatial limits are crossed."""
        wide_cloud = np.array([[0.0, 0.0, 0.0], [2000.0, 0.0, 0.0]], dtype=np.float64)
        micro_voxel = 0.0005 
        with self.assertRaises(RuntimeError) as context:
            pcv_voxel_downsample.voxel_downsample(wide_cloud, micro_voxel)
        self.assertIn("exceed the max 21-bit limit", str(context.exception))

    def test_negative_coordinates(self):
        """7. Origin Shifts: Ensure negative coordinates are handled cleanly without truncation errors."""
        negative_cloud = np.array([[-100.5, -200.5, -300.5], [-100.51, -200.51, -300.51]], dtype=np.float64)
        points, indices = pcv_voxel_downsample.voxel_downsample(negative_cloud, 0.5)
        
        self.assertEqual(points.shape[0], 1)
        np.testing.assert_array_almost_equal(points, negative_cloud[indices])

    def test_invalid_input_shapes(self):
        """8. Input Validation: Check that bad shape allocations throw clean Python ValueErrors."""
        bad_shape_1d = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        bad_shape_2d = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
        with self.assertRaises(ValueError):
            pcv_voxel_downsample.voxel_downsample(bad_shape_1d, 0.1)
        with self.assertRaises(ValueError):
            pcv_voxel_downsample.voxel_downsample(bad_shape_2d, 0.1)

    def test_invalid_voxel_sizes(self):
        """9. Parameter Validation: Ensure negative or zero voxel bounds exit safely."""
        cloud = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
        p1, i1 = pcv_voxel_downsample.voxel_downsample(cloud, 0.0)
        self.assertEqual(p1.shape[0], 0)
        
        p2, i2 = pcv_voxel_downsample.voxel_downsample(cloud, -1.5)
        self.assertEqual(p2.shape[0], 0)

    def test_zero_span_axis_planar(self):
        """10. Zero Span: Ensure flat 2D data planes don't throw divide-by-zero math errors."""
        flat_cloud = np.array([
            [1.0, 2.0, 5.0],
            [10.0, 20.0, 5.0],
            [100.0, 200.0, 5.0]
        ], dtype=np.float64)
        
        points, indices = pcv_voxel_downsample.voxel_downsample(flat_cloud, 0.1)
        self.assertEqual(points.shape[0], 3)
        np.testing.assert_array_almost_equal(points, flat_cloud[indices])

    def test_exact_coordinate_duplicates(self):
        """11. Duplicates: Identical points sitting on top of each other."""
        duplicate_cloud = np.array([
            [10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0]
        ], dtype=np.float64)
        
        points, indices = pcv_voxel_downsample.voxel_downsample(duplicate_cloud, 1.0)
        self.assertEqual(points.shape[0], 1)
        self.assertEqual(indices.shape[0], 1)
        np.testing.assert_array_equal(points, [[10.0, 10.0, 10.0]])

    def test_invalid_data_types(self):
        """12. Type Validation: Ensure passing non-float arrays (like integers) triggers a TypeError."""
        bad_type_cloud = np.array([[1, 2, 3]], dtype=np.int32)
        with self.assertRaises(TypeError) as context:
            pcv_voxel_downsample.voxel_downsample(bad_type_cloud, 0.1)
        self.assertIn("Must be np.float32 or np.float64", str(context.exception))

    def test_float32_precision_preservation(self):
        """13. Dual Precision: Verify that float32 inputs return native float32 downsampled arrays."""
        cloud_f32 = np.array([
            [1.5, 2.5, 3.5],
            [1.51, 2.51, 3.51]
        ], dtype=np.float32)
        
        points, indices = pcv_voxel_downsample.voxel_downsample(cloud_f32, 0.5)
        
        # Verify precision structures are mirrored perfectly
        self.assertEqual(points.dtype, np.float32)
        self.assertEqual(indices.dtype, np.int64)
        self.assertEqual(points.shape[0], 1)

    def test_non_contiguous_slices(self):
        """14. Memory Strides: Ensure sliced or non-contiguous arrays don't corrupt C-pointers."""
        np.random.seed(123)
        large_cloud = np.random.uniform(-10, 10, (100, 3))
        sliced_cloud = large_cloud[::2] # Induces non-contiguous stride tracking
        
        points, indices = pcv_voxel_downsample.voxel_downsample(sliced_cloud, 0.5)
        
        # Sliced sub-selection arrays should mirror mapping references perfectly
        np.testing.assert_array_almost_equal(points, sliced_cloud[indices])

    def test_read_only_arrays(self):
        """15. Immutability: Ensure read-only buffer arrays do not trigger allocation panics."""
        cloud = np.array([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]], dtype=np.float64)
        cloud.flags.writeable = False # Explicit buffer lock
        
        try:
            points, indices = pcv_voxel_downsample.voxel_downsample(cloud, 0.5)
        except Exception as e:
            self.fail(f"Voxel downsample crashed on a read-only NumPy array: {e}")
            
        self.assertEqual(points.shape[0], 1)

    def test_nan_and_inf_handling(self):
        """16. Corrupted Telemetry: Verify behavior when handling NaN or Infinite values."""
        nan_cloud = np.array([[1.0, 2.0, 3.0], [np.nan, 0.0, 1.0]], dtype=np.float64)
        inf_cloud = np.array([[1.0, 2.0, 3.0], [np.inf, 0.0, 1.0]], dtype=np.float64)
        
        with self.assertRaises((ValueError, RuntimeError)):
            pcv_voxel_downsample.voxel_downsample(nan_cloud, 0.5)
            
        with self.assertRaises((ValueError, RuntimeError)):
            pcv_voxel_downsample.voxel_downsample(inf_cloud, 0.5)

    def test_micro_macro_scaling(self):
        """17. Scale Bounds: Test massive and minuscule coordinate structures."""
        # Check extremely small geometric scales
        micro_cloud = np.array([[0.000001, 0.000002, 0.000003], [0.0000015, 0.0000025, 0.0000035]], dtype=np.float64)
        p_micro, _ = pcv_voxel_downsample.voxel_downsample(micro_cloud, 0.0000001)
        self.assertTrue(p_micro.shape[0] > 0)

        # Check highly distant coordinate fields
        macro_cloud = np.array([[100000.0, 100000.0, 100000.0], [100000.5, 100000.5, 100000.5]], dtype=np.float64)
        p_macro, _ = pcv_voxel_downsample.voxel_downsample(macro_cloud, 1.0)
        self.assertEqual(p_macro.shape[0], 1)
    
    def test_collinear_and_coplanar_precision(self):
        """18. Geometric Edge Case: Points perfectly equidistant from a voxel center."""
        # Creates a perfectly symmetric cross. The geometric average centroid is exactly.
        # All points are equidistant (dist_sq = 1.0) from the centroid. 
        # Rust must safely select the first-encountered minimal index without panicking or entering an infinite loop.
        cross_cloud = np.array([
            [ 1.0,  0.0,  0.0],
            [-1.0,  0.0,  0.0],
            [ 0.0,  1.0,  0.0],
            [ 0.0, -1.0,  0.0],
        ], dtype=np.float64)
        
        points, indices = pcv_voxel_downsample.voxel_downsample(cross_cloud, 10.0)
        self.assertEqual(points.shape[0], 1)
        self.assertEqual(indices.shape[0], 1)
        # Verify that the index points to a valid point in our input array
        self.assertIn(indices[0], [0, 1, 2, 3])

    def test_python_list_fallback_conversion(self):
        """19. Serialization Fallback: Passing a raw Python list nested structure."""
        # Ensure our PyO3 wrapper handles safe explicit conversion via np.asarray implicitly.
        list_cloud = [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]]
        points, indices = pcv_voxel_downsample.voxel_downsample(list_cloud, 0.5)
        
        self.assertEqual(points.shape[0], 1)
        self.assertEqual(points.dtype, np.float64) # Default NumPy conversion type

    def test_fortran_contiguous_arrays(self):
        """20. Memory Layout: Column-Major (Fortran-style 'F') array tracking."""
        # F_CONTIGUOUS arrays change stride iteration directions.
        # Rust-NumPy must read rows accurately without reading corrupted/transposed data points.
        np.random.seed(7)
        f_cloud = np.asfortranarray(np.random.uniform(-5, 5, (50, 3)))
        
        points, indices = pcv_voxel_downsample.voxel_downsample(f_cloud, 1.0)
        # Ensure indices slice cleanly back into the original array and perfectly match coordinates
        np.testing.assert_array_almost_equal(points, f_cloud[indices])

    def test_massive_cloud_stress_and_gil(self):
        """21. Threading Stress: Ensure massive data vectors do not overflow stack or lock runtime threads."""
        # Generates a 250,000 point payload to saturate memory parsing buffers.
        np.random.seed(99)
        massive_cloud = np.random.uniform(-100, 100, (250000, 3)).astype(np.float32)
        
        try:
            points, indices = pcv_voxel_downsample.voxel_downsample(massive_cloud, 2.0)
            self.assertTrue(points.shape[0] < 250000)
            self.assertEqual(points.dtype, np.float32)
        except Exception as e:
            self.fail(f"Extension choked or panicked under large data-stream load conditions: {e}")

    def test_exception_type_validation(self):
        """22. Exception Target Accuracy: Ensure PyO3 raises the exact exception types expected by Python callers."""
        bad_shape = np.array([1.0, 2.0, 3.0, 4.0]) # 1D array
        
        # Test that we raise ValueError specifically, not generic RuntimeError or TypeError
        with self.assertRaises(ValueError) as ctx:
            pcv_voxel_downsample.voxel_downsample(bad_shape, 0.1)
        self.assertIn("Input point cloud must have shape", str(ctx.exception))
    
    def test_voxel_downsample_releases_gil(self):
        """Verify that the Rust logic drops the GIL and allows concurrent Python code to run."""
        # 1. Generate a large point cloud to ensure the Rust execution takes visible time
        # num_points = 5_000_000
        num_points = 10_000_000
        np.random.seed(42)
        cloud = np.random.rand(num_points, 3).astype(np.float32) * 100.0
        voxel_size = 0.5

        # Shared execution state flags
        self.bg_thread_running = True
        self.bg_heartbeat_count = 0

        def background_heartbeat():
            """A background job requiring the GIL to increment variables."""
            while self.bg_thread_running:
                _ = 1 + 1
                self.bg_heartbeat_count += 1
                time.sleep(0.01)

        # 2. Spin up the monitoring worker thread
        thread = threading.Thread(target=background_heartbeat, daemon=True)
        thread.start()

        # Allow the thread setup to warm up
        time.sleep(0.05)
        initial_count = self.bg_heartbeat_count

        # 3. Execute the PyO3 wrapper function
        start_time = time.perf_counter()
        points, indices = pcv_voxel_downsample.voxel_downsample(cloud, voxel_size)
        duration = time.perf_counter() - start_time

        # 4. Tear down the monitoring loop
        self.bg_thread_running = False
        thread.join(timeout=1.0)

        actual_ticks = self.bg_heartbeat_count - initial_count

        # Print metrics directly to the console output stream
        print(f"\n[Metric] Rust execution duration: {duration:.4f} seconds")
        print(f"\n[Metric] Concurrent Python heartbeats: {actual_ticks}")

        # 5. Core Assertions
        # Validate data shapes and consistency
        self.assertGreater(len(points), 0, "Downsampled cloud should not be empty")
        self.assertEqual(len(indices), len(points), "Indices mapping size must equal returned points")

        # Validate GIL performance
        # If the GIL was locked, actual_ticks would be 0 or 1.
        self.assertGreater(
            actual_ticks, 
            # 5,
            2, 
            f"The GIL was blocked. The background thread only captured {actual_ticks} loops."
        )


if __name__ == "__main__":
    unittest.main()
