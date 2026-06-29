import time
import threading
import unittest
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import pcv_voxel_downsample


class TestVoxelDownsampleGIL(unittest.TestCase):

    def test_voxel_downsample_releases_gil(self):
        """Verify that the C++ logic drops the GIL and allows concurrent Python code to run."""
        # 1. Generate a large point cloud to ensure the C++ execution takes visible time
        num_points = 5_000_000
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

        # 3. Execute the Cython wrapper function
        start_time = time.perf_counter()
        points, indices = pcv_voxel_downsample.voxel_downsample(cloud, voxel_size)
        duration = time.perf_counter() - start_time

        # 4. Tear down the monitoring loop
        self.bg_thread_running = False
        thread.join(timeout=1.0)

        actual_ticks = self.bg_heartbeat_count - initial_count

        # Print metrics directly to the console output stream
        print(f"\n[Metric] C++ execution duration: {duration:.4f} seconds")
        print(f"\n[Metric] Concurrent Python heartbeats: {actual_ticks}")

        # 5. Core Assertions
        # Validate data shapes and consistency
        self.assertGreater(len(points), 0, "Downsampled cloud should not be empty")
        self.assertEqual(len(indices), len(points), "Indices mapping size must equal returned points")

        # Validate GIL performance
        # If the GIL was locked, actual_ticks would be 0 or 1.
        self.assertGreater(
            actual_ticks, 
            5, 
            f"The GIL was blocked. The background thread only captured {actual_ticks} loops."
        )

if __name__ == "__main__":
    unittest.main()
