import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import time
import numpy as np
import pcv_voxel_downsample

def run_benchmark():
    num_points = 10 ** 6
    voxel_sizes = [0.1, 0.5, 1.0, 5.0]
    num_runs = 5
    
    np.random.seed(123)
    base_cloud = np.random.uniform(-50.0, 50.0, (num_points, 3))
    
    cloud_f64 = base_cloud.astype(np.float64)
    cloud_f32 = base_cloud.astype(np.float32)
    
    print(f"{'Voxel Size':<12} | {'Data Type':<10} | {'Input Points':<14} | {'Output Points':<14} | {'Avg Time (ms)':<15}")
    print("-" * 80)
    
    for voxel_size in voxel_sizes:
        for dtype_str, cloud in [("float32", cloud_f32), ("float64", cloud_f64)]:
            _, _ = pcv_voxel_downsample.voxel_downsample(cloud, voxel_size)
            
            start_time = time.perf_counter()
            for _ in range(num_runs):
                res_points, res_indices = pcv_voxel_downsample.voxel_downsample(cloud, voxel_size)
            end_time = time.perf_counter()
            
            avg_time_ms = ((end_time - start_time) / num_runs) * 1000
            input_count = cloud.shape[0]
            output_count = res_points.shape[0]
            
            print(f"{voxel_size:<12} | {dtype_str:<10} | {input_count:<14,} | {output_count:<14,} | {avg_time_ms:<15.2f}")
        print("-" * 80)

if __name__ == "__main__":
    run_benchmark()
