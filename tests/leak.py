import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', ))

import numpy as np
import pcv_voxel_downsample

cloud = np.random.rand(10 ** 6, 3).astype(np.float32)

for i in range(10 ** 4):
    print(i)
    _ = pcv_voxel_downsample.voxel_downsample(cloud, 0.05)
print("done")
