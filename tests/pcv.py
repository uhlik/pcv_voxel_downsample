# import os
# import sys
# sys.path.append("/path/to/directory/pcv_voxel_downsample")

import bpy
import numpy as np
import bl_ext.user_default.point_cloud_visualizer as pcv
import pcv_voxel_downsample

o = bpy.context.active_object
pd = pcv.common.get_data(name=o.name, )
vs = pd.vs.astype(np.float64)

dvs, dii = pcv_voxel_downsample.voxel_downsample(vs, 0.02)

pd.filter(dii)
pcv.mechanist.PCVMechanist.force_update(o.name)
pcv.mechanist.PCVMechanist.tag_redraw()

