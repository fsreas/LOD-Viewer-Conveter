# 2024.4.26 +D
# Last modified time: 2024.7.28
# Convert multi-level 3D LOD Gaussian output ply form to Potree form

import numpy as np
import os
import sys
from scene import Scene, GaussianModel
from tqdm import tqdm, trange

from option import read_yaml, read_cam_cfg
from model import SimpleGaussian, Point, Potree
import open3d as o3d

# parameters
option_file = "para.yaml"
para = read_yaml(option_file)

input_prefix = para["inputFilePrefix"]
SH_degree = para["SHDegreeInput"]
SH_degree_out = para["SHDegreeOutput"]
d = para["delta"]

def get_global_bbox(ply_list: list):
    min_x, min_y, min_z = np.inf, np.inf, np.inf
    max_x, max_y, max_z = -np.inf, -np.inf, -np.inf
    
    print("Calculating the global bounding box...")
    for ply in tqdm(ply_list): # using open3d
        pcd = o3d.io.read_point_cloud(ply)
        xyz = np.asarray(pcd.points)
        min_x = min(min_x, np.min(xyz[:, 0]))
        min_y = min(min_y, np.min(xyz[:, 1]))
        min_z = min(min_z, np.min(xyz[:, 2]))
        max_x = max(max_x, np.max(xyz[:, 0]))
        max_y = max(max_y, np.max(xyz[:, 1]))
        max_z = max(max_z, np.max(xyz[:, 2]))
        
    return [min_x, min_y, min_z, max_x + d, max_y + d, max_z + d]

# ------ main ------
if len(sys.argv) != 3:
    print("Usage: python ply2las.py <path> <scene_name>")
    exit(0)      

# set the iteration num with the newest one or parameter setting
input_folder = [f for f in os.listdir(
        os.path.join(sys.argv[1])) 
        if f.startswith(input_prefix) and f.endswith(".ply")]
if len(input_folder) == 0:
    print("No gaussian output folder")
    exit(0)
input_folder.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    # level_0.ply -> 0

# if not exist backup folder, create it
if not os.path.exists(os.path.join(sys.argv[1], "backup")):
    os.makedirs(os.path.join(sys.argv[1], "backup"))

print("The input files are:")
print('\n'.join(os.path.join(sys.argv[1], f) for f in input_folder))

BBox = get_global_bbox([os.path.join(sys.argv[1], f)
                       for f in input_folder])
    # min_x, min_y, min_z, max_x, max_y, max_z

Gau_list = []
for f in input_folder:
    print("Loading {} ...".format(f))
    
    level = int(f.split('_')[1].split('.')[0])

    # set the input, backup, reference ply
    input_ply = os.path.join(sys.argv[1], f)
    # copy input_ply to backup folder
    backup_ply = os.path.join(sys.argv[1], "backup", f)
    os.system("cp {} {}".format(input_ply, backup_ply)) # backup

    input_gaussian = GaussianModel(SH_degree)
    input_gaussian.load_ply(input_ply)
    dtype_full = [(attribute, 'f4') for attribute in 
                input_gaussian.construct_list_of_attributes()]
    result = SimpleGaussian(input_gaussian, level)
    
    Gau_list.append(result)

# convert to potree structure
potree = Potree(BBox, SH_degree_out, Gau_list)

# save to potree
output_folder = os.path.join(sys.argv[1], "potree")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

octree_bin = os.path.join(output_folder, para["octreeBin"])
hierarchy_bin = os.path.join(output_folder, para["hierarchyBin"])
metadata_json = os.path.join(output_folder, para["metadataFile"])
potree.output_octree(octree_bin)
potree.output_hierarchy(hierarchy_bin)
potree.output_metadata(metadata_json, sys.argv[2])

print("The output files are:", octree_bin, hierarchy_bin, metadata_json)