# +D 2024.4.26
# Last modified time: 2024.4.28
import numpy as np
from scene import Scene, GaussianModel
from plyfile import PlyData, PlyElement
from tqdm import tqdm
import open3d as o3d
import struct
import json

# parameters
chunk_dict = {} 
    # Level 0 : {root: [points]}
    # ...
    # Level n : {node_name_a: [points], ..., node_name_b: [points]}
byte_dict = {} 
    # Level 0 : {root: [byte_offset, byte_size]}
    # ...
    # Level n : {node_name_a: [byte_offset, byte_size], ..., 
    #               node_name_b: [byte_offset, byte_size]}
    # promise first meet level-n node, then meet level-(n+1) node

class SimpleGaussian:
    def __init__(self, Gaussian_model, level):
        # check the type of Gaussian_model
        if isinstance(Gaussian_model, GaussianModel):
            self.xyz = Gaussian_model._xyz.detach().cpu().numpy()
            self.features_dc = Gaussian_model._features_dc.detach().transpose(1, 2)\
                .flatten(start_dim=1).contiguous().cpu().numpy()
            self.features_rest = Gaussian_model._features_rest.detach().transpose(1, 2)\
                .flatten(start_dim=1).contiguous().cpu().numpy()
            self.opacity = Gaussian_model._opacity.detach().cpu().numpy()
            self.scaling = Gaussian_model._scaling.detach().cpu().numpy()
            self.rotation = Gaussian_model._rotation.detach().cpu().numpy()
            self.normals = np.zeros_like(self.xyz)
            self.level = level
            
        else: # init
            self.xyz = None
            self.features_dc = None
            self.features_rest = None
            self.opacity = None
            self.scaling = None
            self.rotation = None
            self.normals = None
            self.level = -1
            
    def output(self, dtype_full, file_name):
        elements = np.empty(self.xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((self.xyz, self.normals, 
                    self.features_dc, self.features_rest,
                    self.opacity, self.scaling, self.rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        
        PlyData([el]).write(file_name)

class Point:
    def __init__(self, xyz, normal, features_dc, features_rest, 
                 opacity, scaling, rotation, SH_degree):
        self.position = xyz.tolist()
        self.intensity = 0
        self.return_number = 0
        self.number_of_returns = 0
        self.classification = 0
        self.scan_angle_rank = 0
        self.user_data = 0
        self.point_source_id = 0
        self.rgb = [0, 0, 0]
        self.f_dc_0 = features_dc[0]
        self.f_dc_1 = features_dc[1]
        self.f_dc_2 = features_dc[2]
        
        if SH_degree > 0:
            self.f_rest_0 = features_rest[0]
            self.f_rest_1 = features_rest[1]
            self.f_rest_2 = features_rest[2]
            self.f_rest_3 = features_rest[3]
            self.f_rest_4 = features_rest[4]
            self.f_rest_5 = features_rest[5]
            self.f_rest_6 = features_rest[6]
            self.f_rest_7 = features_rest[7]
            self.f_rest_8 = features_rest[8]
        
        if SH_degree > 1:
            self.f_rest_9 = features_rest[9]
            self.f_rest_10 = features_rest[10]
            self.f_rest_11 = features_rest[11]
            self.f_rest_12 = features_rest[12]
            self.f_rest_13 = features_rest[13]
            self.f_rest_14 = features_rest[14]
            self.f_rest_15 = features_rest[15]
            self.f_rest_16 = features_rest[16]
            self.f_rest_17 = features_rest[17]
            self.f_rest_18 = features_rest[18]
            self.f_rest_19 = features_rest[19]
            self.f_rest_20 = features_rest[20]
            self.f_rest_21 = features_rest[21]
            self.f_rest_22 = features_rest[22]
            self.f_rest_23 = features_rest[23]
        
        if SH_degree > 2:
            self.f_rest_24 = features_rest[24]
            self.f_rest_25 = features_rest[25]
            self.f_rest_26 = features_rest[26]
            self.f_rest_27 = features_rest[27]
            self.f_rest_28 = features_rest[28]
            self.f_rest_29 = features_rest[29]
            self.f_rest_30 = features_rest[30]
            self.f_rest_31 = features_rest[31]
            self.f_rest_32 = features_rest[32]
            self.f_rest_33 = features_rest[33]
            self.f_rest_34 = features_rest[34]
            self.f_rest_35 = features_rest[35]
            self.f_rest_36 = features_rest[36]
            self.f_rest_37 = features_rest[37]
            self.f_rest_38 = features_rest[38]
            self.f_rest_39 = features_rest[39]
            self.f_rest_40 = features_rest[40]
            self.f_rest_41 = features_rest[41]
            self.f_rest_42 = features_rest[42]
            self.f_rest_43 = features_rest[43]
            self.f_rest_44 = features_rest[44]

        self.opacity = opacity
        self.scale_0 = scaling[0]
        self.scale_1 = scaling[1]
        self.scale_2 = scaling[2]
        self.rot_0 = rotation[0]
        self.rot_1 = rotation[1]
        self.rot_2 = rotation[2]
        self.rot_3 = rotation[3]
      
def point_is_in_BBox(xyz: list, BBox: list):
    return xyz[0] >= BBox[0] and xyz[0] < BBox[3] and \
           xyz[1] >= BBox[1] and xyz[1] < BBox[4] and \
           xyz[2] >= BBox[2] and xyz[2] < BBox[5]
           
def create_child_BBox(BBox, index):
    size = [BBox[3] - BBox[0], BBox[4] - BBox[1], BBox[5] - BBox[2]]
    result_BBox = BBox.copy()

    if index & 0b0001 > 0: result_BBox[2] += size[2] / 2 # min_z
    else:                  result_BBox[5] -= size[2] / 2 # max_z  
    if index & 0b0010 > 0: result_BBox[1] += size[1] / 2 # min_y
    else:                  result_BBox[4] -= size[1] / 2 # max_y
    if index & 0b0100 > 0: result_BBox[0] += size[0] / 2 # min_x
    else:                  result_BBox[3] -= size[0] / 2 # max_x
        
    return result_BBox

def point_in_which_BBox(xyz: list, BBox: list):
    # inverse of create_child_BBox
    size = [BBox[3] - BBox[0], BBox[4] - BBox[1], BBox[5] - BBox[2]]
    index = 0

    if xyz[2] >= BBox[2] + size[2] / 2: index |= 0b0001
    if xyz[1] >= BBox[1] + size[1] / 2: index |= 0b0010
    if xyz[0] >= BBox[0] + size[0] / 2: index |= 0b0100

    return index
    
class PotreeNode:
    def __init__(self, BBox, level, f_name, index, SH_degree, Gau_list):
        self.name = str(f_name) + str(index)
        assert len(self.name) == level + 1
        chunk_dict["level_"+str(level)][self.name] = [] # init
        
        self.BBox = BBox
        self.level = level
        # self.points = [Point(Gau_list[level].xyz[i], 
        #     Gau_list[level].normals[i], Gau_list[level].features_dc[i], 
        #     Gau_list[level].features_rest[i], Gau_list[level].opacity[i], 
        #     Gau_list[level].scaling[i], Gau_list[level].rotation[i], SH_degree) 
        #     for i in range(self.num_points)
        #     if point_is_in_BBox(Gau_list[level].xyz[i], self.BBox)]
        #      # TOO SLOW

        if level < len(Gau_list) - 1:
            self.children = [PotreeNode(create_child_BBox(self.BBox, i), level + 1, 
                    self.name, i, SH_degree, Gau_list) for i in range(8)]
        else: self.children = []
            
        # features
        self.points = []
        self.num_points = 0
        
        # # hierarchy features
        # self.byte_offset = 0
        # self.byte_size = 0
        
    def update_PotreeNode(self):
        self.points = chunk_dict["level_"+str(self.level)][self.name]
        self.num_points = len(self.points)
        
        # recursively update the children
        for child in self.children:
            child.update_PotreeNode()
        
        # check
        # self.check()
            
    def check(self):
        print("Name:", self.name)
        print("BBox:", self.BBox)
        for i in self.points:
            assert point_is_in_BBox(i.position, self.BBox)
        
    def visualize(self):
        print("Level", self.level, "chunking ...")
        print("Name:", self.name)
        print("BBox:", self.BBox)
        print("Number:", len(chunk_dict["level_"+str(self.level)][self.name]))
        print()
        
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(
            [p.position for p in chunk_dict["level_"+str(self.level)][self.name]])
        
        bounding_box = o3d.geometry.AxisAlignedBoundingBox\
                (min_bound=self.BBox[:3], max_bound=self.BBox[3:])
        o3d.visualization.draw_geometries([point_cloud, bounding_box])
            
        
class Potree:
    def __init__(self, BBox, SH_degree, Gau_list):
        self.BBox = BBox # min_x, min_y, min_z, max_x, max_y, max_z
        self.SH_degree = SH_degree
              
        global chunk_dict, byte_dict
        for i in range(len(Gau_list)):
            chunk_dict["level_"+str(i)] = {}
            byte_dict["level_"+str(i)] = {}
            
        self.root = self.generate_octree(Gau_list)
        self.assign_points(Gau_list)
        self.root.update_PotreeNode()
        
        # for output
        self.byte_of_per_point = self.get_byte_of_per_point()
        self.total_points_num = sum([len(i.xyz) for i in Gau_list])
        self.bytedict_init()
                   
    def generate_octree(self, Gau_list):
        # total_chunk_num = 8^0 + 8^1 + ... + 8^(len(Gau_list)-1)
        global total_chunk_num
        total_chunk_num = sum([8**i for i in range(len(Gau_list))])
        self.root = PotreeNode(self.BBox, 0, "", "r", self.SH_degree, Gau_list)
        
        return self.root
    
    def assign_points(self, Gau_list):
        # level = 0    
        global chunk_dict
        print("Level 0 chunking ...")
        chunk_dict["level_0"]["r"] = [Point(Gau_list[0].xyz[i],
            Gau_list[0].normals[i], Gau_list[0].features_dc[i],
            Gau_list[0].features_rest[i], Gau_list[0].opacity[i],
            Gau_list[0].scaling[i], Gau_list[0].rotation[i], self.SH_degree)
            for i in range(len(Gau_list[0].xyz))]
        
        # level 1 ~ len(Gau_list) - 1
        for l in range(1, len(Gau_list)):
            print("Level", l, "chunking ...")
            for p in tqdm(range(Gau_list[l].xyz.shape[0]), desc = "LOD " + str(l)):
                # each gau point, from root to this node
                curr_BBox = self.BBox
                name = "r"
                for i in range(l):
                    index = point_in_which_BBox(Gau_list[l].xyz[p], curr_BBox)
                    curr_BBox = create_child_BBox(curr_BBox, index)
                    name = name + str(index)
                    
                # if name not in chunk_dict["level_"+str(l)].keys():
                #      chunk_dict["level_"+str(l)][name] = []
                    
                chunk_dict["level_"+str(l)][name].append(Point(Gau_list[l].xyz[p],
                    Gau_list[l].normals[p], Gau_list[l].features_dc[p],
                    Gau_list[l].features_rest[p], Gau_list[l].opacity[p],
                    Gau_list[l].scaling[p], Gau_list[l].rotation[p], self.SH_degree))
                
    def get_byte_of_per_point(self):
        if   self.SH_degree == 0: return 71
        elif self.SH_degree == 1: return 71 +  9 * 4
        elif self.SH_degree == 2: return 71 + 24 * 4
        elif self.SH_degree == 3: return 71 + 45 * 4
        else:
            print("SH_degree should be 0, 1, 2, or 3.")
            exit(-1)
          
    def bytedict_init(self):
        print("bytedict initialization")
        curr_byte_offset = 0 # for hierarchy.bin
        for l in range(len(chunk_dict)):
            for key in tqdm(chunk_dict["level_"+str(l)].keys(), desc = "LOD " + str(l)):
                curr_byte_size = \
                        self.byte_of_per_point * len(chunk_dict["level_"+str(l)][key])
                global byte_dict
                byte_dict["level_"+str(l)][key] = [curr_byte_offset, curr_byte_size]
                curr_byte_offset += curr_byte_size
                
                
            
    def output_octree(self, filename: str):
        # output to .bin file
        # curr_byte_offset = 0 # for hierarchy.bin
        
        print("Outputting", filename)
        with open(filename, 'wb') as f:
            for l in range(len(chunk_dict)):
                for key in tqdm(chunk_dict["level_"+str(l)].keys(), desc = "LOD " + str(l)):
                    # curr_byte_size = \
                    #     self.byte_of_per_point * len(chunk_dict["level_"+str(l)][key])
                    # global byte_dict
                    # byte_dict["level_"+str(l)][key] = [curr_byte_offset, curr_byte_size]
                    # curr_byte_offset += curr_byte_size
                    
                    for point in chunk_dict["level_"+str(l)][key]:
                        f.write(struct.pack('f', point.position[0]))
                        f.write(struct.pack('f', point.position[1]))
                        f.write(struct.pack('f', point.position[2]))
                        f.write(struct.pack('H', point.intensity))
                        f.write(struct.pack('B', point.return_number))
                        f.write(struct.pack('B', point.number_of_returns))
                        f.write(struct.pack('B', point.classification))
                        f.write(struct.pack('b', point.scan_angle_rank))
                        f.write(struct.pack('B', point.user_data))
                        f.write(struct.pack('H', point.point_source_id))
                        f.write(struct.pack('H', point.rgb[0]))
                        f.write(struct.pack('H', point.rgb[1]))
                        f.write(struct.pack('H', point.rgb[2]))
                        f.write(struct.pack('f', point.f_dc_0))
                        f.write(struct.pack('f', point.f_dc_1))
                        f.write(struct.pack('f', point.f_dc_2))
                        if self.SH_degree > 0:
                            f.write(struct.pack('f', point.f_rest_0))
                            f.write(struct.pack('f', point.f_rest_1))
                            f.write(struct.pack('f', point.f_rest_2))
                            f.write(struct.pack('f', point.f_rest_3))
                            f.write(struct.pack('f', point.f_rest_4))
                            f.write(struct.pack('f', point.f_rest_5))
                            f.write(struct.pack('f', point.f_rest_6))
                            f.write(struct.pack('f', point.f_rest_7))
                            f.write(struct.pack('f', point.f_rest_8))
                        if self.SH_degree > 1:
                            f.write(struct.pack('f', point.f_rest_9))
                            f.write(struct.pack('f', point.f_rest_10))
                            f.write(struct.pack('f', point.f_rest_11))
                            f.write(struct.pack('f', point.f_rest_12))
                            f.write(struct.pack('f', point.f_rest_13))
                            f.write(struct.pack('f', point.f_rest_14))
                            f.write(struct.pack('f', point.f_rest_15))
                            f.write(struct.pack('f', point.f_rest_16))
                            f.write(struct.pack('f', point.f_rest_17))
                            f.write(struct.pack('f', point.f_rest_18))
                            f.write(struct.pack('f', point.f_rest_19))
                            f.write(struct.pack('f', point.f_rest_20))
                            f.write(struct.pack('f', point.f_rest_21))
                            f.write(struct.pack('f', point.f_rest_22))
                            f.write(struct.pack('f', point.f_rest_23))
                        if self.SH_degree > 2:
                            f.write(struct.pack('f', point.f_rest_24))
                            f.write(struct.pack('f', point.f_rest_25))
                            f.write(struct.pack('f', point.f_rest_26))
                            f.write(struct.pack('f', point.f_rest_27))
                            f.write(struct.pack('f', point.f_rest_28))
                            f.write(struct.pack('f', point.f_rest_29))
                            f.write(struct.pack('f', point.f_rest_30))
                            f.write(struct.pack('f', point.f_rest_31))
                            f.write(struct.pack('f', point.f_rest_32))
                            f.write(struct.pack('f', point.f_rest_33))
                            f.write(struct.pack('f', point.f_rest_34))
                            f.write(struct.pack('f', point.f_rest_35))
                            f.write(struct.pack('f', point.f_rest_36))
                            f.write(struct.pack('f', point.f_rest_37))
                            f.write(struct.pack('f', point.f_rest_38))
                            f.write(struct.pack('f', point.f_rest_39))
                            f.write(struct.pack('f', point.f_rest_40))
                            f.write(struct.pack('f', point.f_rest_41))
                            f.write(struct.pack('f', point.f_rest_42))
                            f.write(struct.pack('f', point.f_rest_43))
                            f.write(struct.pack('f', point.f_rest_44))
                        f.write(struct.pack('f', point.opacity))
                        f.write(struct.pack('f', point.scale_0))
                        f.write(struct.pack('f', point.scale_1))
                        f.write(struct.pack('f', point.scale_2))
                        f.write(struct.pack('f', point.rot_0))
                        f.write(struct.pack('f', point.rot_1))
                        f.write(struct.pack('f', point.rot_2))
                        f.write(struct.pack('f', point.rot_3))
        
    def output_hierarchy(self, filename: str):
        # prune 
        # print(byte_dict)  
        for l in range(len(byte_dict)):
            for key in tqdm(list(byte_dict["level_"+str(l)].keys()), 
                                        desc = "LOD " + str(l) + " pruning"):
                if byte_dict["level_"+str(l)][key][1] == 0:
                    del byte_dict["level_"+str(l)][key]
                    # del chunk_dict["level_"+str(l)][key]
                    
                    for child_level in range(l + 1, len(byte_dict)):
                        for child_key in list(byte_dict["level_"+str(child_level)].keys()):
                            if child_key.startswith(key):
                                del byte_dict["level_"+str(child_level)][child_key]
                                # del chunk_dict["level_"+str(child_level)][child_key]
        
        # print(byte_dict)                        
        for l in range(len(byte_dict)):
            for key in tqdm(byte_dict["level_"+str(l)].keys(), desc = "check pruning"):
                assert byte_dict["level_"+str(l)][key][1] > 0
        
        def create_child_mask(key):
            # inverse： childExists = ((1 << childIndex) & childMask) !== 0;
            child_mask = 0
            for index in range(8):
                if key + str(index) in byte_dict["level_"+str(len(key))].keys():
                    child_mask |= (1 << index)  
                
            return child_mask     
        
        print("Outputting", filename)
        with open(filename, 'wb') as f:
            for l in range(len(byte_dict)):
                for key in tqdm(byte_dict["level_"+str(l)].keys()):
                    f.write(struct.pack('B', 0)) # type
                    if len(key) < len(chunk_dict):
                        # f.write(struct.pack('B', 255)) # child mask
                            # because r0 & r000 exist, but r00-r07 not exist
                            # so set 255, store all children although child has no points
                        f.write(struct.pack('B', create_child_mask(key))) # child mask
                    else:
                        f.write(struct.pack('B', 0)) # leaf node has no child
                    f.write(struct.pack('I', len(chunk_dict["level_"+str(l)][key])))
                    f.write(struct.pack('Q', byte_dict["level_"+str(l)][key][0]))
                    f.write(struct.pack('Q', byte_dict["level_"+str(l)][key][1]))
                
    def output_metadata(self, filename: str, name: str):
        data = {
	"version": "2.0",
	"name": name,
	"description": "",
	"points": self.total_points_num,
	"projection": "",
	"hierarchy": {
		"firstChunkSize": sum([len(chunk_dict[level]) for level in chunk_dict.keys()]) * 22, 
		"stepSize": 4, 
		"depth": len(chunk_dict)
	},
	"offset": [0, 0, 0],
	"scale": [1, 1, 1],
	"spacing": 0.78742187499999994,
	"boundingBox": {
		"min": [self.BBox[0], self.BBox[1], self.BBox[2]],
		"max": [self.BBox[3], self.BBox[4], self.BBox[5]]
	},
	"attributes": [
		{
			"name": "position",
			"description": "",
			"size": 12,
			"numElements": 3,
			"elementSize": 4,
			"type": "int32",
            "min": [self.BBox[0], self.BBox[1], self.BBox[2]],
            "max": [self.BBox[3], self.BBox[4], self.BBox[5]]
		},{
			"name": "intensity",
			"description": "",
			"size": 2,
			"numElements": 1,
			"elementSize": 2,
			"type": "uint16",
			"min": [0],
			"max": [0]
		},{
			"name": "return number",
			"description": "",
			"size": 1,
			"numElements": 1,
			"elementSize": 1,
			"type": "uint8",
			"min": [0],
			"max": [0]
		},{
			"name": "number of returns",
			"description": "",
			"size": 1,
			"numElements": 1,
			"elementSize": 1,
			"type": "uint8",
			"min": [0],
			"max": [0]
		},{
			"name": "classification",
			"description": "",
			"size": 1,
			"numElements": 1,
			"elementSize": 1,
			"type": "uint8",
			"min": [0],
			"max": [0]
		},{
			"name": "scan angle rank",
			"description": "",
			"size": 1,
			"numElements": 1,
			"elementSize": 1,
			"type": "uint8",
			"min": [0],
			"max": [0]
		},{
			"name": "user data",
			"description": "",
			"size": 1,
			"numElements": 1,
			"elementSize": 1,
			"type": "uint8",
			"min": [0],
			"max": [0]
		},{
			"name": "point source id",
			"description": "",
			"size": 2,
			"numElements": 1,
			"elementSize": 2,
			"type": "uint16",
			"min": [0],
			"max": [0]
		},{
			"name": "rgb",
			"description": "",
			"size": 6,
			"numElements": 3,
			"elementSize": 2,
			"type": "uint16",
			"min": [0, 0, 0],
			"max": [0, 0, 0]
		},{
			"name": "f_dc_0",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-2.4582734107971191],
			"max": [11.07319164276123]
		},{
			"name": "f_dc_1",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-2.5723543167114258],
			"max": [10.608137130737305]
		},{
			"name": "f_dc_2",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-2.7653396129608154],
			"max": [10.402518272399902]
		}]
        }
        
        if self.SH_degree > 0:
            data["attributes"].extend([{
			"name": "f_rest_0",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.6647828221321106],
			"max": [0.69167155027389526]
		},{
			"name": "f_rest_1",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.65622037649154663],
			"max": [0.74413615465164185]
		},{
			"name": "f_rest_2",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.74981796741485596],
			"max": [0.79847460985183716]
		},{
			"name": "f_rest_3",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.71122455596923828],
			"max": [0.57561463117599487]
		},{
			"name": "f_rest_4",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.76607441902160645],
			"max": [0.71668189764022827]
		},{
			"name": "f_rest_5",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.69944930076599121],
			"max": [0.73511236906051636]
		},{
			"name": "f_rest_6",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.72960412502288818],
			"max": [0.68665027618408203]
		},{
			"name": "f_rest_7",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.73256313800811768],
			"max": [0.72801482677459717]
		},{
			"name": "f_rest_8",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.81797313690185547],
			"max": [0.70387691259384155]
		}])

        if self.SH_degree > 1:
            data["attributes"].extend([{
			"name": "f_rest_9",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.67035371065139771],
			"max": [0.62769144773483276]
		},{
			"name": "f_rest_10",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.71357452869415283],
			"max": [0.70244914293289185]
		},{
			"name": "f_rest_11",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.73703843355178833],
			"max": [0.59758758544921875]
		},{
			"name": "f_rest_12",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.747935950756073],
			"max": [0.66759634017944336]
		},{
			"name": "f_rest_13",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.70942342281341553],
			"max": [0.70607751607894897]
		},{
			"name": "f_rest_14",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.73989236354827881],
			"max": [0.77428674697875977]
		},{
			"name": "f_rest_15",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.72205346822738647],
			"max": [0.71835964918136597]
		},{
			"name": "f_rest_16",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.77432429790496826],
			"max": [0.62449032068252563]
		},{
			"name": "f_rest_17",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.64565891027450562],
			"max": [0.72051209211349487]
		},{
			"name": "f_rest_18",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.75492310523986816],
			"max": [0.61498719453811646]
		},{
			"name": "f_rest_19",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.62955480813980103],
			"max": [0.83931291103363037]
		},{
			"name": "f_rest_20",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.724040687084198],
			"max": [0.74494194984436035]
		},{
			"name": "f_rest_21",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.59663909673690796],
			"max": [0.64260637760162354]
		},{
			"name": "f_rest_22",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.65768957138061523],
			"max": [0.61022317409515381]
		},{
			"name": "f_rest_23",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.70371192693710327],
			"max": [0.76185494661331177]
		}])
            
        if self.SH_degree > 2:
            data["attributes"].extend([{
			"name": "f_rest_24",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.73405128717422485],
			"max": [0.63920551538467407]
        },{
			"name": "f_rest_25",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.7267799973487854],
			"max": [0.76526784896850586]
		},{
			"name": "f_rest_26",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.74943464994430542],
			"max": [0.58203870058059692]
		},{
			"name": "f_rest_27",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.74349486827850342],
			"max": [0.66099637746810913]
		},{
			"name": "f_rest_28",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.85665357112884521],
			"max": [0.73723173141479492]
		},{
			"name": "f_rest_29",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.70733922719955444],
			"max": [0.67758208513259888]
		},{
			"name": "f_rest_30",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.85392791032791138],
			"max": [0.63739776611328125]
		},{
			"name": "f_rest_31",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.61804771423339844],
			"max": [0.81112807989120483]
		},{
			"name": "f_rest_32",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.68888139724731445],
			"max": [0.60017663240432739]
		},{
			"name": "f_rest_33",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.60267543792724609],
			"max": [0.62262105941772461]
		},{
			"name": "f_rest_34",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.752846360206604],
			"max": [0.66014015674591064]
		},{
			"name": "f_rest_35",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.74492651224136353],
			"max": [0.73031306266784668]
		},{
			"name": "f_rest_36",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.80007082223892212],
			"max": [0.6908145546913147]
		},{
			"name": "f_rest_37",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.74193006753921509],
			"max": [0.75640332698822021]
		},{
			"name": "f_rest_38",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.91326040029525757],
			"max": [0.66999775171279907]
		},{
			"name": "f_rest_39",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.63477849960327148],
			"max": [0.61217200756072998]
		},{
			"name": "f_rest_40",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.81131958961486816],
			"max": [0.65762543678283691]
		},{
			"name": "f_rest_41",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.82246571779251099],
			"max": [0.67946845293045044]
		},{
			"name": "f_rest_42",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.64124161005020142],
			"max": [0.69097524881362915]
		},{
			"name": "f_rest_43",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.84428238868713379],
			"max": [0.773406982421875]
		},{
			"name": "f_rest_44",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-0.78714293241500854],
			"max": [0.74573200941085815]
		}])
           
        data["attributes"].extend([{
			"name": "opacity",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-7.1026501655578613],
			"max": [18.458217620849609]
		},{
			"name": "scale_0",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-22.883628845214844],
			"max": [1.516323447227478]
		},{
			"name": "scale_1",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-22.913396835327148],
			"max": [1.874864935874939]
		},{
			"name": "scale_2",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-20.649715423583984],
			"max": [1.3969866037368774]
		},{
			"name": "rot_0",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-1.0283368825912476],
			"max": [3.9363696575164795]
		},{
			"name": "rot_1",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-2.1333160400390625],
			"max": [1.9026168584823608]
		},{
			"name": "rot_2",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-2.0993406772613525],
			"max": [1.9349937438964844]
		},{
			"name": "rot_3",
			"description": "",
			"size": 4,
			"numElements": 1,
			"elementSize": 4,
			"type": "float",
			"min": [-1.9207870960235596],
			"max": [1.6994497776031494]
		}])
        
        print("Outputting", filename)
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)        