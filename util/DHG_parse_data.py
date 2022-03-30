from pathlib import Path
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2

#change the path to your downloaded DHG dataset

data_fold = "/home/l.kulikov/datasets/dataset_dhg1428"

def read_data_from_disk():
    def parse_data(src_file, n=3):
        video = []
        for line in src_file:
            line = line.split("\n")[0]
            data = line.split(" ")
            frame = []
            point = []
            for data_ele in data:
                point.append(float(data_ele))
                if len(point) == n:
                    frame.append(point)
                    point = []
            video.append(frame)
        return video
    
    result = {}
    result_proj = {}
    result_gen_info = {}
    result_depth_maps = {}
    for g_id in range(1,15):
        print("gesture {} / {}".format(g_id,14))
        for f_id in range(1,3):
            for sub_id in range(1,21):
                for e_id in range(1,6):
                    root = data_fold + "/gesture_{}/finger_{}/subject_{}/essai_{}".format(g_id, f_id, sub_id, e_id)
                    src_path = root + '/skeleton_world.txt'
                    src_file = open(src_path)
                    video = parse_data(src_file, n=3) #the 22 points for each frame of the video
                    
                    src_proj_path = root + '/skeleton_image.txt'
                    src_proj_file = open(src_proj_path)
                    video_proj = parse_data(src_proj_file, n=2)
                    
                    src_gen_info_path = root + '/general_information.txt'
                    gen_info = np.loadtxt(src_gen_info_path)[:, 1:]  # x, y, width, height
                    
                    depth_maps = [p for p in glob(root + '/depth*.png')]
                    
                    key = "{}_{}_{}_{}".format(g_id, f_id, sub_id, e_id)
                    
                    result[key] = video
                    result_proj[key] = video_proj
                    result_gen_info[key] = gen_info
                    result_depth_maps[key] = depth_maps
                    
                    src_file.close()
                    src_proj_file.close()
                    
    return result, result_proj, result_gen_info, result_depth_maps

def get_valid_frame(video_data, video_data_proj, video_data_gen_info, video_data_depth_maps):
    # filter frames using annotation
    info_path = data_fold + "/informations_troncage_sequences.txt"
    info_file = open(info_path)
    used_key = []
    for line in info_file:
        line = line.split("\n")[0]
        data = line.split(" ")
        g_id =  data[0]
        f_id = data[1]
        sub_id = data[2]
        e_id = data[3]
        key = "{}_{}_{}_{}".format(g_id, f_id, sub_id, e_id)
        used_key.append(key)
        start_frame = int(data[4])
        end_frame = int(data[5])
        
        data = video_data[key]
        data_proj = video_data_proj[key]
        data_gen_info = video_data_gen_info[key]
        data_depth = video_data_depth_maps[key]
        
        video_data[key] = data[start_frame : end_frame + 1]
        video_data_proj[key] = data_proj[start_frame : end_frame + 1]
        video_data_gen_info[key] = data_gen_info[start_frame : end_frame + 1]
        video_data_depth_maps[key] = data_depth[start_frame : end_frame + 1]
        
    return video_data, video_data_proj, video_data_gen_info, video_data_depth_maps

def split_train_test(test_subject_id, filtered_video_data, filtered_video_data_proj, 
                     filtered_video_data_gen_info, filtered_video_data_depth_maps, cfg):

    #split data into train and test
    #cfg = 0 >>>>>>> 14 categories      cfg = 1 >>>>>>>>>>>> 28 cate
    train_data = []
    test_data = []
    for g_id in range(1, 15):
        for f_id in range(1, 3):
            for sub_id in range(1, 21):
                for e_id in range(1, 6):
                    key = "{}_{}_{}_{}".format(g_id, f_id, sub_id, e_id)

                    #set table to 14 or
                    if cfg == 0:
                        label = g_id
                    elif cfg == 1:
                        if f_id == 1:
                            label = g_id
                        else:
                            label = g_id + 14

                    #split to train and test list
                    data = filtered_video_data[key]
                    data_proj = filtered_video_data_proj[key]
                    data_gen_info = filtered_video_data_gen_info[key]
                    data_depth_maps = filtered_video_data_depth_maps[key]
                    
                    sample = {
                        "skeleton": data, 
                        "skeleton_proj": data_proj, 
                        "gen_info": data_gen_info, 
                        "depth": data_depth_maps, 
                        "label": label 
                    }
                    
                    if sub_id == test_subject_id:
                        test_data.append(sample)
                    else:
                        train_data.append(sample)
    if len(test_data) == 0:
        raise "no such test subject"

    return train_data, test_data

def get_train_test_data(test_subject_id, cfg):
    print("reading data from desk.......")
    data = read_data_from_disk()
    
    print("filtering frames .......")
    data = get_valid_frame(*data)
    
    train_data, test_data = split_train_test(test_subject_id, *data, cfg)
    
    return train_data, test_data
