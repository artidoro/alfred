import sys
import os
# %env ALFRED_ROOT="/home/ubuntu/alfred"
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))


import numpy as np
from PIL import Image
import json
from env.thor_env import ThorEnv
import random



import time

def setup_scene(env, traj_data):
    '''
    intialize the scene and agent from the task info
    '''
    # scene setup
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']
    object_toggles = traj_data['scene']['object_toggles']

    scene_name = 'FloorPlan%d' % scene_num
    env.reset(scene_name)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)

    # initialize to start position
    env.step(dict(traj_data['scene']['init_action']))



def generate_feasibility_data(env, filepath):
    feasibilty_filepath = filepath + "/feas_info.jsonl"

    mask = None
    smooth_nav = False
    debug = False 

    dict_list = []

    valid_scene_path = os.path.abspath("data/json_2.1.0/valid_unseen")
    all_scene_paths = os.listdir(valid_scene_path)



    scene_num_set = set()

    scene_counter = 0 
    for scene_path in all_scene_paths:
        start = time.time()
        json_path = valid_scene_path + "/" + scene_path + "/" + os.listdir(valid_scene_path + "/" + scene_path)[0] + "/traj_data.json"

        # if scene_counter == 2:
        #     assert False

        with open(json_path) as f:
            traj_data = json.load(f)

        scene_num = traj_data['scene']['scene_num']

        if scene_num in scene_num_set:
            continue
        
        scene_num_set.add(scene_num)

        #first scene setup
        setup_scene(env, traj_data)
        
        

        #get all unoccupied locations
        all_locations = env.step(dict(action='GetReachablePositions')).metadata["actionReturn"]

        random.seed(0)
        random.shuffle(all_locations)
        # print("number of locations: ", str(len(all_locations)))
        # all_locations = all_locations[:100]

        for location in all_locations:

            traj_data["scene"]["init_action"]["x"] = location["x"]
            traj_data["scene"]["init_action"]["y"] = location["y"]
            traj_data["scene"]["init_action"]["z"] = location["z"]

            location_extension = str(location["x"]) + "_" + str(location["y"]) + "_" + str(location["z"])
            #for organizing file paths for save
            for rotation in range(0,360, 90):
                one_dict = {}


                traj_data["scene"]["init_action"]["rotation"] = rotation

                env.step(dict(traj_data['scene']['init_action']))
                # This sets the position and rotation of the robot within the scene

                curr_image = Image.fromarray(np.uint8(env.last_event.frame))
                image_save_filepath = filepath + "/images/" + scene_path + "_" + location_extension + "_" + str(rotation) + ".jpg"

                # print(image_save_filepath)
                #saving image
                with open(image_save_filepath,"w") as f:
                    curr_image.save(f)

                success, _, _, err, _ = env.va_interact("MoveAhead_15", interact_mask=mask, smooth_nav=smooth_nav, debug=debug)

                # feasibility_list.append(success)
                # error_list.append(err)
                one_dict["success"] = success
                one_dict["error_msg"] = err
                one_dict["scene_num"] = scene_num
                one_dict["image_filepath"] = image_save_filepath
                one_dict["pos_x"] = location["x"]
                one_dict["pos_y"] = location["y"]
                one_dict["pos_z"] = location["z"]
                one_dict["rotation"] = rotation
                dict_list.append(one_dict)
            
                with open(feasibilty_filepath, "a") as f:
                    new_line = json.dumps(one_dict)
                    f.write('\n' + new_line)


        end = time.time()

        print("Time taken for scene " + str(scene_num) +": " + str(end - start) + ". " + str(scene_counter) + " out of " + str(len(all_scene_paths)) + " total scenes." + " # of locations = " + str(len(all_locations)))
        scene_counter += 1
    with open(filepath + "/feas_info_end.json", "w") as f:
        json.dump(dict_list, f)

    # # non_move_actions = ['RotateLeft_90', 'RotateRight_90', 'LookDown_15', 'LookUp_15']
    # non_move_actions = ['RotateLeft_90', 'RotateRight_90','nop']
    # # feasibility_list = []
    # # error_list = []
    # mask = None
    # smooth_nav = False
    # debug = False 
    # #these are the defaults for the args inputs

    # dict_list = []

    # for step_iter in range(n_rand_steps):
    #     one_dict = {}

    #     sampled_action = np.random.choice(non_move_actions)
    #     if sampled_action != 'nop':
    #         _, _, _, _, _ = env.va_interact(sampled_action, interact_mask=mask, smooth_nav=smooth_nav, debug=debug)
    #     #all actions in non_move_actions are ALWAYS feasible

    #     curr_image = Image.fromarray(np.uint8(env.last_event.frame))
    #     image_save_filepath = filepath + "/images/frame" + str(step_iter) +".jpg"
    #     print(image_save_filepath)
    #     with open(image_save_filepath,"w") as f:
    #         curr_image.save(f)

    #     success, _, _, err, _ = env.va_interact("MoveAhead_15", interact_mask=mask, smooth_nav=smooth_nav, debug=debug)

    #     # feasibility_list.append(success)
    #     # error_list.append(err)
    #     one_dict["success"] = success
    #     one_dict["error_msg"] = err
    #     dict_list.append(one_dict)


    # feasibilty_filepath = filepath + "/feas_info.csv"
    # with open(feasibilty_filepath, "w") as f:
    #     json.dump(one_dict , f)
    
    # feasibilty_filepath = filepath + "/feasibility_list.csv"
    # feasibility_list = np.array(feasibility_list)
    # with open(feasibilty_filepath, "w") as f:
    #     np.savetxt(f, feasibility_list, delimiter = ",")

    # error_msg_filepath = filepath + "/error_msg_list.csv"
    # error_msg_filepath = np.array(error_msg_filepath)
    # with open(error_msg_filepath,"w") as f:
    #     np.savetxt(f, error_list, delimiter = ",")

    


# traj_data_path = os.path.abspath("trial_traj_data.json")

filepath = os.path.abspath("unseen")



env = ThorEnv()

generate_feasibility_data(env, filepath)

env.stop()

