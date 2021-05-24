#%%
import sys
import os
%env ALFRED_ROOT="/home/ubuntu/alfred"
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))


import numpy as np
from PIL import Image
import json
from env.thor_env import ThorEnv
#%%
import sys
%env ALFRED_ROOT="/home/ubuntu/alfred"
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))

import numpy as np
from PIL import Image
import json

print("hello")
# %%

from env.thor_env import ThorEnv
# %%
!which python
# %%
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
# %%

traj_data_path = "trial_traj_data.json"
n_rand_steps = 10
filepath = "trial_1"

with open(traj_data_path) as f:
    traj_data = json.load(f)

env = ThorEnv()
# %%
env = ThorEnv()

# %%
setup_scene(env, traj_data)
# %%
curr_image = Image.fromarray(np.uint8(env.last_event.frame))
# %%
curr_image
# %%
env_store = env 
# %%
env_store.va_interact("RotateLeft_90", interact_mask=None, smooth_nav=None, debug=None)
# %%
Image.fromarray(np.uint8(env_store.last_event.frame))
# %%
env_store.va_interact("MoveAhead_15", interact_mask=None, smooth_nav=None, debug=None)
env_store.va_interact("RotateLeft_90", interact_mask=None, smooth_nav=None, debug=None)
env_store.va_interact("RotateLeft_90", interact_mask=None, smooth_nav=None, debug=None)
env_store.va_interact("MoveAhead_15", interact_mask=None, smooth_nav=None, debug=None)
env_store.va_interact("RotateLeft_90", interact_mask=None, smooth_nav=None, debug=None)
env_store.va_interact("RotateLeft_90", interact_mask=None, smooth_nav=None, debug=None)
Image.fromarray(np.uint8(env_store.last_event.frame))


# %%

def generate_feasibility_data(env, traj_data, n_rand_steps, filepath):

    setup_scene(env, traj_data)

    # nav_actions = ['MoveAhead_25', 'RotateLeft_90', 'RotateRight_90', 'LookDown_15', 'LookUp_15']
    #I have suspicion RotateLeft_90, RotateRight_90, LookUp_15 and LookDown_15 are always feasible, in which case can always set to one. But testing first.
    #nevertheless, understanding when MoveAhead_25 is feasible is very significant.
    #counter argument is that it may not be possilbe to infinitely look up. There may be a limit. Same applies for looking down.

    non_move_actions = ['RotateLeft_90', 'RotateRight_90', 'LookDown_15', 'LookUp_15']
    # rotational_actions = ['RotateLeft_90', 'RotateRight_90']
    feasibility_list = []

    mask = None
    smooth_nav = False
    debug = False 
    #these are the defaults for the args inputs

    for step_iter in range(n_rand_steps):

        sampled_action = np.random.choice(non_move_actions)
        _, _, _, _, _ = env.va_interact(sampled_action, interact_mask=mask, smooth_nav=smooth_nav, debug=debug)
        #all actions in non_move_actions are ALWAYS feasible

        curr_image = Image.fromarray(np.uint8(env.last_event.frame))
        image_save_filepath = filepath + "/images/frame" + str(step_iter) +".jpg"
        Image.save(image_save_filepath)

        success, _, _, err, _ = env_store.va_interact("MoveAhead_15", interact_mask=mask, smooth_nav=smooth_nav, debug=debug)

        feasibility_list.append(success == True)

        # for action_iter in range(nav_actions):
            
        #     action = nav_actions[action_iter]
        #     env_store = env   #THIS EQUALITY NEEDS TO BE BY COPY AND NOT REFERENCE. IF REFERENCE, THIS IS NEEDS CORRECTION.
        #     #If reference, then need to export and save current state, then load it after every action interaction. 

        #     success, _, _, err, _ = env_store.va_interact(action, interact_mask=mask, smooth_nav=smooth_nav, debug=debug)

        #     if not success:

        #         feasibility_matrix[step_iter, action_iter] = False
        #         print("Interact API failed with error:", err)

        #by the end of this for loop we know which actions are feasible for the current robot frame. 
        
        # possible_actions = nav_actions[feasibility_matrix[step_iter]]
        # sampled_action = np.random.choice(possible_actions)
        # _, _, _, _, _ = env.va_interact(sampled_action, interact_mask=mask, smooth_nav=smooth_nav, debug=debug)

    feasibilty_filepath = filepath + "/feasibility_list.csv"
    feasibility_list = np.array(feasibility_list)
    np.savetxt(feasibilty_filepath, feasibility_list, delimiter = ",")

# %%
env_store.va_interact("RotateLeft_15", interact_mask=None, smooth_nav=None, debug=None)

# %%
Image.fromarray(np.uint8(env_store.last_event.frame))

# %%

generate_feasibility_data(env, traj_data, n_rand_steps, filepath)

# %%
env.stop()
# %%

A = np.array([1,2,3])
# %%
feasibilty_filepath = "~/home/ubuntu/alfred/trial_folder" + "/feasibility_list.csv"
A = np.array(A)
np.savetxt(feasibilty_filepath, A, delimiter = ",")
# %%
import os
os.getcwd()
# %%

with open(feasibilty_filepath,"w") as f:
    np.savetxt(f, A, delimiter = ",")
# %%
os.path.abspath("hello.txt")
# %%
os.listdir()
# %%
Image.fromarray(np.uint8(env.last_event.frame))
# %%
env.va_interact("RotateLeft_90", interact_mask=None, smooth_nav=None, debug=None)

# %%
Image.fromarray(np.uint8(env.last_event.frame))

# %%
env.va_interact("MoveAhead_15", interact_mask=None, smooth_nav=None, debug=None)

# %%
env.random_initialize()
# %%
env.check_x_display()
# %%
env
# %%
env.
# %%
positions = env.step(action="GetReachablePositions").metadata["actionReturn"]
# %%
env.last_event.metadata['agent']
# %%
event = env.step(dict(action='GetReachablePositions'))

# %%
env.step(dict(action='GetReachablePositions')).metadata["actionReturn"]
# %%

# for each feasible location, do 4 rotations. Store image, agent position and rotation (using env.last_event.metadata['agent']).
# store scene num

# %%
traj_data["scene"]["init_action"]["x"]
# %%

from PIL import Image
with Image.open("/home/ubuntu/alfred/trial/images/pick_and_place_with_movable_recep-Knife-Mug-DiningTable-27_1.25_0.9010001_1.5_0.jpg") as im:
    im.show()


# %%

train_scene_path = os.path.abspath("data/json_2.1.0/train")
all_scene_paths = os.listdir(train_scene_path)

# for scene_path in all_scene_paths:
# %%
all_scene_paths
# %%
import json
scene_num_list= []
for scene_path in all_scene_paths:
    json_path = train_scene_path + "/" + scene_path + "/" + os.listdir(train_scene_path + "/" + scene_path)[0] + "/traj_data.json"

    with open(json_path) as f:
        traj_data = json.load(f)
    
    scene_num = traj_data['scene']['scene_num']

    scene_num_list.append(scene_num)
# %%
