#%%
# %%
import json
path1 = "/home/ubuntu/alfred/data/json_2.1.0/train/look_at_obj_in_light-BasketBall-None-DeskLamp-301/trial_T20190907_025232_370454/traj_data.json"
path2 = "/home/ubuntu/alfred/data/json_2.1.0/train/look_at_obj_in_light-AlarmClock-None-DeskLamp-301/trial_T20190907_174127_043461/traj_data.json"

j1 = json.load(open(path1))['scene']['object_poses']
j2 = json.load(open(path2))['scene']['object_poses']
d1 = {j['objectName']: json.dumps(j) for j in j1}
d2 = {j['objectName']: json.dumps(j) for j in j2}
# %%
j1['scene']['object_poses'] == j2['scene']['object_poses']
# %%
set(j1['scene']['object_poses']) ^ set(j2['scene']['object_poses'])
# %%
from deepdiff import DeepDiff
# %%
len(set(d1.items()) - set(d2.items()))


# %%
len(d2)
# %%
