# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling RVT or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
#
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from multiprocessing import Value

import numpy as np
import torch
from yarr.agents.agent import Agent
from yarr.envs.env import Env
from yarr.utils.transition import ReplayTransition
from yarr.agents.agent import ActResult

import os
import csv
import random
import json
import copy
import shutil
import pickle
import logging
import warnings
from PIL import Image
from collections import Counter
from scipy.spatial.transform import Rotation

from rlbench.backend import utils
from rlbench.backend.const import *

from rvt.data_aug.base import Action, WayPoint, Episode, DataAugmentor
from rvt.data_aug.heuristic import Heuristic

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning) 

# np.set_printoptions(formatter={'float': '{:0.3f}'.format})
np.set_printoptions(precision=5, suppress=True, linewidth=np.inf)


TASK_BOUND = [-0.075, -0.455, 0.752, 0.480, 0.455, 1.477]
PANDA_INIT_ACTION = np.array([2.78467745e-01, -8.16867873e-03, 1.47196412e+00, -2.93883204e-06, 9.92665470e-01, -2.89610603e-06, 1.20894387e-01, 1, 0])

# default rollout generator class with some additional functions
class RolloutGenerator(object):

    def __init__(self, env_device = 'cuda:0'):
        self._env_device = env_device

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def generator(self, step_signal: Value, env: Env, agent: Agent,
                  episode_length: int, timesteps: int,
                  eval: bool, eval_demo_seed: int = 0,
                  record_enabled: bool = False,
                  replay_ground_truth: bool = False):

        if eval:
            obs = env.reset_to_demo(eval_demo_seed)
            # get ground-truth action sequence
            if replay_ground_truth:
                actions, keypoints, dense_actions, waypoints = env.get_ground_truth_action(eval_demo_seed, 'heuristic', stopping_delta=0.06)
        else:
            obs = env.reset()
        agent.reset()
        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
        for step in range(episode_length):

            prepped_data = {k:torch.tensor(np.array([v]), device=self._env_device) for k, v in obs_history.items()}
            if not replay_ground_truth:
                act_result = agent.act(step_signal.value, prepped_data, deterministic=eval)
            else:
                if step >= len(actions):
                    return
                act_result = ActResult(actions[step])

            # Convert to np if not already
            agent_obs_elems = {k: np.array(v) for k, v in act_result.observation_elements.items()}
            extra_replay_elements = {k: np.array(v) for k, v in act_result.replay_elements.items()}

            transition = env.step(act_result)
            obs_tp1 = dict(transition.observation)
            timeout = False
            if step == episode_length - 1:
                # If last transition, and not terminal, then we timed out
                timeout = not transition.terminal
                if timeout:
                    transition.terminal = True
                    if "needs_reset" in transition.info:
                        transition.info["needs_reset"] = True

            obs_and_replay_elems = {}
            obs_and_replay_elems.update(obs)
            obs_and_replay_elems.update(agent_obs_elems)
            obs_and_replay_elems.update(extra_replay_elements)

            for k in obs_history.keys():
                obs_history[k].append(transition.observation[k])
                obs_history[k].pop(0)

            transition.info["active_task_id"] = env.active_task_id

            replay_transition = ReplayTransition(
                obs_and_replay_elems, act_result.action, transition.reward,
                transition.terminal, timeout, summaries=transition.summaries,
                info=transition.info)

            if transition.terminal or timeout:
                # If the agent gives us observations then we need to call act
                # one last time (i.e. acting in the terminal state).
                if len(act_result.observation_elements) > 0:
                    prepped_data = {k: torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}
                    act_result = agent.act(step_signal.value, prepped_data, deterministic=eval)
                    agent_obs_elems_tp1 = {k: np.array(v) for k, v in act_result.observation_elements.items()}
                    obs_tp1.update(agent_obs_elems_tp1)
                replay_transition.final_observation = obs_tp1

            if record_enabled and transition.terminal or timeout or step == episode_length - 1:
                env.env._action_mode.arm_action_mode.record_end(env.env._scene, steps=60, step_scene=True)

            obs = dict(transition.observation)

            yield replay_transition

            if transition.info.get("needs_reset", transition.terminal):
                return

    def check_and_make(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def save_rgb_and_depth_img(self, obs, save_path, keypoint_state):

        if obs is None:
            return
        
        # Save image data first, and then None the image data, and pickle
        left_shoulder_rgb_path = os.path.join(save_path, LEFT_SHOULDER_RGB_FOLDER)
        left_shoulder_depth_path = os.path.join(save_path, LEFT_SHOULDER_DEPTH_FOLDER)
        right_shoulder_rgb_path = os.path.join(save_path, RIGHT_SHOULDER_RGB_FOLDER)
        right_shoulder_depth_path = os.path.join(save_path, RIGHT_SHOULDER_DEPTH_FOLDER)
        wrist_rgb_path = os.path.join(save_path, WRIST_RGB_FOLDER)
        wrist_depth_path = os.path.join(save_path, WRIST_DEPTH_FOLDER)
        front_rgb_path = os.path.join(save_path, FRONT_RGB_FOLDER)
        front_depth_path = os.path.join(save_path, FRONT_DEPTH_FOLDER)

        self.check_and_make(left_shoulder_rgb_path)
        self.check_and_make(left_shoulder_depth_path)
        self.check_and_make(right_shoulder_rgb_path)
        self.check_and_make(right_shoulder_depth_path)
        self.check_and_make(wrist_rgb_path)
        self.check_and_make(wrist_depth_path)
        self.check_and_make(front_rgb_path)
        self.check_and_make(front_depth_path)

        left_shoulder_rgb = Image.fromarray(obs.left_shoulder_rgb)
        left_shoulder_depth = utils.float_array_to_rgb_image(obs.left_shoulder_depth, scale_factor=DEPTH_SCALE)
        right_shoulder_rgb = Image.fromarray(obs.right_shoulder_rgb)
        right_shoulder_depth = utils.float_array_to_rgb_image(obs.right_shoulder_depth, scale_factor=DEPTH_SCALE)
        wrist_rgb = Image.fromarray(obs.wrist_rgb)
        wrist_depth = utils.float_array_to_rgb_image(obs.wrist_depth, scale_factor=DEPTH_SCALE)
        front_rgb = Image.fromarray(obs.front_rgb)
        front_depth = utils.float_array_to_rgb_image(obs.front_depth, scale_factor=DEPTH_SCALE)

        left_shoulder_rgb.save(os.path.join(left_shoulder_rgb_path, f"{keypoint_state}.png"))
        left_shoulder_depth.save(os.path.join(left_shoulder_depth_path, f"{keypoint_state}.png"))
        right_shoulder_rgb.save(os.path.join(right_shoulder_rgb_path, f"{keypoint_state}.png"))
        right_shoulder_depth.save(os.path.join(right_shoulder_depth_path, f"{keypoint_state}.png"))
        wrist_rgb.save(os.path.join(wrist_rgb_path, f"{keypoint_state}.png"))
        wrist_depth.save(os.path.join(wrist_depth_path, f"{keypoint_state}.png"))
        front_rgb.save(os.path.join(front_rgb_path, f"{keypoint_state}.png"))
        front_depth.save(os.path.join(front_depth_path, f"{keypoint_state}.png"))

        front_rgb.save(os.path.join(front_rgb_path, f"current.png"))

    def save_low_dim(self, obs, save_path, keypoint_state):
        if obs is None:
            return

        low_dim_obs_path = os.path.join(save_path, "obs.pkl")

        if os.path.exists(low_dim_obs_path):
            with open(low_dim_obs_path, 'rb') as file:
                low_dim_obs_dict = pickle.load(file)
        else:
            low_dim_obs_dict = {}

        obs_copy = copy.deepcopy(obs)

        obs_copy.left_shoulder_rgb = None
        obs_copy.left_shoulder_depth = None
        obs_copy.left_shoulder_point_cloud = None
        obs_copy.left_shoulder_mask = None
        obs_copy.right_shoulder_rgb = None
        obs_copy.right_shoulder_depth = None
        obs_copy.right_shoulder_point_cloud = None
        obs_copy.right_shoulder_mask = None
        obs_copy.overhead_rgb = None
        obs_copy.overhead_depth = None
        obs_copy.overhead_point_cloud = None
        obs_copy.overhead_mask = None
        obs_copy.wrist_rgb = None
        obs_copy.wrist_depth = None
        obs_copy.wrist_point_cloud = None
        obs_copy.wrist_mask = None
        obs_copy.front_rgb = None
        obs_copy.front_depth = None
        obs_copy.front_point_cloud = None
        obs_copy.front_mask = None

        obs_copy.joint_velocities = None
        obs_copy.joint_positions = None
        obs_copy.joint_forces = None
        # obs_copy.gripper_open = gripper_open
        # obs_copy.gripper_pose = gripper_pose
        obs_copy.gripper_matrix = None
        # obs_copy.gripper_joint_positions = gripper_joint_positions
        obs_copy.gripper_touch_forces = None
        obs_copy.task_low_dim_state = None
        # obs_copy.ignore_collisions = ignore_collisions
        # obs_copy.misc = misc

        low_dim_obs_dict[keypoint_state] = obs_copy

        with open(os.path.join(save_path, "obs.pkl"), 'wb') as file:
            pickle.dump(low_dim_obs_dict, file)

        # # Save the low-dimension data
        # with open(os.path.join(save_path, LOW_DIM_PICKLE), 'wb') as f:
        #     pickle.dump(demo, f)

        # with open(os.path.join(save_path, VARIATION_NUMBER), 'wb') as f:
        #     pickle.dump(variation, f)

    def check_within_bound(self, waypoint):
        if (waypoint[0] > TASK_BOUND[0] and waypoint[0] < TASK_BOUND[3]) and \
           (waypoint[1] > TASK_BOUND[1] and waypoint[1] < TASK_BOUND[4]) and \
           (waypoint[2] > TASK_BOUND[2] and waypoint[2] < TASK_BOUND[5]):
            return True
        else:
            return False

    def check_correction_distance(self, perturbed, keypoint, min, max):
        if min < np.linalg.norm(perturbed - keypoint) < max:
            print(np.linalg.norm(perturbed - keypoint))
            return True
        else:
            return False
        
    def annotate_transition(self, current_wp, next_wp, perturbing=False):
        transitions = []
        action_descriptions = []

        # Convert quaternions to Euler angles (XYZ convention)
        current_euler = Rotation.from_quat(current_wp[3:7]).as_euler('XYZ', degrees=True)
        next_euler = Rotation.from_quat(next_wp[3:7]).as_euler('XYZ', degrees=True)

        # Compute differences between waypoints
        pos_diff = next_wp[:3] - current_wp[:3]
        ori_diff = np.abs(next_euler - current_euler)
        gripper_diff = next_wp[7] - current_wp[7]
        ignore_collision = next_wp[8]
        transitions.append(np.hstack([pos_diff, ori_diff, gripper_diff, ignore_collision]))

        actions = []
        action_dict = {"failure_explanation": None, "instruction": [], "gripper_open": None, "collision_allow": ignore_collision}

        if gripper_diff == 1:
            action_dict["gripper_open"] = True
        elif gripper_diff == -1:
            action_dict["gripper_open"] = False

        if ignore_collision:
            action_dict["collision_allow"] = True

        # Determine action based on heuristics
        translation_magnitude = np.linalg.norm(pos_diff)
        small_thres = 0.01
        large_thres = 0.05
        rotation_threshold = 10
        action_performed = False

        if translation_magnitude > small_thres:
            translation_type = "L" if translation_magnitude > large_thres else "S"
            directions = [("down", 2, -1), ("up", 2, 1), ("backward", 0, -1), ("forward", 0, 1), ("right", 1, 1), ("left", 1, -1)]
            for direction, axis, sign in directions:
                if sign * pos_diff[axis] > small_thres:
                    action_performed = True
                    action_dict["instruction"].append(f"move {direction} {translation_type}")

        if np.any(ori_diff > rotation_threshold):
            action_performed = True
            if (ori_diff[0] > rotation_threshold and ori_diff[1] > rotation_threshold and ori_diff[2] > rotation_threshold):
                action_dict["instruction"].append("rotate")
            else:
                axes = ["x", "y", "z"]
                for axis, diff in zip(axes, ori_diff):
                    if diff > rotation_threshold:
                        action_dict["instruction"].append(f"rotate about {axis}-axis")

        if gripper_diff != 0:
            action_performed = True
            action = "open gripper" if gripper_diff == 1 else "close gripper"
            action_dict["instruction"].append(action)

        if not action_performed:
            action_dict["instruction"].append("unknown action")

        if perturbing:
            action_dict["failure_explanation"] = "failed bc you " + ", ".join(action_dict["instruction"])
            action_dict["instruction"] = []

        action_descriptions.append(action_dict)

        return action_descriptions

    def perturb_keypoint(self, kypt_idx, keypoints_with_init, actions_with_init, dense_actions, waypoints, skip_intermediate):
        # perturbation injection
        # keypoints: [56, 79, 101, 138, 147, 185] (x start from 0)
        # the perturbed idx is where failure is injected. we fallback to a previous waypoint and reapproach the same waypoint correctly
        
        # Currently checks the following:
        # 1. Perturb rotation for all roll, pitch, yaw.
        #   -> Or do we want to focus on the yaw perturbations? The yaw alignment issue was prevalent in picking up action failure modes.
        # 2. Toggle gripper by 0.1 probability.
        # 3. Perturb i) x, y only, ii) z only, iii) x, y, z by small amount. Chosen uniform random.
        #   -> One problem in iii) is that when the correction happens it may cause the model to learn go backwards on step?
        # 4. If the last keypoint is rotation-only transition, don't perturb. Likely to be "screw" action.
        perturbed_idx = []
        corrective_idx = []
        resolution = 5

        # noise params
        translation_noise = 0.01
        rotation_noise = 45
        min_dist_thres = 0.02
        max_dist_thres = 0.05

        random_seed = int.from_bytes(os.urandom(4), 'big')
        random.seed(random_seed)
        np.random.seed(random_seed)
        print("Random seed:", random_seed)

        # Perturbation injection to expert keypoint to induce failure
        perturbed_action = actions_with_init[kypt_idx].copy()

        # 1. 
        # euler = quaternion_to_euler(perturbed_action[3:7], resolution).astype(np.float64)
        # rot_perturbation = np.random.normal(0, rotation_noise, size=(1,))
        # euler[2] += rot_perturbation # only yaw
        # disc = Rotation.from_euler("xyz", euler, degrees=True).as_quat()
        # perturbed_action[3:7] = disc

        # 2.
        # gripper_toggle = np.random.choice([0, 1], p=[0.9, 0.1])
        # # print(f"Toggle gripperZ: {bool(gripper_toggle)}")
        # if gripper_toggle:
        #     if perturbed_action[8]:
        #         perturbed_action[8] = 0.0
        #     else:
        #         perturbed_action[8] = 1.0
        
        # 3.
        perturbed_action_ckpt = perturbed_action[0:3].copy()

        
        translation_perturbation_mode = np.random.choice([0, 1])
        while True:
            # print(f"Translation perturbation mode: {translation_perturbation_mode}")
            perturbed_action[0:3] = perturbed_action_ckpt  # reset
            # for debugging; comment out when running data aug
            translation_perturbation_mode = 2
            if translation_perturbation_mode == 0:
                # perturb only x and y
                xy_perturbation = np.random.normal(0, translation_noise, size=(2,))
                perturbed_action[0:2] += xy_perturbation
            # elif translation_perturbation_mode == 1:
            #     # perturb only z
            #     z_perturbation = -1
            #     while z_perturbation <= 0:
            #         z_perturbation = np.random.normal(0, translation_noise)
            #     perturbed_action[2] += z_perturbation
            else:
                # perturb all x, y, and z
                xy_perturbation = np.random.normal(0, translation_noise, size=(2,))
                perturbed_action[0:2] += xy_perturbation
                # z_perturbation = -1
                # while z_perturbation <= 0:
                z_perturbation = np.random.normal(0, translation_noise)
                perturbed_action[2] += z_perturbation
            # print(np.linalg.norm(perturbed_action[0:3] - actions[idx + p_count * 2,:][0:3]))
            if self.check_within_bound(perturbed_action[0:3]):
                # print(np.linalg.norm(perturbed_action[0:3] - actions[idx + p_count * 2,:][0:3]))
                if 0.02 < np.linalg.norm(perturbed_action[0:3] - actions_with_init[kypt_idx][0:3]) < 0.05:
                    break

        if not skip_intermediate:
            # Sampling corrective behavior
            # randomly sample a correct waypoint from a segment of waypoints leading up to the expert keypoint
            half = (keypoints_with_init[kypt_idx] - keypoints_with_init[kypt_idx-1]) // 4
            half = 0
            waypoint_segment = waypoints[keypoints_with_init[kypt_idx-1] + half : keypoints_with_init[kypt_idx] - half]
            # count = 0
            while True:
                random_idx = np.random.choice(waypoint_segment)
                if self.check_correction_distance(dense_actions[random_idx][0:3], actions_with_init[kypt_idx][0:3], min_dist_thres, max_dist_thres):
                    # print(random_idx)
                    break
                # count += 1
                # if count == 5:
                #     break
            # random_idx = np.random.choice(waypoint_segment)
            # print(random_idx)
            corrective_idx.append(random_idx)
            intermediate_action = dense_actions[random_idx]
        else:
            intermediate_action = None

        print(actions_with_init[kypt_idx])
        print(perturbed_action)
        print(intermediate_action)

        return perturbed_action, intermediate_action

# expert replay with optional perturbation given keypoint index (kypt idx) to perturb
class ExpertReplay(RolloutGenerator):

    def __init__(self, env_device = 'cuda:0'):
        super().__init__(env_device)

    def update_subgoal_language_description(self, language_description, curr_kypt, perturb_num, action_dict, act, curr_kypt_idx, perturb_config, prev_kypt=None):
        if action_dict['type'] == 'perturb':
            language_description["subgoal"][f"{curr_kypt}"]["fail"].setdefault(f"perturb_{perturb_num}", {})
            language_description["subgoal"][f"{curr_kypt}"]["fail"][f"perturb_{perturb_num}"]["label"] = "failure"
            language_description["subgoal"][f"{curr_kypt}"]["fail"][f"perturb_{perturb_num}"].setdefault("lang", {})
            transition_lang = super().annotate_transition(act[prev_kypt]['expert_action']['action'], act[curr_kypt]['perturb_action']['action'], perturbing=True) # from prev kypt expert action -> perturbed curr kypt 
            language_description["subgoal"][f"{curr_kypt}"]["fail"][f"perturb_{perturb_num}"]["lang"].setdefault("reflection", json.dumps(transition_lang))
            language_description["subgoal"][f"{curr_kypt}"]["fail"][f"perturb_{perturb_num}"].setdefault("failure_state", json.dumps(np.round(action_dict['action'],3).tolist())) # = act[curr_kypt]['perturb_action']

        if action_dict['type'] == 'expert' and curr_kypt_idx in perturb_config:
            transition_lang = super().annotate_transition(act[curr_kypt]['perturb_action']['action'], act[curr_kypt]['expert_action']['action'], perturbing=False) # from perturbed curr kypt -> back to correct curr kypt
            language_description["subgoal"][f"{curr_kypt}"]["fail"][f"perturb_{perturb_num}"]["lang"].setdefault("instruction", json.dumps(transition_lang))
            language_description["subgoal"][f"{curr_kypt}"]["fail"][f"perturb_{perturb_num}"].setdefault("correction_action", json.dumps(np.round(action_dict['action'],3).tolist())) # = act[curr_kypt]['expert_action']

        if action_dict['type'] == 'intermediate':
            language_description["subgoal"][f"{curr_kypt}"]["ongoing"].setdefault(f"perturb_{perturb_num}", {})
            language_description["subgoal"][f"{curr_kypt}"]["ongoing"][f"perturb_{perturb_num}"]["label"] = "ongoing"
            transition_lang = super().annotate_transition(act[curr_kypt]['intermediate_action']['action'], act[curr_kypt]['expert_action']['action'], perturbing=False) # from intermediate curr kypt -> back to correct curr kypt
            language_description["subgoal"][f"{curr_kypt}"]["ongoing"][f"perturb_{perturb_num}"].setdefault("lang", json.dumps(transition_lang))
            language_description["subgoal"][f"{curr_kypt}"]["ongoing"][f"perturb_{perturb_num}"].setdefault("action", json.dumps(np.round(act[curr_kypt]['expert_action']['action'],3).tolist()))

    def create_episode(self, dense_actions, dense_action_indices, keypoint_actions, keypoint_action_indices):
        episode = Episode([])

        for i, action in enumerate(dense_actions):
            if i in keypoint_action_indices:
                waypoint_type = 'keypoint'
                a = keypoint_action_indices.index(i)
                action = keypoint_actions[keypoint_action_indices.index(i)]
            else:
                waypoint_type = 'intermediate'
                action = dense_actions[dense_action_indices.index(i)]

            translation = action[:3]
            rotation = action[3:7]
            gripper_open = bool(action[7])
            ignore_collision = bool(action[8])

            waypoint = WayPoint(i, waypoint_type, Action(translation, rotation, gripper_open, ignore_collision))
            episode.waypoints.append(waypoint)

        return episode

    def generator(self, env: Env, episode_length: int, log_dir, task_name, 
                  episode_number, language_description, eval_demo_seed: int = 0,
                  record_enabled: bool = False, perturb_num: int = 0, skip_wypt: bool = False, num_wypt_set: set = (), skip_and_continue: bool = False):
        
        # episode save path (imgs and low dim obs)
        episode_folder = os.path.join(log_dir, task_name, str(episode_number))

        # reset
        obs, obs_copy = env.reset_to_demo(eval_demo_seed)

        # expert actions: 9D action = 7D ee pose + 1D gripper open + 1D ignore collision
        kypt_actions, kypt_idx, dense_actions, dense_idx = env.get_ground_truth_action(eval_demo_seed, 'heuristic', stopping_delta=0.1)            
        # dense actions already include the init pose of the robot
        kypt_idx_w_init = [0] + kypt_idx
        kypt_actions_w_init = np.vstack([PANDA_INIT_ACTION, kypt_actions])

        episode = self.create_episode(dense_actions, dense_idx.tolist(), kypt_actions_w_init, kypt_idx_w_init)
        heuristic_data_augmenter = Heuristic(task_name)
        perturbed_episode = heuristic_data_augmenter.heuristic_perturb(episode, N=1)

        # Set true if we want to skip demos with same number of keypoints as before
        if skip_wypt:
            if len(kypt_idx) in num_wypt_set:
                skip_and_continue[0] = True
                shutil.rmtree(os.path.join(log_dir, task_name, str(episode_number)))
                return
            else:
                skip_and_continue[0] = False
                num_wypt_set.add(len(kypt_idx))

        # dict to track
        # new or load from existing language_description.json
        if "task" not in language_description:
            language_description["task"] = env._lang_goal
        if "init" not in language_description:
            language_description["init"] = json.dumps(np.round(PANDA_INIT_ACTION,3).tolist())
        if "keypoints" not in language_description:
            language_description["keypoints"] = str(kypt_idx_w_init)
        if "expert_keypoints_length" not in language_description:
            language_description["expert_keypoints_length"] = str(len(kypt_idx_w_init))
        if "subgoal" not in language_description:
            language_description["subgoal"] = {}
        
        kypt_perturb, _ = perturbed_episode.return_()
        step = 0
        for curr_kypt_idx, curr_kypt in enumerate(kypt_perturb):
            # we will loop through this action_buffer later
            action_buffer = []

            language_description["subgoal"].setdefault(f"{curr_kypt.id}", {})
            language_description["subgoal"][f"{curr_kypt.id}"]["label"] = "start" if curr_kypt.id == 0 else "success"
            prev_kypt = kypt_perturb[curr_kypt_idx - 1] if curr_kypt_idx > 1 else None
            next_kypt = kypt_perturb[curr_kypt_idx + 1] if curr_kypt_idx < len(kypt_perturb) - 1 else None

            if curr_kypt_idx < len(kypt_idx_w_init) - 1:
                language_description["subgoal"][f"{curr_kypt.id}"]["lang"] = json.dumps(super().annotate_transition(curr_kypt.action.action_to_array(), next_kypt.action.action_to_array()))
                language_description["subgoal"][f"{curr_kypt.id}"]["action"] = json.dumps(np.round(next_kypt.action.action_to_array(), 3).tolist())


            if curr_kypt.perturbations:
                perturbation = curr_kypt.perturbations[episode_number]
                action_buffer.append({'action': perturbation.mistake.action.action_to_array(), 'type': 'perturb'}) # perturbing action
                language_description["subgoal"][f"{curr_kypt.id}"].setdefault("fail", {})
                if perturbation.correction is not None:
                    action_buffer.append({'action': perturbation.correction.action.action_to_array(), 'type': 'intermediate'}) # optional correcting action
                    language_description["subgoal"][f"{curr_kypt.id}"].setdefault("ongoing", {})
            action_buffer.append({'action': curr_kypt.action.action_to_array(), 'type': 'expert'}) # expert action after optional perturbations to the same kypt
            
            for action_dict in action_buffer:
                # Step through environment
                act_result = ActResult(action_dict['action'])
                transition, obs_copy = env.step(act_result)

                # unique identifier for saving imgs at each timestep
                # expert -> perturb -> intermediate -> expert -> perturb & skip intermediate -> expert
                keypoint_state = f"{curr_kypt.id}_{action_dict['type']}"                
                print(f"Step {step} -> {step+1}|  Action: {np.round(act_result.action, 3)} --> (keypoint {keypoint_state})")
                
                # annotate each step of action transition
                # self.update_subgoal_language_description(language_description, curr_kypt.id, perturb_num, action_dict, act, curr_kypt_idx, perturb_config, prev_kypt)

                # handling timeout and terminal condition
                obs_tp1 = dict(transition.observation)
                timeout = False
                if step == episode_length - 1:
                    # If last transition, and not terminal, then we timed out
                    timeout = not transition.terminal
                    if timeout:
                        transition.terminal = True
                        if "needs_reset" in transition.info:
                            transition.info["needs_reset"] = True

                transition.info["active_task_id"] = env.active_task_id

                replay_transition = ReplayTransition({}, act_result.action, transition.reward,
                    transition.terminal, timeout, summaries=transition.summaries, info=transition.info)

                if transition.terminal or timeout:
                    replay_transition.final_observation = obs_tp1

                if record_enabled and transition.terminal or timeout or step == episode_length - 1:
                    env.env._action_mode.arm_action_mode.record_end(env.env._scene, steps=60, step_scene=True)

                yield replay_transition

                # save imgs and low dim obs
                super().save_rgb_and_depth_img(obs_copy, episode_folder, keypoint_state)
                super().save_low_dim(obs_copy, episode_folder, keypoint_state)
                step += 1

            if curr_kypt_idx == len(kypt_idx_w_init) - 1:
                def numpy_encoder(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()  # Convert NumPy array to list
                    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
                json_data = json.dumps(action_buffer, default=numpy_encoder, indent=4)
                with open(os.path.join(episode_folder, "action.json"), "w") as file:
                    file.write(json_data)

            if transition.info.get("needs_reset", transition.terminal):
                print()
                return
            

class InteractiveRollout(RolloutGenerator):
    def __init__(self, env_device = 'cuda:0'):
        super().__init__(env_device)

    def get_user_input_action(self):
        user_action_str = input("Enter proposed action:")
        user_action = np.array([float(item) for item in user_action_str.split()])
        return user_action

    def generator(self, env: Env, episode_length: int, log_dir, task_name, 
                    episode_number, eval_demo_seed: int = 0,
                    record_enabled: bool = False, interactive: bool = True):
            
            # episode save path (imgs and low dim obs)
            episode_folder = os.path.join(log_dir, task_name, str(episode_number))

            # reset
            obs, obs_copy = env.reset_to_demo(eval_demo_seed)

            # expert actions: 9D action = 7D ee pose + 1D gripper open + 1D ignore collision
            kypt_actions, kypt_idx, dense_actions, dense_idx = env.get_ground_truth_action(eval_demo_seed, 'heuristic', stopping_delta=0.1)            
            kypt_idx_with_init = [0] + kypt_idx
            kypt_actions_with_init = np.vstack([PANDA_INIT_ACTION, kypt_actions])


            # init modifiable action dict with expert actions
            act = {
                kp: {
                    'expert_action': {'action': action, 'type': 'expert'},
                    'perturb_action': {'action': '', 'type': 'perturb'},
                    'intermediate_action': {'action': '', 'type': 'intermediate'}
                }
                for kp, action in zip(kypt_idx_with_init, kypt_actions_with_init)
            }

            # keypoints to perturb (probs as idx of keypoints assuming there are equal num of keypoints across demos per task)
            step = 0
            for curr_kypt_idx, curr_kypt in enumerate(kypt_idx_with_init):
                # init per keypoint
                actions_to_execute = [act[curr_kypt]['expert_action']]

                # execute action buffer (expert or perturb+expert or perturb+intermediate+expert)
                for i, action_dict in enumerate(actions_to_execute):

                    # Step through environment
                    act_result = ActResult(action_dict['action'])
                    print(f"Step {step} | Original expert action: {np.array2string(act_result.action, max_line_width=np.inf)}")
                    if interactive:
                        act_result.action = self.get_user_input_action()

                    print(f"Step {step} | User input action: {np.round(act_result.action, 3)}")

                    transition, obs_copy = env.step(act_result)

                    # handling timeout and terminal condition
                    obs_tp1 = dict(transition.observation)
                    timeout = False
                    if step == episode_length - 1:
                        # If last transition, and not terminal, then we timed out
                        timeout = not transition.terminal
                        if timeout:
                            transition.terminal = True
                            if "needs_reset" in transition.info:
                                transition.info["needs_reset"] = True

                    transition.info["active_task_id"] = env.active_task_id

                    replay_transition = ReplayTransition({}, act_result.action, transition.reward,
                        transition.terminal, timeout, summaries=transition.summaries, info=transition.info)

                    if transition.terminal or timeout:
                        replay_transition.final_observation = obs_tp1

                    if record_enabled and transition.terminal or timeout or step == episode_length - 1:
                        env.env._action_mode.arm_action_mode.record_end(env.env._scene, steps=60, step_scene=True)

                    yield replay_transition

                    # save imgs and low dim obs
                    keypoint_state = f"{curr_kypt}_{action_dict['type']}" 
                    super().save_rgb_and_depth_img(obs_copy, episode_folder, keypoint_state)
                    super().save_low_dim(obs_copy, episode_folder, keypoint_state)
                    step += 1

                if curr_kypt_idx == len(kypt_idx_with_init) - 1:
                    def numpy_encoder(obj):
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()  # Convert NumPy array to list
                        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
                    json_data = json.dumps(act, default=numpy_encoder, indent=4)
                    with open(os.path.join(episode_folder, "action.json"), "w") as file:
                        file.write(json_data)

                if transition.info.get("needs_reset", transition.terminal):
                    print()
                    return