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
import random
import numpy as np
import torch
from yarr.agents.agent import Agent
from yarr.envs.env import Env
from yarr.utils.transition import ReplayTransition
from yarr.agents.agent import ActResult

from rvt.utils.peract_utils import CAMERAS
from rvt.libs.peract.helpers.demo_loading_utils import keypoint_discovery
from waypoint_extraction.extract_waypoints import dp_waypoint_selection
from rvt.mvt.aug_utils import quaternion_to_discrete_euler, sensitive_gimble_fix, discrete_euler_to_quaternion, quaternion_to_euler
from scipy.spatial.transform import Rotation

from rvt.libs.RLBench.rlbench.backend.waypoints import Waypoint, Point

import os
import shutil
import csv
from collections import Counter
from PIL import Image
import logging

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG)
np.set_printoptions(formatter={'float': '{:0.3f}'.format})

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

TASK_BOUND = [-0.075, -0.455, 0.752, 0.480, 0.455, 1.100]

class RolloutGenerator(object):

    def __init__(self, env_device = 'cuda:0'):
        self._env_device = env_device

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype
    
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
        
    def annotate_transitions(self, waypoints):
        transitions = []
        action_descriptions = []
        grasping = False

        for i in range(len(waypoints) - 1):
            current_wp = waypoints[i]
            next_wp = waypoints[i + 1]

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

            if grasping:
                txt = "State: Holding(object) "
                if ignore_collision:
                    txt += "AllowCollision"
                actions.append(txt + " |")
            else:
                if ignore_collision:
                    actions.append("State: AllowCollision |")
                else:
                    actions.append("State: None |")
            actions.append("Actions: ")
            # Determine action based on heuristics\
            translation_magnitude = np.linalg.norm(pos_diff)
            small_thres = 0.01
            large_thres = 0.05
            rotation_threshold = 10
            if translation_magnitude > small_thres:
                translation_type = "L" if translation_magnitude > large_thres else "S"
                if pos_diff[2] < -small_thres:
                    actions.append(f"move down {translation_type}.")
                elif pos_diff[2] > small_thres:
                    actions.append(f"move up {translation_type}.")
                if pos_diff[0] < -small_thres:
                    actions.append(f"move backward {translation_type}.")
                elif pos_diff[0] > small_thres:
                    actions.append(f"move forward {translation_type}.")
                if pos_diff[1] > small_thres:
                    actions.append(f"move right {translation_type}.")
                elif pos_diff[1] < -small_thres:
                    actions.append(f"move left {translation_type}.")

            if np.any(ori_diff > rotation_threshold):
                if (ori_diff[0] > rotation_threshold and ori_diff[1] > rotation_threshold and ori_diff[2] > rotation_threshold):
                    actions.append("rotate.")
                else:
                    if ori_diff[0] > rotation_threshold:
                        actions.append("rotate about x-axis.")
                    if ori_diff[1] > rotation_threshold:
                        actions.append("rotate about y-axis.")
                    if ori_diff[2] > rotation_threshold:
                        actions.append("rotate about z-axis.")
            # else:
            #     actions.append("negligible rotation")

            if gripper_diff == 1:
                actions.append("open gripper.")
                grasping = False
            elif gripper_diff == -1:
                actions.append("close gripper.")
                # if ignore_collision:
                grasping = True

            if not actions:
                actions.append("unknown action")

            action_descriptions.append(" ".join(actions))

        return transitions, action_descriptions


    def generator(self, step_signal: Value, env: Env, agent: Agent,
                  episode_length: int, timesteps: int,
                  eval: bool, log_dir, task_name, episode_number, eval_demo_seed: int = 0,
                  record_enabled: bool = False,
                  replay_ground_truth: bool = False,
                  perturb: bool = False,
                  interactive: bool = False, num_wypt_set: set = (), skip_wypt: bool = False
                  ):
        multiview_img_folder = os.path.join(log_dir, task_name, str(episode_number), "multiview")
        # 1. reset
        # 2. initial obs
        if eval:
            obs = env.reset_to_demo(eval_demo_seed) # this is using the same initial config of a test data
            # get ground-truth action sequence
            if replay_ground_truth:
                actions, keypoints, dense_actions, waypoints = env.get_ground_truth_action(eval_demo_seed, 'heuristic') # 'dense' / 'heuristic' / 'awe'
                if skip_wypt:
                    if len(keypoints) in num_wypt_set:
                        shutil.rmtree(os.path.join(log_dir, task_name, str(episode_number)))
                        return
                    else:
                        num_wypt_set.add(len(keypoints))
                actions_with_init = np.vstack([np.array([-0.30895138, 0, 0.82001764, 0.70493394, -0.05539087, 0.70493394, 0.05539087, 1, 0]), actions])
                transitions, languages = self.annotate_transitions(actions_with_init)
                for i, lang in enumerate(languages):
                    print(i+1, lang)
                    languages[i] = f"{i}. {lang}"
                with open(os.path.join(multiview_img_folder, "transitions.csv"), 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    for transition in transitions:
                        writer.writerow(np.round(transition,1))
                with open(os.path.join(multiview_img_folder, "fg_lang.csv"), 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    for fg_lang in languages:
                        writer.writerow([fg_lang])
                
        else:
            obs = env.reset()

        if agent is not None:
            agent.reset()

        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
        count = 0
        prev_act = np.zeros((9,))
        user_has_control = False

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
        translation_noise = 0.2
        rotation_noise = 45
        min_dist_thres = 0.02
        max_dist_thres = 0.05

        if replay_ground_truth & perturb:
            perturbed_idx = list(range(len(keypoints))) # perurb 79 = keypoints[peturbed_idx[0]]
            p_count = 0 
            for i, idx in enumerate(perturbed_idx):
                # Perturbation injection to expert keypoint to induce failure
                perturbed_action = actions[idx + p_count * 2,:].copy()

                # 4.
                if i == len(perturbed_idx) - 1:
                    prev_action = actions[perturbed_idx[idx-1] + p_count * 2,:].copy()
                    # print(np.sum(np.abs(prev_action[0:2] - perturbed_action[0:2])))
                    if np.sum(np.abs(prev_action[0:2] - perturbed_action[0:2])) < 0.001:
                        # print("continue")
                        perturbed_idx.remove(idx)
                        break

                # 1. 
                # euler = quaternion_to_euler(perturbed_action[3:7], resolution).astype(np.float64)
                # rot_perturbation = np.random.normal(0, rotation_noise, size=(1,))
                # euler[2] += rot_perturbation # only yaw
                # disc = Rotation.from_euler("xyz", euler, degrees=True).as_quat()
                # perturbed_action[3:7] = disc

                # 2.
                gripper_toggle = np.random.choice([0, 1], p=[0.9, 0.1])
                # print(f"Toggle gripperZ: {bool(gripper_toggle)}")
                if gripper_toggle:
                    if perturbed_action[8]:
                        perturbed_action[8] = 0.0
                    else:
                        perturbed_action[8] = 1.0
                
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
                        z_perturbation = -1
                        while z_perturbation <= 0:
                            z_perturbation = np.random.normal(0, translation_noise)
                        perturbed_action[2] += z_perturbation
                    # print(np.linalg.norm(perturbed_action[0:3] - actions[idx + p_count * 2,:][0:3]))
                    if self.check_within_bound(perturbed_action[0:3]):
                        # print(np.linalg.norm(perturbed_action[0:3] - actions[idx + p_count * 2,:][0:3]))
                        if 0.02 < np.linalg.norm(perturbed_action[0:3] - actions[idx + p_count * 2,:][0:3]) < 0.05:
                            break


                # Sampling corrective behavior
                # randomly sample a correct waypoint from a segment of waypoints leading up to the expert keypoint
                if idx == 0:
                    half = keypoints[idx] // 4
                    half = 0
                    waypoint_segment = waypoints[0 + half : keypoints[idx] - half]
                else:
                    half = (keypoints[idx] - keypoints[idx-1]) // 4
                    half = 0
                    waypoint_segment = waypoints[keypoints[idx-1] + half : keypoints[idx] - half]
                while True:
                    random_idx = np.random.choice(waypoint_segment)
                    if self.check_correction_distance(dense_actions[random_idx][0:3], actions[idx + p_count * 2,:][0:3], min_dist_thres, max_dist_thres):
                        # print(random_idx)
                        break
                # random_idx = np.random.choice(waypoint_segment)
                # print(random_idx)
                corrective_idx.append(random_idx)
                corrective_action = dense_actions[random_idx]

                # insert perturbed action and correction action to actions
                actions = np.insert(actions, p_count * 2 + idx, np.vstack([perturbed_action, corrective_action]), axis=0)
                p_count += 1
            # print(np.round(actions,3))
                
        actions_with_init = np.vstack([np.array([-0.30895138, 0, 0.82001764, 0.70493394, -0.05539087, 0.70493394, 0.05539087, 1, 0]), actions])
        languages = self.annotate_transitions(actions_with_init)
        for i, lang in enumerate(languages):
            print(i+1, lang)

        # 3. stepping
        step = 0
        perturb_count = 0
        while True:
            for cam in CAMERAS:
                rgb = obs[f'{cam}_rgb'] # (3, IMAGE_SIZE, IMAGE_SIZE)
                rgb = Image.fromarray(rgb.T).rotate(-90)
                if not replay_ground_truth:
                    rgb.save(os.path.join(multiview_img_folder, f"{cam}", f"{step}.png"))
                else:
                    if step == 0:
                        img_name = 0
                    else:
                        if perturb_count <= len(perturbed_idx) - 1:
                            if step == 2 * perturb_count + perturbed_idx[perturb_count] + 1:
                                img_name = f"{keypoints[step - 2 * perturb_count - 1]}_1_perturbed"
                            elif step == 2 * perturb_count + perturbed_idx[perturb_count] + 2:
                                img_name = f"{keypoints[step - 2 * perturb_count - 2]}_2_corrected"
                                perturb_count += 1
                            else:
                                img_name = f"{keypoints[step - 2 * perturb_count - 1]}"
                        else:
                            img_name = f"{keypoints[step - 2 * perturb_count - 1]}"
                    rgb.save(os.path.join(multiview_img_folder, f"{cam}", f"{img_name}.png"))
                if interactive:
                    rgb.save(os.path.join(multiview_img_folder, f"{cam}", "current.png"))

            prepped_data = {k:torch.tensor(np.array([v]), device=self._env_device) for k, v in obs_history.items()}
            
            # 3-1. take RVT or GT action
            # 9D action = 7D ee pose + 1D gripper open + 1D ignore collision
            if not replay_ground_truth and agent is not None:
                act_result = agent.act(step_signal.value, prepped_data, deterministic=eval)
            else:
                if step >= len(actions):
                    return
                act_result = ActResult(actions[step]) # selects the ith keypoint as action
                if perturb_count <= len(perturbed_idx) - 1:
                    if step == 2 * perturb_count + perturbed_idx[perturb_count]:
                        print(f"Step {step} | pred action: {np.round(act_result.action, 3)} --> (perturbed waypoint {keypoints[step - 2 * perturb_count]})")
                    elif step == 2 * perturb_count + perturbed_idx[perturb_count] + 1:
                        print(f"Step {step} | pred action: {np.round(act_result.action, 3)} --> (corrected waypoint {keypoints[step - 2 * perturb_count - 1]} = dense waypoint idx {corrective_idx[perturb_count]})")
                    else:
                        print(f"Step {step} | pred action: {np.round(act_result.action, 3)} --> (waypoint {keypoints[step - 2 * perturb_count]})")
                else:
                    print(f"Step {step} | pred action: {np.round(act_result.action, 3)} --> (waypoint {keypoints[step - 2 * perturb_count]})")
                    

            # 3-2. take interactive actions
            if interactive:
                if not user_has_control:
                    take_control = input("T or F: ")  # early failure prediction here
                    if take_control == "T":
                        user_has_control = True

                if user_has_control:
                    # option 1: just generate a new waypoint
                    if False:
                        user_action_str = input("Enter proposed action:")
                        user_action = np.array([float(item) for item in user_action_str.split(',')])
                        act_result.action[0:3] = user_action[0:3]
                    # option 2: add delta action to current waypoint
                    if True:
                        act_result.action = prev_act
                        key = input('Move:')
                        key_count = Counter(key)

                        # in the front view
                        # - x range = [-0.075, 0.575] m
                        # - y range = [-0.455, 0.455] m
                        # - z range = > 0.752 m
                        # - right = decrease y
                        # - left  = increase y
                        # - up    = increase z
                        # - down  = decrease z
                        # - pull  = increase x
                        # - push  = decrease x

                        if key_count['w']:
                            act_result.action[0] -= 0.01 * key_count['w']
                        if key_count['s']:
                            act_result.action[0] += 0.01 * key_count['s']
                        if key_count['d']:
                            act_result.action[1] -= 0.01 * key_count['d']
                        if key_count['a']:
                            act_result.action[1] += 0.01 * key_count['a']
                        if key_count['q']:
                            act_result.action[2] += 0.01 * key_count['q']
                        if key_count['e']:
                            act_result.action[2] -= 0.01 * key_count['e']
                        if key == 'g':
                            # idx = 7: grasping. if == 0, close. if == 1, open
                            if act_result.action[7]:
                                act_result.action[7] = 0
                            else:
                                act_result.action[7] = 1

                        if key == 'r':  # 'r' key to release control
                            user_has_control = False

                        prev_act = act_result.action
                else:
                    prev_act = act_result.action
                print(np.round(act_result.action, 3))



            agent_obs_elems = {k: np.array(v) for k, v in act_result.observation_elements.items()}
            extra_replay_elements = {k: np.array(v) for k, v in act_result.replay_elements.items()}



            # 4. step through environment; 
            transition = env.step(act_result)
            # new obs?
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
            # TODO_JJL print what's inside obs_and_replay_elems

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
                    if agent is not None:
                        act_result = agent.act(step_signal.value, prepped_data, deterministic=eval)
                        agent_obs_elems_tp1 = {k: np.array(v) for k, v in act_result.observation_elements.items()}
                        obs_tp1.update(agent_obs_elems_tp1)
                replay_transition.final_observation = obs_tp1

            if record_enabled and transition.terminal or timeout or step == episode_length - 1:
                env.env._action_mode.arm_action_mode.record_end(env.env._scene, steps=60, step_scene=True)

            # 4. new obs
            # TODO_JJL
            obs = dict(transition.observation)

            yield replay_transition

            # TODO_JJL how to deal with termination or too early termination
            if transition.info.get("needs_reset", transition.terminal):
                step += 1
                for cam in CAMERAS:
                    rgb = obs[f'{cam}_rgb'] # (3, IMAGE_SIZE, IMAGE_SIZE)
                    rgb = Image.fromarray(rgb.T).rotate(-90)
                    if not replay_ground_truth:
                        rgb.save(os.path.join(multiview_img_folder, f"{cam}", f"{step}.png"))
                    else:
                        if step == 0:
                            img_name = 0
                        else:
                            if perturb_count <= len(perturbed_idx) - 1:
                                if step == 2 * perturb_count + perturbed_idx[perturb_count] + 1:
                                    img_name = f"{keypoints[step - 2 * perturb_count - 1]}_1_perturbed"
                                elif step == 2 * perturb_count + perturbed_idx[perturb_count] + 2:
                                    img_name = f"{keypoints[step - 2 * perturb_count - 2]}_2_corrected"
                                    perturb_count += 1
                                else:
                                    img_name = f"{keypoints[step - 2 * perturb_count - 1]}"
                            else:
                                img_name = f"{keypoints[step - 2 * perturb_count - 1]}"
                        rgb.save(os.path.join(multiview_img_folder, f"{cam}", f"{img_name}.png"))
                    if interactive:
                        rgb.save(os.path.join(multiview_img_folder, f"{cam}", "current.png"))
                return

            if not replay_ground_truth and step == episode_length:
                return
            if replay_ground_truth and step == len(actions):
                return
            
            step += 1