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

import os
import shutil
import csv
from collections import Counter
from PIL import Image
import logging
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG)

class RolloutGenerator(object):

    def __init__(self, env_device = 'cuda:0'):
        self._env_device = env_device

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def generator(self, step_signal: Value, env: Env, agent: Agent,
                  episode_length: int, timesteps: int,
                  eval: bool, log_dir, task_name, episode_number, eval_demo_seed: int = 0,
                  record_enabled: bool = False,
                  replay_ground_truth: bool = False,
                  interactive: bool = False
                  ):
        # 1. reset
        # 2. initial obs
        if eval:
            obs = env.reset_to_demo(eval_demo_seed) # this is using the same initial config of a test data
            # get ground-truth action sequence
            if replay_ground_truth:
                actions, keypoints, dense_actions, waypoints = env.get_ground_truth_action(eval_demo_seed, 'heuristic') # 'dense' / 'heuristic' / 'awe'
                print(np.round(actions,3))
        else:
            obs = env.reset()

        if agent is not None:
            agent.reset()

        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
        count = 0
        prev_act = np.zeros((9,))
        user_has_control = False

        # perturbation injection
        # waypoints: range(0,whatever)
        # dense_actions: shape (186,9)
        # keypoints: [56, 79, 101, 138, 147, 185]
        # actions: 
        # array([ 1.17711231e-01,  1.71137631e-01,  9.28920507e-01, -2.97863096e-01, 9.54608440e-01,  4.98470072e-05, -4.77327849e-04,  1.00000000e+00,        0.00000000e+00])
        # array([ 1.17820233e-01,  1.71247438e-01,  7.58766115e-01, -2.98061818e-01, 9.54546452e-01, -4.02269507e-05, -4.55035712e-04,  0.00000000e+00,        1.00000000e+00])
        # array([ 1.17517643e-01,  1.71243042e-01,  9.27575827e-01, -2.98106015e-01,        9.54532564e-01, -3.12376324e-05, -6.38707366e-04,  0.00000000e+00,        1.00000000e+00])
        # array([ 3.06168437e-01, -2.57705301e-01,  9.27317142e-01, -3.00539017e-01,        9.53768909e-01,  7.41665310e-04, -7.54668145e-04,  0.00000000e+00,        0.00000000e+00])
        # array([ 3.06513101e-01, -2.58281738e-01,  8.67196202e-01, -3.00613463e-01,        9.53745604e-01,  5.94818091e-04, -7.41272117e-04,  0.00000000e+00,        1.00000000e+00])
        # array([ 3.06698292e-01, -2.58368641e-01,  8.67090642e-01,  4.57508296e-01,        8.89205217e-01, -1.29942171e-04, -5.57533582e-04,  1.00000000e+00,        1.00000000e+00])
        
        # the perturbed idx is where failure is injected. we fallback to a previous waypoint and reapproach the same waypoint correctly
        perturbed_idx = [1, 4] # perurb 79 = keypoints[peturbed_idx[0]]
        p_count = 0 
        for idx in perturbed_idx:
            # inject perturbation to expert keypoint to induce failure
            perturbed_action = actions[idx,:].copy()
            perturbation = np.random.normal(0, 0.1, size=(2,))
            perturbed_action[0:2] += perturbation

            # randomly sample a correct waypoint from a segment of waypoints leading up to the expert keypoint
            waypoint_segment = waypoints[keypoints[idx-1] : keypoints[idx]]
            corrective_action = dense_actions[np.random.choice(waypoint_segment)]

            # insert perturbed action and correction action to actions
            actions = np.insert(actions, p_count * 2 + idx, np.vstack([perturbed_action, corrective_action]), axis=0)
            p_count += 1
        print(np.round(actions,3))

        # 3. stepping
        step = 0
        perturb_count = 0
        while True:

            multiview_img_folder = os.path.join(log_dir, task_name, str(episode_number), "multiview")
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
                # if interactive:
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
                if step == 0:
                    print(f"Step {step} | pred action: {np.round(act_result.action, 3)} --> (waypoint {keypoints[step]})")
                elif step == 2 * perturb_count + perturbed_idx[perturb_count]:
                    print(f"Step {step} | pred action: {np.round(act_result.action, 3)} --> (perturbed waypoint {keypoints[step - 2 * perturb_count]})")
                elif step == 2 * perturb_count + perturbed_idx[perturb_count] + 1:
                    print(f"Step {step} | pred action: {np.round(act_result.action, 3)} --> (corrected waypoint {keypoints[step - 2 * perturb_count - 1]})")
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
                print("terminal~~")
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
                return

            if not replay_ground_truth and step == episode_length:
                return
            if replay_ground_truth and step == len(actions):
                return
            
            step += 1