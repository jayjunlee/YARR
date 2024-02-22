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

from rvt.utils.peract_utils import CAMERAS

import os
import csv
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
                  ):
        # 1. reset
        # 2. initial obs
        if eval:
            obs = env.reset_to_demo(eval_demo_seed) # this is using the same initial config of a test data
            # get ground-truth action sequence
            if replay_ground_truth:
                actions = env.get_ground_truth_action(eval_demo_seed)
        else:
            obs = env.reset()
        agent.reset()
        # initial obs?
        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
        count = 0
        prev_act = np.zeros((9,))
        for step in range(episode_length):
            # 3. take action
            prepped_data = {k:torch.tensor(np.array([v]), device=self._env_device) for k, v in obs_history.items()}
            if not replay_ground_truth:
                act_result = agent.act(step_signal.value, prepped_data, deterministic=eval)
            else:
                if step >= len(actions):
                    return
                act_result = ActResult(actions[step])

            multiview_img_folder = os.path.join(log_dir, task_name, str(episode_number), "multiview")
            for cam in CAMERAS:
                rgb = obs[f'{cam}_rgb'] # (3, IMAGE_SIZE, IMAGE_SIZE)
                rgb = Image.fromarray(rgb.T).rotate(-90)
                rgb.save(os.path.join(multiview_img_folder, f"{cam}", f"{0}.png"))

            # TODO_JJL but what is ActResult?
            # here before we step through the base policy's action,
            # we prompt GPT-4V using images maybe from "obs" variable
            # e.g.
            # if not failure:
            #   transition = env.step(act_result)
            # else:
            #   transition = env.step(corrected_act_GPT4V) in the form of ActResult

            # 0.25 +/- 0.65/2, 0 +/- 0.91/2

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

            print("Predicted action:", np.round(act_result.action, 3))
            take_control = input("T or F: ")
            
            if take_control == "T":
                # option 1: just generate a new waypoint
                if False:
                    user_action_str = input("Enter proposed action:")
                    user_action = np.array([float(item) for item in user_action_str.split(',')])
                    act_result.action[0:3] = user_action[0:3]
                # option 2: add delta action to current waypoint
                if True:
                    act_result.action = prev_act
                    key = input('Move:')
                    if key == 'w':
                        act_result.action[0] -= 0.01
                    if key == 's':
                        act_result.action[0] += 0.01
                    if key == 'd':
                        act_result.action[1] -= 0.01
                    if key == 'dd':
                        act_result.action[1] -= 0.1
                    if key == 'a':
                        act_result.action[1] += 0.01
                    if key == 'aa':
                        act_result.action[1] += 0.1
                    if key == 'q':
                        act_result.action[2] += 0.01
                    if key == 'e':
                        act_result.action[2] -= 0.01
                    prev_act = act_result.action
            else:
                prev_act = act_result.action
            print(act_result.action)


            # Convert to np if not already
            agent_obs_elems = {k: np.array(v) for k, v in
                               act_result.observation_elements.items()}
            extra_replay_elements = {k: np.array(v) for k, v in
                                     act_result.replay_elements.items()}

            # PROBING VARIABLES
            # logging.debug(f"Observation variable: {obs}")
            # logging.debug(f"Observation keys: {obs.keys()}")



            # (9,) = (7,) EE pose + (1,) Gripper if zero close if 1 open + (1,) Ignore collision
            # action = act_result.action
            # if step == 0:
            #     open(os.path.join(multiview_img_folder, "actions.csv"), 'w').close() # delete all content first
            #     open(os.path.join(multiview_img_folder, "lang.csv"), 'w').close()
            #     with open(os.path.join(multiview_img_folder, "lang.csv"), 'a', newline='') as csvfile:
            #         csvfile.write(env._lang_goal)

            # with open(os.path.join(multiview_img_folder, "actions.csv"), 'a', newline='') as csvfile:
            #     writer = csv.writer(csvfile)
            #     writer.writerow(np.round(action,4))
            # logging.debug(f"action at step {step}: {action}")

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
                if count == 2:
                    return
                count += 1
