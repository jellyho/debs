import json
import os
import numpy as np
from policy_wrapper import ReplayWrapper
# import torch
# from collections import OrderedDict
# from copy import deepcopy

from droid.controllers.oculus_controller import VRPolicy
# from droid.evaluation.policy_wrapper import PolicyWrapperRobomimic
from droid.robot_env import RobotEnv
from droid.user_interface.data_collector import DataCollecter
from droid.user_interface.gui import RobotGUI

# import robomimic.utils.file_utils as FileUtils
# import robomimic.utils.torch_utils as TorchUtils
# import robomimic.utils.tensor_utils as TensorUtils

# import cv2

import os
import sys
import atexit
import signal

os.system('nmcli connection up "Local Switch"')

# --- CLEANUP SETUP START ---
def restore_internet():
    """Restores internet connection on exit."""
    print("\n[System] Restoring Internet connection...")
    os.system('nmcli connection up "Internet"')

# 1. Register for normal exits (finished script, errors, etc.)
atexit.register(restore_internet)

# 2. Register for 'kill' signals (e.g. closing terminal window, kill command)
def signal_handler(signum, frame):
    sys.exit(0) # This triggers atexit automatically

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

try:
    def eval_launcher(variant, run_id, exp_id):
        # Get Directory #
        dir_path = os.path.dirname(os.path.realpath(__file__))

        # Prepare Log Directory #
        variant["exp_name"] = os.path.join(variant["exp_name"], "run{0}/id{1}/".format(run_id, exp_id))
        log_dir = os.path.join(dir_path, "evaluation_logs", variant["exp_name"])

        # Set Random Seeds #
        # torch.manual_seed(variant["seed"])
        np.random.seed(variant["seed"])

        # Set Compute Mode #
        # use_gpu = variant.get("use_gpu", False)
        # torch.device("cuda:0" if use_gpu else "cpu")

        ckpt_path = variant["ckpt_path"]

        # device = TorchUtils.get_torch_device(try_to_use_cuda=True)
        # ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
        # config = json.loads(ckpt_dict["config"])
        imsize = 224

        # ckpt_dict["config"] = json.dumps(config)
        # policy, _ = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=True)

        # policy = None
        
        action_space = "cartesian_velocity"
        gripper_action_space = "position" # None #"position"

        # Prepare Policy Wrapper #
        data_processing_kwargs = dict(
            timestep_filtering_kwargs=dict(
                action_space=action_space,
                gripper_action_space=gripper_action_space,
                robot_state_keys=["cartesian_position", "gripper_position", "joint_positions"],
                # camera_extrinsics=[],
            ),
            image_transform_kwargs=dict(
                remove_alpha=True,
                bgr_to_rgb=True,
                to_tensor=False,
                augment=False,
            ),
        )
        timestep_filtering_kwargs = data_processing_kwargs.get("timestep_filtering_kwargs", {})
        image_transform_kwargs = data_processing_kwargs.get("image_transform_kwargs", {})

        policy_timestep_filtering_kwargs = {}
        policy_image_transform_kwargs = {}

        policy_timestep_filtering_kwargs.update(timestep_filtering_kwargs)
        policy_image_transform_kwargs.update(image_transform_kwargs)

        # fs = config["train"]["frame_stack"]

        # wrapped_policy = PolicyWrapperRobomimic(
        #     policy=policy,
        #     timestep_filtering_kwargs=policy_timestep_filtering_kwargs,
        #     image_transform_kwargs=policy_image_transform_kwargs,
        #     frame_stack=fs,
        #     eval_mode=True,
        # )

        wrapped_policy = ReplayWrapper(
            '/home/rllab2/jellyho/Environment_Reset/droid/data/2026-01-15/pick_carrot/success/trajectory_9.h5'
        )

        camera_kwargs = dict(
            hand_camera=dict(image=True, concatenate_images=False, resolution=(imsize, imsize), resize_func="cv2"),
            varied_camera=dict(image=True, concatenate_images=False, resolution=(imsize, imsize), resize_func="cv2"),
        )
        
        policy_camera_kwargs = {}
        policy_camera_kwargs.update(camera_kwargs)

        env = RobotEnv(
            action_space=policy_timestep_filtering_kwargs["action_space"],
            gripper_action_space=policy_timestep_filtering_kwargs["gripper_action_space"],
            camera_kwargs=policy_camera_kwargs
        )
        controller = VRPolicy()

        # Launch GUI #
        data_collector = DataCollecter(
            env=env,
            controller=controller,
            policy=wrapped_policy,
            save_traj_dir=log_dir,
            save_data=variant.get("save_data", True),
        )
        RobotGUI(robot=data_collector)


    variant = dict(
        exp_name="policy_test",
        save_data=True,
        use_gpu=True,
        seed=0,
        policy_logdir="test",
        task="",
        layout_id=None,
        model_id=50,
        camera_kwargs=dict(),
        data_processing_kwargs=dict(
            timestep_filtering_kwargs=dict(),
            image_transform_kwargs=dict(),
        ),
        ckpt_path='',
    )
        
    print("Evaluating Policy")
    eval_launcher(variant, run_id=1, exp_id=0)
finally:
    # This block ensures cleanup runs even if the code above crashes
    pass 
    # The actual restoration happens in restore_internet() via atexit