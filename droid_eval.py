from utils.flax_utils import restore_agent
import glob, tqdm, wandb, os, json, random, time, jax
from absl import app, flags
from ml_collections import config_flags
from utils.flax_utils import save_agent, load_example_batch
from agents import agents
import numpy as np
from droid_eval_utils.policy_wrapper import ReplayWrapper, JAXWrapper
from droid.controllers.oculus_controller import VRPolicy
from droid.robot_env import RobotEnv
from droid.user_interface.data_collector import DataCollecter
from droid.user_interface.gui import RobotGUI  
import json
import ml_collections


if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

def load_config_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # 딕셔너리를 ConfigDict로 변환 (점 표기법 사용 가능 및 Type safety)
    config = ml_collections.ConfigDict(data)
    return config

FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_path', '', 'Checkpoint path')
flags.DEFINE_integer('checkpoint_step', 1000000, 'Checkpoint step')
flags.DEFINE_integer('seed', 100, 'seed')

def get_param_count(agent):
    """Calculate and return the number of parameters in the network."""
    params = agent.network.params
    if hasattr(params, 'unfreeze'):
        params = params.unfreeze()
    
    param_counts = {}
    
    # Calculate module-wise parameter counts
    for module_name, module_params in params.items():
        module_leaves = jax.tree_util.tree_leaves(module_params)
        param_counts[module_name] = sum(param.size for param in module_leaves)
    
    # Calculate total parameters
    all_leaves = jax.tree_util.tree_leaves(params)
    param_counts['total'] = sum(param.size for param in all_leaves)
    
    return param_counts

def print_param_stats(agent):
    """Print network parameter statistics."""
    param_counts = get_param_count(agent)
    
    print("Network Parameter Statistics:")
    print("-" * 50)
    
    # Print module-wise parameter counts
    for module_name, count in param_counts.items():
        if module_name != 'total':
            print(f"{module_name}: {count:,} parameters ({count * 4 / (1024**2):.2f} MB)")
    
    # Print total parameter count
    total = param_counts['total']
    print("-" * 50)
    print(f"Total parameters: {total:,} ({total * 4 / (1024**2):.2f} MB)")

def main(_):
    path = FLAGS.checkpoint_path
    step = FLAGS.checkpoint_step
    flag_config = load_config_from_json(f'{path}/flags.json')
    config = flag_config.agent
    config["horizon_length"] = flag_config.horizon_length
    ckpt_name = path.split("/")[-1]
    log_dir = "eval_logs/" + ckpt_name

    def print_batch_shapes(batch, prefix=""):
        for k, v in batch.items():
            try:
                print(f"{prefix}{k}: {v.shape}")
            except (AttributeError, TypeError):
                if isinstance(v, dict):
                    print_batch_shapes(v, prefix=f"{prefix}{k}.")
                else:
                    pass
    example_batch = load_example_batch(path)
    print_batch_shapes(example_batch)
    
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )
    agent = restore_agent(agent, path, step)
    
    # return True

    ## policy wrapper
    np.random.seed(FLAGS.seed)

    # policy = ReplayWrapper(
    #     '/home/rllab2/jellyho/Environment_Reset/droid/data/2026-01-15/pick_carrot/success/trajectory_9.h5'
    # )
    policy = JAXWrapper(agent)

    ## robo Env
    imsize = 224
    action_space = "cartesian_velocity"
    gripper_action_space = "position" # None #"position"

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

    camera_kwargs = dict(
        hand_camera=dict(image=True, concatenate_images=False, resolution=(imsize, imsize), resize_func="cv2"),
        varied_camera=dict(image=True, concatenate_images=False, resolution=(imsize, imsize), resize_func="cv2"),
    )
    
    policy_camera_kwargs = {}
    policy_camera_kwargs.update(camera_kwargs)

    import os
    os.system('nmcli connection up "Local Switch"')

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
        policy=policy,
        save_traj_dir=log_dir,
        save_data=True,
    )
    RobotGUI(robot=data_collector)


if __name__ == '__main__':
    import os
    import sys
    import atexit
    import signal

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
        app.run(main)
    finally:
        pass