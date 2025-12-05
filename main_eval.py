from utils.flax_utils import restore_agent
import glob, tqdm, wandb, os, json, random, time, jax
from absl import app, flags
from ml_collections import config_flags
from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger

from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets
from envs.robomimic_utils import is_robomimic_env

from utils.flax_utils import save_agent
from utils.datasets import Dataset, ReplayBuffer

from evaluation import evaluate
from agents import agents
import numpy as np
        
import json
import ml_collections

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-triple-play-singletask-task2-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 100000, 'Save interval.')
flags.DEFINE_integer('start_training', 5000, 'when does training start')

flags.DEFINE_integer('utd_ratio', 1, "update to data ratio")

flags.DEFINE_float('discount', 0.99, 'discount factor')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/debs.py', lock_config=False)

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval, used for large datasets because of memory constraints')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')

flags.DEFINE_integer('horizon_length', 5, 'action chunking length.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")

flags.DEFINE_bool('save_all_online_states', False, "save all trajectories to npy")
flags.DEFINE_string('checkpoint_path', '', 'Checkpoint path')
flags.DEFINE_integer('checkpoint_step', 1000000, 'Checkpoint step')

def load_config_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # 딕셔너리를 ConfigDict로 변환 (점 표기법 사용 가능 및 Type safety)
    config = ml_collections.ConfigDict(data)
    return config

# handle dataset
def process_train_dataset(ds):
    """
    Process the train dataset to 
        - handle dataset proportion
        - handle sparse reward
        - convert to action chunked dataset
    """

    ds = Dataset.create(**ds)

    return ds

def main(_):
    path = FLAGS.checkpoint_path
    step = FLAGS.checkpoint_step
    flag_config = load_config_from_json(f'{path}/flags.json')
    config = flag_config.agent

    # data loading
    if flag_config.ogbench_dataset_dir is not None:
        # custom ogbench dataset
        assert flag_config.dataset_replace_interval != 0
        assert flag_config.dataset_proportion == 1.0
        dataset_idx = 0
        dataset_paths = [
            file for file in sorted(glob.glob(f"{flag_config.ogbench_dataset_dir}/*.npz")) if '-val.npz' not in file
        ]
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            flag_config.env_name,
            dataset_path=dataset_paths[dataset_idx],
            compact_dataset=False,
        )
    else:
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(flag_config.env_name)

    # house keeping
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    train_dataset = process_train_dataset(train_dataset)
    example_batch = train_dataset.sample(())

    last_path = path.split('/')[-1]

    config["horizon_length"] = flag_config.horizon_length

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    run = setup_wandb(project='debs_eval', group=config['agent_name'], name=f'eval-{last_path}')

    cfg_sweep = [1.0, 3.0, 5.0, 10.0, 30.0]

    for i, cfg_value in enumerate(cfg_sweep):
        config['cfg'] = cfg_value
        agent = agent_class.create(
            FLAGS.seed,
            example_batch['observations'],
            example_batch['actions'],
            config,
        )
        agent = restore_agent(agent, path, step)

        eval_info, _, video = evaluate(
            agent=agent,
            env=eval_env,
            action_dim=example_batch["actions"].shape[-1],
            num_eval_episodes=50,
            num_video_episodes=10,
            video_frame_skip=3,
        )
        log_step = cfg_value
        wandb.log({f'{k}': v for k, v in eval_info.items()}, step=int(log_step))
        # wandb.log(eval_info, log_step, "eval")
        wandb.log({
            f"eval_video": wandb.Video(np.vstack(video).transpose(0, 3, 1, 2), fps=20, format="mp4")
        }, step=int(log_step))

if __name__ == '__main__':
    app.run(main)