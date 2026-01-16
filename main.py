import glob, tqdm, wandb, os, json, random, time, jax
from absl import app, flags
from ml_collections import config_flags
from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger

from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets
from envs.robomimic_utils import is_robomimic_env

from utils.flax_utils import save_agent, save_example_batch
from utils.datasets import Dataset, ReplayBuffer

from agents import agents
import numpy as np

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_string('project', 'MFQ', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('task_config', 'NO', 'suite:task_name:alpha:task_num')
flags.DEFINE_string('task_name', 'cube-triple-play', 'Task Name')
flags.DEFINE_integer('task_num', 0, 'Task Num')
flags.DEFINE_string('env_name', 'cube-triple-play-singletask-task2-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 10, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 100000, 'Save interval.')
flags.DEFINE_integer('start_training', 5000, 'when does training start')

flags.DEFINE_integer('utd_ratio', 1, "update to data ratio")

flags.DEFINE_float('discount', 0.99, 'discount factor')
flags.DEFINE_float('p_aug', 0.5, 'aug prob')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/debs.py', lock_config=False)

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval, used for large datasets because of memory constraints')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')

flags.DEFINE_string('droid_dataset_dir', None, 'DROID dataset directory')
flags.DEFINE_bool('droid_use_failure', False, 'Use failure DROID dataset or not')

flags.DEFINE_integer('horizon_length', 5, 'action chunking length.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")
flags.DEFINE_bool('save_all_online_states', False, "save all trajectories to npy")
flags.DEFINE_bool('record_time', False, "time_rocording")

class LoggingHelper:
    def __init__(self, wandb_logger):
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()

    def iterate(self, key, value):
        if 'hist' in key:
            return wandb.Histogram(value)
        else:
            return value

    def log(self, data, step, prefix=None,):
        if prefix is None:
            self.wandb_logger.log({f'{k}': self.iterate(k, v) for k, v in data.items()}, step=step)
        else:
            self.wandb_logger.log({f'{prefix}/{k}': self.iterate(k, v) for k, v in data.items()}, step=step)

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
    if FLAGS.task_config != 'NO':
        suite, task_name, alpha, task_num = FLAGS.task_config.split(':')
        FLAGS.task_name = str(task_name)
        FLAGS.agent.alpha = float(alpha)
        FLAGS.task_num = int(task_num)
        if suite == 'OG':
            FLAGS.env_name = f"{task_name}-singletask-task{task_num}-v0"

    exp_name = get_exp_name(FLAGS.seed)
    run = setup_wandb(project=FLAGS.project, group=FLAGS.run_group, name=exp_name)
    run.tags = run.tags + (FLAGS.env_name,)
    
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()

    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    config = FLAGS.agent
    config.training_steps=FLAGS.offline_steps
    
    # data loading
    if FLAGS.ogbench_dataset_dir is not None:
        # custom ogbench dataset
        assert FLAGS.dataset_replace_interval != 0
        assert FLAGS.dataset_proportion == 1.0
        dataset_idx = 0
        dataset_paths = [
            file for file in sorted(glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz")) if '-val.npz' not in file
        ]
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_paths[dataset_idx],
            compact_dataset=False,
        )
    else:
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
            FLAGS.env_name, 
            droid_dir=FLAGS.droid_dataset_dir,
            droid_use_failure=FLAGS.droid_use_failure
        )

    # house keeping
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    log_step = 0
    
    discount = FLAGS.discount
    config["horizon_length"] = FLAGS.horizon_length

    # handle dataset
    def process_train_dataset(dataset):
        """
        Process the train dataset to 
            - handle dataset proportion
            - handle sparse reward
            - convert to action chunked dataset
        """

        dataset = Dataset.create(**dataset)
        if FLAGS.dataset_proportion < 1.0:
            new_size = int(len(dataset['masks']) * FLAGS.dataset_proportion)
            dataset = Dataset.create(
                **{k: v[:new_size] for k, v in dataset.items()}
            )
        
        if is_robomimic_env(FLAGS.env_name):
            penalty_rewards = dataset["rewards"] - 1.0
            ds_dict = {k: v for k, v in dataset.items()}
            ds_dict["rewards"] = penalty_rewards
            dataset = Dataset.create(**ds_dict)
        
        if "puzzle-3x3" in FLAGS.task_name or "scene" in FLAGS.task_name:
            # Create a new dataset with modified rewards instead of trying to modify the frozen one
            sparse_rewards = (dataset["rewards"] != 0.0) * -1.0
            ds_dict = {k: v for k, v in dataset.items()}
            ds_dict["rewards"] = sparse_rewards
            dataset = Dataset.create(**ds_dict)

        dataset.actor_action_sequence = ( FLAGS.horizon_length )
        dataset.critic_action_sequence = ( FLAGS.horizon_length )
        dataset.nstep = 1 # Actually N step
        dataset.discount = FLAGS.discount
        dataset.discount2 = FLAGS.discount
        dataset.p_aug = FLAGS.p_aug
        return dataset
    
    train_dataset = process_train_dataset(train_dataset)
    example_batch = train_dataset.sample(config['batch_size'])

    save_example_batch(example_batch, FLAGS.save_dir)

    def print_batch_shapes(batch, prefix=""):
        for k, v in batch.items():
            try:
                print(f"{prefix}{k}: {v.shape}")
            except (AttributeError, TypeError):
                if isinstance(v, dict):
                    print_batch_shapes(v, prefix=f"{prefix}{k}.")
                else:
                    pass

    print_batch_shapes(example_batch)

    is_droid = True if FLAGS.droid_dataset_dir is not None else False

    agent_class = agents[config['agent_name']]

    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    print_param_stats(agent)

    # Setup logging.
    prefixes = ["eval", "env"]
    if FLAGS.offline_steps > 0:
        prefixes.append("offline_agent")

    logger = LoggingHelper(
        wandb_logger=wandb,
    )

    # Offline RL
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1)):
        log_step += 1
        batch = train_dataset.sample(config['batch_size'])
        agent, info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            logger.log(info, step=log_step)
        
        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.eval_interval == 0:
            if not is_droid and (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
                # during eval, the action chunk is executed fully
                if "bandit" in FLAGS.env_name:
                    from envs.bandit_utils import evaluate
                    eval_info, _, _ = evaluate(
                        agent=agent,
                        env=eval_env,
                        action_dim=example_batch["actions"].shape[-1],
                        num_eval_episodes=FLAGS.eval_episodes,
                        num_video_episodes=FLAGS.video_episodes,
                        video_frame_skip=FLAGS.video_frame_skip,
                    )
                    logger.log(eval_info, log_step, "eval")
                else:
                    from evaluation import evaluate
                    eval_info, _, video = evaluate(
                        agent=agent,
                        env=eval_env,
                        action_dim=example_batch["actions"].shape[-1],
                        num_eval_episodes=FLAGS.eval_episodes,
                        num_video_episodes=FLAGS.video_episodes,
                        video_frame_skip=FLAGS.video_frame_skip,
                    )
                    logger.log(eval_info, log_step, "eval")
                    if len(video) > 0:
                        wandb.log({
                            f"eval_video": wandb.Video(np.vstack(video).transpose(0, 3, 1, 2), fps=20, format="mp4")
                        }, step=log_step)
            save_agent(agent, FLAGS.save_dir, log_step)

if __name__ == '__main__':
    app.run(main)
