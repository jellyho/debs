import glob, tqdm, wandb, os, json, random, time, jax
from absl import app, flags
from ml_collections import config_flags
from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger

from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets
from envs.robomimic_utils import is_robomimic_env

from utils.flax_utils import save_agent
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
flags.DEFINE_string('task_name', 'cube-triple-play', 'Task Name')
flags.DEFINE_integer('task_num', 0, 'Task Num')
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

def main(_):
    exp_name = get_exp_name(FLAGS.seed)
    run = setup_wandb(project=FLAGS.project, group=FLAGS.run_group, name=exp_name)
    run.tags = run.tags + (FLAGS.env_name,)
    if 'meanflow' in FLAGS.agent.agent_name:
        run.tags = run.tags + (FLAGS.agent.mf_method,)
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()

    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    config = FLAGS.agent
    
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
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name)

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
            ds = Dataset.create(**ds_dict)
        
        if "puzzle" in FLAGS.task_name or "scene" in FLAGS.task_name:
            # Create a new dataset with modified rewards instead of trying to modify the frozen one
            sparse_rewards = (dataset["rewards"] != 0.0) * -1.0
            ds_dict = {k: v for k, v in dataset.items()}
            ds_dict["rewards"] = sparse_rewards
            dataset = Dataset.create(**ds_dict)

        dataset.actor_action_sequence = ( FLAGS.horizon_length )
        dataset.critic_action_sequence = ( FLAGS.horizon_length )
        dataset.nstep = 1
        dataset.discount = FLAGS.discount
        dataset.discount2 = FLAGS.discount
        # dataset.discount2 = config['discount']
        # dataset.normalize_rewards = FLAGS.normalize_rewards
        # dataset.additional_normalize_rewards = FLAGS.additional_normalize_rewards
        # dataset.additional_normalize_rewards_scale = FLAGS.additional_normalize_rewards_scale
        # if config['agent_name'] == 'rebrac' or config['agent_name'] == 'sarsa_fql':
        #     dataset.return_next_actions = True
        if 'bandit' in FLAGS.env_name:
            dataset.v_max = 1
            dataset.v_min = 0
        elif 'singletask' in FLAGS.env_name:
            dataset.v_min = 1 / (1 - dataset.discount) * dataset['rewards'].min().item()
            dataset.v_max = 0
        return dataset
    
    train_dataset = process_train_dataset(train_dataset)
    example_batch = train_dataset.sample(config['batch_size'])

    config['v_min'] = train_dataset.v_min
    config['v_max'] = train_dataset.v_max

    for k, v in example_batch.items():
        try:
            print(f"{k}: {v.shape}")
        except:
            pass

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Setup logging.
    prefixes = ["eval", "env"]
    if FLAGS.offline_steps > 0:
        prefixes.append("offline_agent")

    logger = LoggingHelper(
        wandb_logger=wandb,
    )

    hlmean_flat = 'meanflow' in config['agent_name']
    hlmean_late = 'meanflow' in config['agent_name'] and config['late_update']
    hlmean_curr = 'meanflow' in config['agent_name'] and not config['late_update']

    # Offline RL
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1)):
        log_step += 1
        batch = train_dataset.sample(config['batch_size'])

        if config['agent_name'] in ['hldebs', 'debs', 'cfgrl', 'hlcfgrl']:
            agent, info = agent.critic_update(batch)
        else:
            agent, info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            logger.log(info, step=log_step)
        
        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)

        if hlmean_curr:
            if i == FLAGS.offline_steps - 1 or \
                (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
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
                    wandb.log({
                        f"eval_video": wandb.Video(np.vstack(video).transpose(0, 3, 1, 2), fps=20, format="mp4")
                    }, step=log_step)

    if not hlmean_flat or hlmean_late:
        for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1)):
            log_step += 1
            # batch = train_dataset.sample_sequence(config['batch_size'], sequence_length=FLAGS.horizon_length, discount=discount)
            batch = train_dataset.sample(config['batch_size'])

            info = None
            if config['agent_name'] in ['hldebs', 'debs', 'cfgrl', 'hlcfgrl']:
                agent, info = agent.actor_update(batch)
            elif config['agent_name'] in ['resf', 'addf']:
                agent, info = agent.residual_actor_update(batch)
            elif 'meanflow' in config['agent_name']:
                agent, info = agent.latent_actor_update(batch)

            if info is not None:
                if i % FLAGS.log_interval == 0:
                    logger.log(info, step=log_step)
            
            # saving
            if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
                save_agent(agent, FLAGS.save_dir, log_step)

            # eval
            if i == FLAGS.offline_steps - 1 or \
                (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
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
                    wandb.log({
                        f"eval_video": wandb.Video(np.vstack(video).transpose(0, 3, 1, 2), fps=20, format="mp4")
                    }, step=log_step)

if __name__ == '__main__':
    app.run(main)
