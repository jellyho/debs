from utils.flax_utils import restore_agent
import glob, tqdm, wandb, os, json, random, time, jax
from absl import app, flags
from ml_collections import config_flags
from utils.flax_utils import save_agent, load_example_batch
from agents import agents
import numpy as np
        
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
    print_param_stats(agent)


    ## policy wrapper

    ## robo Env

    ## START!!


if __name__ == '__main__':
    app.run(main)