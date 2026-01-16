import functools
import glob
import os
import pickle
from typing import Any, Dict, Mapping, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import ml_collections

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)

def get_batch_shape(observations, leaf_ndims):
    """
    Get the batch shape from observations.
    Robustly handles Dictionary inputs even if keys don't perfectly match (subset).
    """
    # 1. Handle Dictionary types (Dict, FrozenDict, ConfigDict)
    if isinstance(observations, (dict, flax.core.FrozenDict, ml_collections.ConfigDict)):
        # Iterate through keys in observations to find the first match in leaf_ndims.
        for key, value in observations.items():
            # Check if leaf_ndims is also a dict and has the same key
            if isinstance(leaf_ndims, (dict, flax.core.FrozenDict, ml_collections.ConfigDict)) and key in leaf_ndims:
                ref_ndim = leaf_ndims[key]
                
                # If value is a nested dictionary, recurse.
                if isinstance(value, (dict, flax.core.FrozenDict, ml_collections.ConfigDict)):
                    return get_batch_shape(value, ref_ndim)
                
                # Compare dimensions (Return immediately if a match is found)
                if value.ndim == ref_ndim:
                    return () # Unbatched
                elif value.ndim == ref_ndim + 1:
                    return (value.shape[0],) # Batched (B,)
                else:
                    raise ValueError(f"Dim mismatch at key '{key}': got {value.ndim}, expected {ref_ndim} or {ref_ndim+1}")

    # 2. Fallback to standard Tree Flatten logic (for Arrays or perfectly matching Lists/Tuples)
    flat_obs, _ = jax.tree_util.tree_flatten(observations)
    flat_ref_ndims, _ = jax.tree_util.tree_flatten(leaf_ndims)
    
    if not flat_obs:
        return ()

    obs_leaf = flat_obs[0]
    
    # If leaf_ndims structure doesn't match or is empty, try to infer safely
    if flat_ref_ndims:
        ref_ndim = flat_ref_ndims[0]
    else:
        # If no reference info is available, assume the 0-th dimension is the batch dimension.
        return (obs_leaf.shape[0],)

    if obs_leaf.ndim == ref_ndim:
        return ()
    elif obs_leaf.ndim == ref_ndim + 1:
        return (obs_leaf.shape[0],)
    else:
        raise ValueError(f"Input dimension mismatch! Expected {ref_ndim} or {ref_ndim+1}, got {obs_leaf.ndim}")
    
def save_example_batch(batch, save_dir, filename="example_batch.pkl"):
    """
    JAX Array를 CPU Numpy로 변환하여 가볍게 저장합니다.
    batch: 학습 데이터로더에서 뽑은 첫 번째 배치의 복사본
    """
    # 1. GPU/TPU에 있는 JAX Array를 CPU Numpy Array로 변환 (Portability)
    # 딕셔너리 구조(tree)를 유지하면서 잎사귀(배열)만 변환합니다.
    batch_np = jax.tree_util.tree_map(lambda x: jax.device_get(x), batch)
    
    # 2. 저장
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    
    with open(path, 'wb') as f:
        pickle.dump(batch_np, f)
        
    print(f"[System] Example batch saved for inference: {path}")

def load_example_batch(load_dir, filename="example_batch.pkl"):
    path = os.path.join(load_dir, filename)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Example batch not found at {path}. Cannot init model!")
        
    with open(path, 'rb') as f:
        batch_np = pickle.load(f)
        
    print(f"[System] Example batch loaded from {path}")
    return batch_np

class ModuleDict(nn.Module):
    """A dictionary of modules.

    This allows sharing parameters between modules and provides a convenient way to access them.

    Attributes:
        modules: Dictionary of modules.
    """

    modules: Dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        """Forward pass.

        For initialization, call with `name=None` and provide the arguments for each module in `kwargs`.
        Otherwise, call with `name=<module_name>` and provide the arguments for that module.
        """
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    f'When `name` is not specified, kwargs must contain the arguments for each module. '
                    f'Got kwargs keys {kwargs.keys()} but module keys {self.modules.keys()}'
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out

        return self.modules[name](*args, **kwargs)


class TrainState(flax.struct.PyTreeNode):
    """Custom train state for models.

    Attributes:
        step: Counter to keep track of the training steps. It is incremented by 1 after each `apply_gradients` call.
        apply_fn: Apply function of the model.
        model_def: Model definition.
        params: Parameters of the model.
        tx: optax optimizer.
        opt_state: Optimizer state.
    """

    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    batch_stats: Any = nonpytree_field()
    opt_state: Any

    @classmethod
    def create(cls, model_def, params, tx=None, batch_stats=None, **kwargs):
        """Create a new train state."""
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        if batch_stats is None:
            batch_stats = flax.core.FrozenDict({})

        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            batch_stats=batch_stats,
            **kwargs,
        )

    def __call__(self, *args, params=None, method=None, batch_stats=None, **kwargs):
        """Forward pass.

        When `params` is not provided, it uses the stored parameters.

        The typical use case is to set `params` to `None` when you want to *stop* the gradients, and to pass the current
        traced parameters when you want to flow the gradients. In other words, the default behavior is to stop the
        gradients, and you need to explicitly provide the parameters to flow the gradients.

        Args:
            *args: Arguments to pass to the model.
            params: Parameters to use for the forward pass. If `None`, it uses the stored parameters, without flowing
                the gradients.
            method: Method to call in the model. If `None`, it uses the default `apply` method.
            **kwargs: Keyword arguments to pass to the model.
        """
        if params is None:
            params = self.params

        if batch_stats is None:
            batch_stats = self.batch_stats

        variables = {'params': params, 'batch_stats': batch_stats}

        if method is not None:
            method_name = getattr(self.model_def, method)
        else:
            method_name = None

        return self.apply_fn(variables, *args, method=method_name, mutable=False, **kwargs)

    def select(self, name):
        """Helper function to select a module from a `ModuleDict`."""
        return functools.partial(self, name=name)

    def apply_gradients(self, grads, new_batch_stats=None, **kwargs):
        """Apply the gradients and return the updated state."""
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        new_stats = new_batch_stats if new_batch_stats is not None else self.batch_stats

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            batch_stats=new_stats,
            **kwargs,
        )

    def apply_loss_fn(self, loss_fn):
        """Apply the loss function and return the updated state and info.

        It additionally computes the gradient statistics and adds them to the dictionary.
        """
        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)

        grad_max = jax.tree_util.tree_map(jnp.max, grads)
        grad_min = jax.tree_util.tree_map(jnp.min, grads)
        grad_norm = jax.tree_util.tree_map(jnp.linalg.norm, grads)

        grad_max_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_max)], axis=0)
        grad_min_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_min)], axis=0)
        grad_norm_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_norm)], axis=0)

        final_grad_max = jnp.max(grad_max_flat)
        final_grad_min = jnp.min(grad_min_flat)
        final_grad_norm = jnp.linalg.norm(grad_norm_flat, ord=1)

        info.update(
            {
                'grad/max': final_grad_max,
                'grad/min': final_grad_min,
                'grad/norm': final_grad_norm,
            }
        )

        return self.apply_gradients(grads=grads), info


def save_agent(agent, save_dir, epoch):
    """Save the agent to a file.

    Args:
        agent: Agent.
        save_dir: Directory to save the agent.
        epoch: Epoch number.
    """

    save_dict = dict(
        agent=flax.serialization.to_state_dict(agent),
    )
    save_path = os.path.join(save_dir, f'params_{epoch}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)

    print(f'Saved to {save_path}')


def restore_agent_with_file(agent, file_path):
    """Just like restore_agent() but expect file_path to include restore_epoch
    """
    assert os.path.exists(file_path), f'File {file_path} does not exist'
    with open(file_path, 'rb') as f:
        load_dict = pickle.load(f)

    agent = flax.serialization.from_state_dict(agent, load_dict['agent'])

    print(f'Restored from {file_path}')

    return agent

def restore_agent(agent, restore_path, restore_epoch):
    """Restore the agent from a file.

    Args:
        agent: Agent.
        restore_path: Path to the directory containing the saved agent.
        restore_epoch: Epoch number.
    """
    candidates = glob.glob(restore_path)

    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'

    restore_path = candidates[0] + f'/params_{restore_epoch}.pkl'

    with open(restore_path, 'rb') as f:
        load_dict = pickle.load(f)

    agent = flax.serialization.from_state_dict(agent, load_dict['agent'])

    print(f'Restored from {restore_path}')

    return agent
