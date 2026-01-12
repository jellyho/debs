import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.meanflow_utils import adaptive_l2_loss, sample_t_r, sample_latent_dist
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorMeanFlowField
from utils.dit import MFDiT_REAL

class MEANFLOWAgent(flax.struct.PyTreeNode):
    """Don't extract but select! with action chunking. 
    """
    rng: Any
    network: Any
    config: Any = nonpytree_field()
        
    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))  # fold in horizon_length together with action_dim
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC mean flow loss.
        x = batch_actions
        t, r = sample_t_r(batch_size, t_rng, self.config['flow_ratio'])

        ##### It does not need to be normal distirbution
        e = sample_latent_dist(x_rng, (batch_size, action_dim), self.config['latent_dist'])
        z = (1 - t) * x + t * e
        v = e - x

        def mean_flow_forward(z, t, r):
            # Network 입력 순서에 맞춰서 호출 (Obs, Z, T, R)
            return self.network.select('actor_bc_flow')(
                batch['observations'], 
                z, 
                t, 
                t - r, # This seems to work better
                params=grad_params
            )
        
        def mean_flow_forward_no_r(z, t):
            # Network 입력 순서에 맞춰서 호출 (Obs, Z, T, R)
            return self.network.select('actor_bc_flow')(
                batch['observations'], 
                z, 
                t, 
                t,
                params=grad_params
            )

        ############################################
        if self.config['mf_method'] == 'jit_mf':
            x_pred, dxdt = jax.jvp(
                mean_flow_forward, 
                (z, t, r), 
                (v, jnp.ones_like(t), jnp.zeros_like(r))
            )
            u, dudt = z - x_pred, v - dxdt
            u_tgt = v - jnp.clip(t - r, a_min=0.0, a_max=1.0) * dudt
            u_tgt = jax.lax.stop_gradient(u_tgt)
            err = u - u_tgt
            loss = adaptive_l2_loss(err)
        ############################################

        elif self.config['mf_method'] == 'jit_mf_nor':
            r = jnp.zeros_like(t)
            x_pred, dxdt = jax.jvp(
                mean_flow_forward_no_r, 
                (z, t), 
                (v, jnp.ones_like(t))
            )
            u, dudt = z - x_pred, v - dxdt
            u_tgt = v - jnp.clip(t - r, a_min=0.0, a_max=1.0) * dudt
            u_tgt = jax.lax.stop_gradient(u_tgt)
            err = u - u_tgt
            loss = adaptive_l2_loss(err)

        ############################################
        elif self.config['mf_method'] == 'mfql':
            g_pred, dgdt = jax.jvp(
                mean_flow_forward, 
                (z, t, r), 
                (v, jnp.ones_like(t), jnp.zeros_like(r))
            )
            g_tgt = z + (t - r - 1) * v - (t - r) * dgdt
            g_tgt = jax.lax.stop_gradient(g_tgt)
            err = g_pred - g_tgt
            loss = adaptive_l2_loss(err)

        elif self.config['mf_method'] == 'mfql_nor':
            r = jnp.zeros_like(t)
            g_pred, dgdt = jax.jvp(
                mean_flow_forward_no_r, 
                (z, t), 
                (v, jnp.ones_like(t))
            )
            g_tgt = z + (t - r - 1) * v - (t - r) * dgdt
            g_tgt = jax.lax.stop_gradient(g_tgt)
            err = g_pred - g_tgt
            loss = adaptive_l2_loss(err)

        ############################################
        elif self.config['mf_method'] == 'ximf':
            x_pred = mean_flow_forward(z, t, r)
            
            # 2. JVP: Calculate time derivative of x_pred
            # Tangents: z -> v (flow direction), t -> 1, r -> 0

            x_pred, dxdt = jax.jvp(
                mean_flow_forward, 
                (z, t, r), 
                (v, jnp.ones_like(t), jnp.zeros_like(r))
            )

            X_est = x_pred + (t - r) * jax.lax.stop_gradient(dxdt)
            X_tgt = x
            
            # 4. Loss Calculation
            err = X_est - X_tgt
            loss = adaptive_l2_loss(err)
        elif self.config['mf_method'] == 'imf':
            v_pred = mean_flow_forward(z, t, t)

            u_pred, dudt = jax.jvp(
                mean_flow_forward, 
                (z, t, r), 
                (v_pred, jnp.ones_like(t), jnp.zeros_like(r))
            )

            X_est = u_pred + (t - r) * jax.lax.stop_gradient(dudt)
            X_tgt = e - x

            # 4. Loss Calculation
            err = X_est - X_tgt
            loss = adaptive_l2_loss(err)

        elif self.config['mf_method'] == 'imf_nor':
            r = jnp.zeros_like(t)
            v_pred = mean_flow_forward(z, t, t)

            u_pred, dudt = jax.jvp(
                mean_flow_forward_no_r, 
                (z, t), 
                (v_pred, jnp.ones_like(t))
            )

            X_est = u_pred + (t - r) * jax.lax.stop_gradient(dudt)
            X_tgt = e - x

            # 4. Loss Calculation
            err = X_est - X_tgt
            loss = adaptive_l2_loss(err)
        #########################################

        return loss, {
            'actor_loss': loss,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng = jax.random.split(rng, 2)

        loss = 0
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        loss += actor_loss
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        return loss, info

    @staticmethod
    def _update(agent, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)
    
    @jax.jit
    def batch_update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        # update_size = batch["observations"].shape[0]
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def sample_values(
        self,
        observations,
        rng=None,
    ):
        return jnp.zeros_like(observations)
    
    @jax.jit
    def sample_actions(
        self,
        observations,
        rng=None,
    ):
        """
        Sample actions. 
        Note: CFG logic and step size are handled inside 'compute_flow_actions' using self.config.
        """
        rng, x_rng = jax.random.split(rng, 2)
        latent_dim = self.config["horizon_length"] * self.config["action_dim"]
        e = sample_latent_dist(x_rng, (*observations.shape[: -len(self.config['ob_dims'])], latent_dim), self.config['latent_dist'])
        actions = self.compute_flow_actions(observations, e)
        actions = jnp.reshape(
            actions, 
            (*observations.shape[: -len(self.config['ob_dims'])],  # batch_size
            self.config["horizon_length"], self.config["action_dim"])
        )
        # return e
        return actions
    
    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def compute_flow_actions(
        self,
        observation,
        noise,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observation = self.network.select('actor_bc_flow_encoder')(observation)

        t = jnp.ones((*observation.shape[:-1], 1))
        r = jnp.zeros((*observation.shape[:-1], 1))

        output = self.network.select('actor_bc_flow')(observation, noise, t, t - r)
        if self.config['mf_method'] == 'imf':
            action = noise - jnp.clip(output, -1, 1)
        else:
            action = jnp.clip(output, -1, 1)
        return action

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_observations[..., :1]
        ob_dims = ex_observations.shape[-1:]
        action_dim = ex_actions.shape[-1]
        action_len = ex_actions.shape[1]
        
        full_actions = jnp.reshape(
            ex_actions,
            (ex_actions.shape[0], -1)
        )
        
        full_action_dim = full_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['actor_bc_flow'] = encoder_module()

        if config['use_DiT']:
            actor_bc_flow_def = MFDiT_REAL(
                hidden_dim=256,
                depth=3,
                num_heads=2,
                output_dim=action_dim,  
                output_len=action_len,
                use_r=True,
                encoders=None,
            )
        else:
            actor_bc_flow_def = ActorMeanFlowField(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=full_action_dim,
                layer_norm=config['actor_layer_norm'],
                encoder=encoders.get('actor_bc_flow'),
                use_fourier_features=config['use_fourier_features'],
                fourier_feature_dim=config['fourier_feature_dim']
            )

        actor_input_shape = (ex_observations, full_actions, ex_times, ex_times)

        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, actor_input_shape),
        )

        if encoders.get('actor_bc_flow') is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)

        if config["weight_decay"] > 0.:
            network_tx = optax.adamw(learning_rate=config['lr'], weight_decay=config["weight_decay"])
        else:
            network_tx = optax.adam(learning_rate=config['lr'])

        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))
    

    def get_param_count(self):
            """Calculate and return the number of parameters in the network."""
            params = self.network.params
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

    def print_param_stats(self):
        """Print network parameter statistics."""
        param_counts = self.get_param_count()
        
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


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='meanflow',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            critic_agg='min',  # Aggregation method for target Q values.
            num_critic=2, # critic ensemble size
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            horizon_length=ml_collections.config_dict.placeholder(int), # will be set
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.,
            flow_ratio=0.25,
            latent_dist='normal',
            use_DiT=False,
            mf_method='mf'
        )
    )
    return config
