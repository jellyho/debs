import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorMeanFlowField

class MEANFLOWAgent(flax.struct.PyTreeNode):
    """Don't extract but select! with action chunking. 
    """
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    ## MF utils
    def adaptive_l2_loss(self, batch_size, error, p=0.5, c=1e-3):
        """
        Adaptive L2 loss with Valid Masking.
        Args:
            error: (Batch, Total_Action_Dim) - Flattened error
            valid_mask: (Batch,)
        """
        ## THIS SHOULD BE modified so support horizion
        # 1. Sample별 Error 계산 (Sum of Squared Error)
        # Action Dim축으로 합침
        squared_error = jnp.mean(
            jnp.reshape(
                jnp.square(error), 
                (batch_size, self.config["horizon_length"], self.config["action_dim"]) 
            ),
            axis=(1, 2)
        )
        
        # 2. Adaptive Weight 계산 (Gradient 흐르지 않게 stop_gradient)
        # p = 1 - gamma (공식 코드 norm_p=1.0과 유사)
        w = 1.0 / (squared_error + c) ** p
        w = jax.lax.stop_gradient(w)
        
        # 3. Weighted Loss 계산
        # loss = w * ||u - u_tgt||^2
        loss = w * squared_error
        
        return loss.mean()

    def sample_t_r(self, batch_size, rng):
        # lognorm sampling (Seems working better than uniform)
        rng, t_rng, r_rng = jax.random.split(rng, 3)

        t = jax.nn.sigmoid(jax.random.normal(t_rng, [batch_size, 1]) - 0.4)
        r = jax.nn.sigmoid(jax.random.normal(r_rng, [batch_size, 1]) - 0.4)
        t, r = jnp.maximum(t, r), jnp.minimum(t, r)

        data_size = int(batch_size * (1 - self.config['flow_ratio']))
        zero_mask = jnp.arange(batch_size) < data_size
        zero_mask = zero_mask.reshape(batch_size, 1)
        r = jnp.where(zero_mask, t, r)

        return t, r

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))  # fold in horizon_length together with action_dim
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng, drop_rng = jax.random.split(rng, 4)

        # BC mean flow loss.
        x = batch_actions

        t, r = self.sample_t_r(batch_size, t_rng)

        ##### It does not need to be normal distirbution
        e = self.sample_latent_dist(x_rng, (batch_size, action_dim))
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

        if self.config['mf_method'] == 'mf':
            u, dudt = jax.jvp(
                mean_flow_forward, 
                (z, t, r), 
                (v, jnp.ones_like(t), jnp.zeros_like(r))
            )
            u_tgt = v - jnp.clip(t - r, a_min=0.0, a_max=1.0) * dudt
            u_tgt = jax.lax.stop_gradient(u_tgt)
        elif self.config['mf_method'] == 'jit_mf':
            x_pred, dxdt = jax.jvp(
                mean_flow_forward, 
                (z, t, r), 
                (v, jnp.ones_like(t), jnp.zeros_like(r))
            )
            u, dudt = z - x_pred, v - dxdt
            u_tgt = v - jnp.clip(t - r, a_min=0.0, a_max=1.0) * dudt
            u_tgt = jax.lax.stop_gradient(u_tgt)

        loss = self.adaptive_l2_loss(batch_size, u - u_tgt)

        return loss, {
            'actor_loss': loss,
            'mf/u_mean': u.mean(),
            'mf/v_mean': v.mean(),
            'mf/dudt_mean': dudt.mean(),
        }

    def sample_latent_dist(self, x_rng, sample_shape):
        if self.config['latent_dist'] == 'normal':
            e = jax.random.normal(x_rng, sample_shape)
        elif self.config['latent_dist'] == 'truncated_normal':
            e = jax.random.truncated_normal(x_rng, sample_shape)
        elif self.config['latent_dist'] == 'uniform':
            e = jax.random.uniform(x_rng, sample_shape, minval=-1.0, maxval=1.0)
        elif self.config['latent_dist'] == 'simplex':
            e = -jnp.log(jax.random.uniform(x_rng, sample_shape)) # Exponential
            e = e / jnp.sum(e, axis=-1, keepdims=True) # Sum = 1
        elif self.config['latent_dist'] == 'sphere':
            e = jax.random.normal(x_rng, sample_shape)
            sq_sum = jnp.sum(jnp.square(e), axis=-1, keepdims=True)
            norm = jnp.sqrt(sq_sum + 1e-6)
            e = e / norm * jnp.sqrt(sample_shape[-1])
        elif self.config['latent_dist'] == 'beta':
            e = jax.random.beta(x_rng, 2.0, 2.0, sample_shape)
            e = e * 2.0 - 1.0
        return e

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
        e = self.sample_latent_dist(x_rng, (*observations.shape[: -len(self.config['ob_dims'])], latent_dim))
        actions = self.compute_flow_actions(observations, e)
        actions = jnp.reshape(
            actions, 
            (*observations.shape[: -len(self.config['ob_dims'])],  # batch_size
            self.config["horizon_length"], self.config["action_dim"])
        )
            
        return actions

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
        
        output = self.network.select('actor_bc_flow')(
            observation,
            noise, 
            t, 
            t - r
        )
        if self.config['mf_method'] == 'jit_mf':
            action = output
        elif self.config['mf_method'] == 'mf':
            action = noise - output
        action = jnp.clip(action, -1, 1)
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

        actor_bc_flow_def = ActorMeanFlowField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )

        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, full_actions, ex_times, ex_times)),
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


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='meanflow',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(128, 128, 128, 128),  # Value network hidden dimensions.
            critic_hidden_dims=(256, 256, 256, 256),  # Value network hidden dimensions.
            latent_actor_hidden_dims=(256, 256),
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
            rl_method='iql', # DDPG, IQL
            expectile_tau=0.9,
            flow_ratio=0.25,
            mf_method='jit_mf',
            late_update=False,
            latent_dist='uniform',
            extract_method='awr', # 'ddpg', 'awr',,
            alpha=1.0

        )
    )
    return config
