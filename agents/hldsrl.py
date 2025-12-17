import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import AdvantageConditionedActorVectorField, QuantileValue
from utils.hlg import _normal_cdf_log_difference

class HLDSRLAgent(flax.struct.PyTreeNode):
    """Don't extract but select! with action chunking. 
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the behavior policy evaluation loss"""
        target_v = self._compute_scalar_target(batch) # B, 1

        target_probs = self._scalar_to_prob(target_v)

        current_logits = self.network.select('critic')(
            batch['observations'],
            params=grad_params
        ) # B, Q

        critic_loss = self._compute_dist_loss(
            current_logits, target_probs, batch['valid'][..., -1]
        )

        current_v = self._logit_to_scalar(current_logits)

        return critic_loss, {
            'critic_loss': critic_loss,
            'v_mean': current_v.mean(),
            'v_max': current_v.max(),
            'v_min': current_v.min(),
            'target_v_hist': target_v,
        }

    def _prob_to_scalar(self, dist_probs):
        """Convert distributional probabilities to scalar value."""
        support = jnp.linspace(
            self.config['v_min'], self.config['v_max'], self.config['num_bins'] + 1
        )
        centers = (support[:-1] + support[1:]) / 2
        scalar_value = jnp.sum(dist_probs * centers, axis=-1)
        return scalar_value

    def _logit_to_scalar(self, dist_logits):
        """Convert distributional probabilities to scalar value."""
        dist_probs = jax.nn.softmax(dist_logits, axis=-1)
        scalar_value = self._prob_to_scalar(dist_probs)
        return scalar_value

    def _scalar_to_prob(self, target_scalar):
        sigma = self.config['sigma'] * (self.config['v_max'] - self.config['v_min']) / (self.config['num_bins'] + 1)

        support = jnp.linspace(
            self.config['v_min'], self.config['v_max'], self.config['num_bins'] + 1
        )

        bin_log_probs = _normal_cdf_log_difference(
            (support[1:] - target_scalar) / (jnp.sqrt(2) * sigma),
            (support[:-1] - target_scalar) / (jnp.sqrt(2) * sigma),
        )
        log_z = _normal_cdf_log_difference(
            (support[-1] - target_scalar) / (jnp.sqrt(2) * sigma),
            (support[0] - target_scalar) / (jnp.sqrt(2) * sigma),
        )

        return jax.lax.stop_gradient(jnp.exp(bin_log_probs - log_z))

    def _compute_scalar_target(self, batch):
        """Calculates the scalar Bellman target: r + gamma * E[V(s')]"""
        
        # 1. Get Next State Distribution (Logits)
        next_logits = self.network.select(f'target_critic')(
            batch['next_observations'][..., -1, :]
        )

        next_v = self._logit_to_scalar(next_logits)
        
        # 3. Bellman Update (Scalar)
        T = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config["horizon_length"]) * \
            batch['masks'][..., -1] * next_v
            
        if T.ndim == 1:
            T = T[..., None]
            
        return jax.lax.stop_gradient(T)

    def _compute_dist_loss(self, current_logits, target_probs, valid_mask):
        """
        current_zs: (Batch, Num_Bins)
        target_z: (Batch, Num_Bins)
        """
        if self.config['target_mode'] == 'expectile':
            cv = self._logit_to_scalar(current_logits)
            tv = self._prob_to_scalar(target_probs)
            g_hard = jnp.where(tv >= cv, self.config['expectile_tau'], 1.0 - self.config['expectile_tau'])
            g_hard=  g_hard * valid_mask
        else:
            g_hard = jnp.ones_like(valid_mask)

        log_probs = jax.nn.log_softmax(current_logits, axis=-1)

        ce_loss = -jnp.sum(target_probs * log_probs, axis=-1)
        
        critic_loss = (ce_loss * g_hard).mean()
        
        return critic_loss

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))  # fold in horizon_length together with action_dim
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng, drop_rng = jax.random.split(rng, 4)

        # BC flow loss.
        g = self._compute_normalized_advantage(batch, params=grad_params)

        x_score = jnp.ones((batch_size, 1)) * (g * alpha)
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        uncond_mask = jax.random.bernoulli(drop_rng, p=self.config['cfg_dropout'], shape=(batch_size, 1))

        # coeff = jnp.where(uncond_mask, 1.0, g >= 0.0).astype(jnp.float32)

        pred_cond = self.network.select('actor_bc_flow')(
            batch['observations'], 
            x_t, 
            advantage=g, 
            times=t,
            mask=uncond_mask,
            params=grad_params
        )

        bc_flow_loss_cond = jnp.mean(
            jnp.reshape(
                (pred_cond - vel) ** 2, 
                (batch_size, self.config["horizon_length"], self.config["action_dim"]) 
            ) * batch["valid"][..., None] # * coeff[..., None]
        )

        bc_flow_loss = bc_flow_loss_cond 

        return bc_flow_loss, {
            'actor_loss': bc_flow_loss,
            'g/mean': g.mean(),
            'g/max': g.max(),
            'g/min': g.min(),
            'g/std': g.std(),
            'g/hist': g
        }

    def _compute_normalized_advantage(self, batch, params):
        """
        Computes Percentile Rank (g) using the learned distribution's CDF.
        g = P(Z < T) where T is the target return.
        """
        # 1. Get Current V(s) Distribution
        logits = self.network.select('critic')(batch['observations'], params=params)
        probs = jax.nn.softmax(logits, axis=-1)
        
        if self.config['num_qs'] > 1:
            probs = probs.mean(axis=0)
        
        # 2. Get Scalar Target T (Same as Bellman Target)
        # "이 행동을 했을 때 실제로 기대되는 점수"
        T = self._compute_scalar_target(batch) # (Batch, 1)
        bin_width = (self.config['v_max'] - self.config['v_min']) / (self.config['num_bins'] + 1)
        idx = jnp.floor((T - self.config['v_min']) / bin_width)

        cdf = probs.cumsum(axis=-1) # (Batch, N)
        cdf_extended = jnp.concatenate(
            [
                jnp.zeros((*cdf.shape[:-1], 1)),  # P(Z < v_min) = 0
                cdf,
            ],
            axis=-1
        )  # (Batch, N + 1)
        idx = jnp.clip(idx, 0, self.config['num_bins']).astype(jnp.int32)
        g = jnp.take_along_axis(cdf_extended, idx, axis=1)
        g = 2.0 * g - 1.0
            
        return jax.lax.stop_gradient(g)

    @jax.jit
    def total_critic_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        rng, critic_rng = jax.random.split(rng, 2)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        return critic_loss, info

    @jax.jit
    def total_actor_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng = jax.random.split(rng, 2)

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        return actor_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(agent, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        return agent.replace(network=new_network, rng=new_rng), info
    
    @staticmethod
    def _critic_update(agent, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_critic_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        return agent.replace(network=new_network, rng=new_rng), info

    @staticmethod
    def _actor_update(agent, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_actor_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def critic_update(self, batch):
        return self._critic_update(self, batch)
    
    @jax.jit
    def actor_update(self, batch):
        return self._actor_update(self, batch)

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
        return self.network.select('critic')(observations)
    
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
        
        # 1. Prepare Action Dimension (Handle Chunking)
        flat_action_dim = self.config['action_dim'] * self.config['horizon_length']

        # 2. Sample Initial Noise
        rng, sample_rng = jax.random.split(rng)

        noises = jax.random.normal(
            rng,
            (
                *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                self.config['action_dim'] * \
                    (self.config['horizon_length'] if self.config["action_chunking"] else 1),
            ),
        )
        
        # 3. Compute Actions (Delegated to core logic)
        # Observations are passed raw; compute_flow_actions handles encoding.
        actions = self.compute_flow_actions(observations, noises)
        
        # 4. Reshape if Chunking
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
        
        action = noise        
        g_pos = jnp.ones((*observation.shape[:-1], 1), dtype=jnp.float32)
        # g_neg = -jnp.ones((*observation.shape[:-1], 1), dtype=jnp.float32)
        uncond_mask = jnp.ones((*observation.shape[:-1], 1), dtype=jnp.float32)
        cond_mask = jnp.zeros((*observation.shape[:-1], 1), dtype=jnp.float32)

        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observation.shape[:-1], 1), i / self.config['flow_steps'])
            vel = self.network.select('actor_bc_flow')(
                observation, 
                action, 
                g_pos, 
                t,
                cond_mask,
                is_encoded=True
            )
            uncond_vel = self.network.select('actor_bc_flow')(
                observation, 
                action, 
                g_pos, 
                t, 
                uncond_mask,
                is_encoded=True
            )
            vel = uncond_vel + self.config['cfg'] * (vel - uncond_vel)
            # vel = uncond_vel
            action = action + vel / self.config['flow_steps']
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

        ex_times = ex_actions[..., :1]
        ex_advantage = ex_actions[..., :1]

        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        
        full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        full_action_dim = full_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        # Define networks.
        critic_def = QuantileValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('critic'),
            num_quantiles=config['num_bins']
        )

        actor_bc_flow_def = AdvantageConditionedActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
            use_fourier_features=config["use_fourier_features"],
            fourier_feature_dim=config["fourier_feature_dim"],
            advantage_fourier_feature_dim=config["advantage_fourier_feature_dim"],
            use_cfg=True
        )

        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, full_actions, ex_advantage, ex_advantage, ex_times)),
            critic=(critic_def, (ex_observations)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations)),
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
        params[f'modules_target_critic'] = params[f'modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='hldsrl',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            v_agg='mean',  # Aggregation method for target Q values.
            num_qs=1, # critic ensemble size
            flow_steps=10,  # Number of flow steps.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            horizon_length=ml_collections.config_dict.placeholder(int), # will be set
            action_chunking=True,  # False means n-step return
            use_fourier_features=False,
            fourier_feature_dim=64,
            advantage_fourier_feature_dim=64,
            weight_decay=0.,
            target_mode='expectile',
            expectile_tau=0.9,
            num_bins=51,
            v_min=-100.0,
            v_max=0.0,
            quantile_bound=0.05,
            alpha=1.0,
            cfg=1.0,
            cfg_dropout=0.1,
            sigma=0.75
        )
    )
    return config
