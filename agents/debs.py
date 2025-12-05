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

class DEBSAgent(flax.struct.PyTreeNode):
    """Don't extract but select! with action chunking. 
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the behavior policy evaluation loss"""

        target_z = self._compute_target_quantiles(batch) # B, Q
        current_zs = self.network.select('critic')(batch['observations'], params=grad_params) # B, Q
        critic_loss = self._compute_quantile_loss(
            current_zs, target_z, batch['valid'][..., -1]
        )

        avg_z = current_zs.mean(axis=0)  # (Batch, Num_Quantiles)

        return critic_loss, {
            'critic_loss': critic_loss,
            'v_mean': avg_z.mean(),
            'v_max': avg_z[..., -1].mean(),
            'v_min': avg_z[..., 0].mean(),
        }

    def _compute_target_quantiles(self, batch):
        # 1. Get Next State Distribution
        next_zs = self.network.select(f'target_critic')(
            batch['next_observations'][..., -1, :]
        )
        
        # 2. Aggregate Ensembles (Mixture)
        if self.config['num_qs'] > 1:
            # 기본적으로는 mean으로 섞습니다.
            next_z_dist = next_zs.mean(axis=0) 
        else:
            next_z_dist = next_zs

        # -------------------------------------------------------
        # [New] Plug-and-Play Target Selection
        # -------------------------------------------------------
        target_mode = self.config.get('target_mode', 'mean') # 'mean' or 'expectile'
        
        if target_mode == 'mean':
            # [Standard DEBS] SARSA Style
            # 분포의 평균(Expectation)을 사용 -> Behavior Value
            next_v = next_z_dist.mean(axis=-1)
            
        elif target_mode == 'expectile':
            # [Optimistic DEBS] IQL Style
            # 분포의 상위권(Expectile)을 사용 -> Optimal Value Approximation
            # Quantile 분포에서 Expectile을 구하는 것은
            # 단순히 상위 tau% 지점의 값을 취하는 것과 유사합니다.
            tau = self.config.get('expectile_tau', 0.9) # 예: 0.9
            
            # 정렬 후 해당 인덱스 선택 (Quantile function)
            sorted_z = jnp.sort(next_z_dist, axis=-1)
            idx = int(self.config['num_quantiles'] * tau)
            next_v = sorted_z[..., idx]
            
        # 3. Bellman Update
        T = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config["horizon_length"]) * \
            batch['masks'][..., -1] * next_v
            
        # Quantile Regression을 하려면 Target은 (Batch, 1) 이고
        # 이를 N번 복사해서 모든 Quantile이 이 Target을 맞추도록 해야 함
        if T.ndim == 1:
            T = T[..., None]
            
        return jax.lax.stop_gradient(T)

    def _compute_quantile_loss(self, current_zs, target_z, valid_mask):
        """
        current_zs: (Batch, Num_Quantiles)
        target_z: (Batch, Num_Quantiles)
        """
        num_quantiles = self.config['num_quantiles']
        
        # Tau: (Num_Quantiles,)
        tau = (jnp.arange(num_quantiles, dtype=jnp.float32) + 0.5) / num_quantiles
        
        # Loss function for a single ensemble member
        # pred_z: (Batch, N), tgt_z: (Batch, N)
        # def single_ensemble_loss_fn(pred_z, tgt_z):
        #     # Calculate Pairwise Difference
        #     # (Batch, N, 1) - (Batch, 1, N) -> (Batch, N, N)
        u = target_z[..., None, :] - current_zs[..., :, None]
        
        # Huber Loss
        kappa = 1.0
        abs_u = jnp.abs(u)
        huber_loss = jnp.where(
            abs_u <= kappa, 
            0.5 * u**2, 
            kappa * (abs_u - 0.5 * kappa)
        )
        
        # Quantile Weighting
        weights = jnp.abs(tau[..., None] - (u < 0).astype(jnp.float32))
        element_wise_loss = weights * huber_loss
        
        # Sum over targets (last dim), Mean over predictions (2nd to last)
        total_loss = element_wise_loss.sum(axis=-1).mean(axis=-1)

        # Vectorize over axis 0 (Ensemble Dimension) of current_zs
        # target_z is broadcasted to all ensembles (in_axes=None)
        # per_ensemble_losses = jax.vmap(single_ensemble_loss_fn, in_axes=(0, None))(current_zs, target_z)
        
        # Sum losses across ensembles (axis 0)
        # Result Shape: (Batch,)
        # total_loss = per_ensemble_losses.sum(axis=0)
        
        # Apply mask and mean
        masked_loss = (total_loss * valid_mask).mean()
        
        return masked_loss

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))  # fold in horizon_length together with action_dim
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        g = self._compute_normalized_advantage(batch, params=grad_params)

        if self.config['use_cfg']:
            pred_cond = self.network.select('actor_bc_flow')(
            batch['observations'], x_t, advantage=g, times=t, params=grad_params
            )
            
            # B. Unconditional Forward (Advantage = None)
            pred_uncond = self.network.select('actor_bc_flow')(
                batch['observations'], x_t, advantage=None, times=t, params=grad_params
            )

            bc_flow_loss_cond = jnp.mean(
                jnp.reshape(
                    (pred_cond - vel) ** 2, 
                    (batch_size, self.config["horizon_length"], self.config["action_dim"]) 
                ) * batch["valid"][..., None]
            )

            bc_flow_loss_uncond = jnp.mean(
                jnp.reshape(
                    (pred_uncond - vel) ** 2, 
                    (batch_size, self.config["horizon_length"], self.config["action_dim"]) 
                ) * batch["valid"][..., None]
            )

            bc_flow_loss = bc_flow_loss_cond + self.config['cfg_dropout'] * bc_flow_loss_uncond
        else:
            pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, advantage=g, times=t, params=grad_params)
            # only bc on the valid chunk indices
            bc_flow_loss = jnp.mean(
                jnp.reshape(
                    (pred - vel) ** 2, 
                    (batch_size, self.config["horizon_length"], self.config["action_dim"]) 
                ) * batch["valid"][..., None]
            )

            # res_vel = vel - pred
            # res_pred = 

            # bc_residual_loss = self.network.select('actor_bc_flow')(batch['observations'], x_t, advantage=g, times=t, params=grad_params)

        return bc_flow_loss, {
            'actor_loss': bc_flow_loss,
            'g/mean': g.mean(),
            'g/max': g.max(),
            'g/min': g.min(),
            'g/std': g.std(),
        }

    def _compute_normalized_advantage(self, batch, params):
        """
        Computes the percentile rank of Target T within the predicted distribution Z(s).
        g \in [0, 1] (0: Worst, 0.5: Median, 1: Best)
        This is robust to scale and distribution shape.
        """
        # 1. Get Current State Value Distribution Z(s)
        # Shape: (Num_Ensembles, Batch, Num_Quantiles)
        zs = self.network.select('critic')(batch['observations'], params=params)
        
        # Aggregate Ensembles
        if self.config['num_qs'] > 1:
            # Use mean distribution for stability
            z = zs.mean(axis=0) 
        else:
            z = zs
        
        # Sort quantiles just in case (Batch, N)
        z_sorted = jnp.sort(z, axis=-1)
        
        # 2. Calculate Target Return T
        # (이전과 동일한 로직)
        next_zs = self.network.select('target_critic')(
            batch['next_observations'][..., -1, :], 
        )
        
        if self.config['num_qs'] > 1:
            next_z = next_zs.mean(axis=0)
        else:
            next_z = next_zs

        # Target Value: 보수적으로 하려면 mean 대신 quantile중 하나를 쓸 수도 있음
        # 여기서는 전체 기댓값 사용
        next_z_mean = next_z.mean(axis=-1)
        
        T = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config["horizon_length"]) * \
            batch['masks'][..., -1] * next_z_mean
            
        # Shape Check: T -> (Batch, 1)
        if T.ndim == 1:
            T = T[..., None]

        # 3. Compute Percentile Rank (g)
        # "T보다 작은 quantile이 몇 개인가?" / "전체 개수"
        # Broadcasting: T (B, 1) vs z_sorted (B, N)
        # count: (B, N) -> sum -> (B,)
        
        # Soft Indicator function (Sigmoid) for differentiability (optional) 
        # or Hard Indicator (Heaviside)
        # Hard Indicator is fine since we don't need gradients flowing back to T here
        
        # (B, 1) > (B, N) -> (B, N) boolean
        is_greater = (T >= z_sorted)
        
        # Mean over quantiles -> Percentile [0, 1]
        g = jnp.mean(is_greater, axis=-1)
        
        # 4. Scale to [-1, 1] (Optional, if you prefer centered range)
        g = 2.0 * g - 1.0
        
        # Ensure shape (Batch, 1)
        if g.ndim == 1:
            g = g[..., None]
            
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
            (self.config["horizon_length"], self.config["action_dim"])
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
        g_pos = jnp.ones((1), dtype=jnp.float32)
        g_neg = -jnp.ones((1), dtype=jnp.float32)

        # Euler method.
        for i in range(self.config['flow_steps']):
            observations = jnp.stack([observation, observation], axis=0)
            actions = jnp.stack([action, action], axis=0)
            gs = jnp.stack([g_pos, g_neg], axis=0)

            ts = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            
            vels = self.network.select('actor_bc_flow')(observations, actions, gs, ts, is_encoded=True)
            v_pos, v_neg = jnp.split(vels, 2, axis=0)
            vel = v_neg + self.config['cfg'] * (v_pos - v_neg)
            # print(v_neg.shape, v_pos.shape, vel.shape)
            action = action + vel[0] / self.config['flow_steps']
            # print(action.shape)
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
            num_quantiles=config['num_quantiles']
        )

        actor_bc_flow_def = AdvantageConditionedActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
            use_fourier_features=config["use_fourier_features"],
            fourier_feature_dim=config["fourier_feature_dim"],
            advantage_fourier_feature_dim=config["advantage_fourier_feature_dim"],
            use_cfg=config["use_cfg"]
        )

        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, full_actions, ex_advantage, ex_times)),
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
            agent_name='debs',  # Agent name.
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
            num_quantiles=51,
            quantile_bound=0.05,
            use_cfg=False,
            use_bad_to_good_cfg=False,
            cfg=1.0,
            cfg_dropout=0.1
        )
    )
    return config
