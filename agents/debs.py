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
        # Shape: (Batch, Num_Quantiles) <--- Corrected
        next_zs = self.network.select(f'target_critic')(batch['next_observations'][..., -1, :]) # B, Q
        
        # 2. Aggregate Ensembles (Mixture Distribution)
        # Shape: (Batch, Num_Quantiles)
        if self.config['num_qs'] > 1:
            if self.config['v_agg'] == 'min':
                next_z = next_zs.min(axis=0)
            else:
                next_z = next_zs.mean(axis=0)
        else:
            next_z = next_zs
        
        # 3. Apply Bellman Operator
        # reward: (Batch, 1), mask: (Batch, 1)
        # target_z: (Batch, Num_Quantiles)
        reward = batch['rewards'][..., -1][..., None] # B, 1
        mask = batch['masks'][..., -1][..., None] # B, 1
        discount = self.config['discount'] ** self.config["horizon_length"] # 1,
        
        target_z = reward + discount * mask * next_z
        return jax.lax.stop_gradient(target_z)

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

        # print('ewl', total_loss.shape)

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
                    (pred_cond - vel) ** 2, 
                    (batch_size, self.config["horizon_length"], self.config["action_dim"]) 
                ) * batch["valid"][..., None]
            )

            bc_flow_loss = loss_cond + self.config['cfg_dropout'] * loss_uncond
        else:
            pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, advantage=g, times=t, params=grad_params)
            # only bc on the valid chunk indices
            bc_flow_loss = jnp.mean(
                jnp.reshape(
                    (pred - vel) ** 2, 
                    (batch_size, self.config["horizon_length"], self.config["action_dim"]) 
                ) * batch["valid"][..., None]
            )

        return bc_flow_loss, {
            'actor_loss': bc_flow_loss,
        }

    def _compute_normalized_advantage(self, batch, params):
        """
        Computes the normalized advantage 'g' from the V-distribution Z(s).
        g = (T - V_min) / (V_max - V_min)
        """
        # 1. Get Current State Value Distribution Z(s)
        # Shape: (Num_Ensembles, Batch, Num_Quantiles)
        # Renamed to 'zs' to match distributional RL convention
        zs = self.network.select('critic')(batch['observations'], params=params)
        
        # 2. Robust Min/Max Estimation
        # Average over ensembles first for stability -> (Batch, Num_Quantiles)
        if self.config['num_qs'] > 1:
            if self.config['v_agg'] == 'min':
                z = zs.min(axis=0)
            else:
                z = zs.mean(axis=0)
        else:
            z = zs
        
        # Sort quantiles (Batch, Num_Quantiles)
        # Since the network output isn't guaranteed to be sorted, we sort it here
        z_sorted = jnp.sort(z, axis=-1)
        
        # Select indices based on config['quantile_bound'] (e.g., 0.05)
        N = self.config['num_quantiles']
        bound = self.config['quantile_bound']
        
        idx_min = int(N * bound)            # e.g., 5%
        idx_max = int(N * (1.0 - bound))    # e.g., 95%
        
        v_min = z_sorted[..., idx_min]
        v_max = z_sorted[..., idx_max]
        
        # 3. Calculate Target Return T = r + gamma * E[Z(s')]
        # Next state distribution from Target V-Network
        # Shape: (Num_Ensembles, Batch, Num_Quantiles)
        next_zs = self.network.select('target_critic')(
            batch['next_observations'][..., -1, :]
        )
        
        # Mean over ensembles and quantiles -> Expected Value E[V(s')]
        if self.config['num_qs'] > 1:
            if self.config['v_agg'] == 'min':
                next_z = next_zs.min(axis=0)
            else:
                next_z = next_zs.mean(axis=0)
        else:
            next_z = next_zs

        next_z_mean = next_z.mean(axis=-1)
        
        # Bellman Target T (Scalar Return)
        T = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config["horizon_length"]) * \
            batch['masks'][..., -1] * next_z_mean

        # 4. Calculate Normalized g
        # Add epsilon to prevent division by zero
        denom = jnp.maximum(v_max - v_min, 1e-6)
        g = 2.0 * (T - v_min) / denom - 1
        
        # Clip g to [-1, 1]
        g = jnp.clip(g, -1.0, 1.0)
        
        # Ensure shape (Batch, 1)
        if g.ndim == 1:
            g = g[..., None]
            
        # IMPORTANT: Stop gradient so actor update doesn't affect critic
        return jax.lax.stop_gradient(g)

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
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        
        actions = noises
        # batch_size = actions.shape[0]
        
        g_pos = jnp.ones((1), dtype=jnp.float32)

        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, g_pos, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    # @jax.jit
    # def compute_flow_actions(
    #     self,
    #     observations,
    #     noises,
    # ):
    #     """
    #     Compute actions from the BC flow model using Euler method with CFG support.
        
    #     Logic:
    #     1. cfg == 1.0: Standard Conditional Generation (g=1).
    #     2. use_bad_to_good_cfg: Extrapolate from Bad(g=-1) to Good(g=1).
    #     3. use_cfg: Extrapolate from Uncond(g=None) to Cond(g=1).
    #     """
    #     # 1. Encode observations once (Efficiency)
    #     if self.config['encoder'] is not None:
    #         observations = self.network.select('actor_bc_flow_encoder')(observations)
            
    #     actions = noises
    #     batch_size = actions.shape[0]
    #     cfg_scale = self.config['cfg']
        
    #     # Prepare G conditions
    #     # Positive Target: 1.0
    #     g_pos = jnp.ones((batch_size, 1), dtype=jnp.float32)
        
    #     # Negative Target (only for bad_to_good)
    #     # Negative: -1.0
    #     g_neg_array = -1.0 * jnp.ones((batch_size, 1), dtype=jnp.float32)

    #     # Euler loop
    #     dt = 1.0 / self.config['flow_steps']
        
    #     def body_fn(i, x):
    #         t_val = i * dt
    #         t_batch = jnp.full((batch_size, 1), t_val)
            
    #         # --- Branching Logic for CFG ---
            
    #         # Case A: No Guidance (cfg=1.0)
    #         if cfg_scale == 1.0:
    #             vel = self.network.select('actor_bc_flow')(
    #                 observations, x, advantage=g_pos, times=t_batch, is_encoded=True
    #             )
                
    #         # Case B: Bad-to-Good Guidance (-1 to 1) -> Batch Doubling Optimization
    #         elif self.config['use_bad_to_good_cfg']:
    #             # Concatenate inputs to run in a single forward pass
    #             obs_in = jnp.concatenate([observations, observations], axis=0)
    #             x_in = jnp.concatenate([x, x], axis=0)
    #             t_in = jnp.concatenate([t_batch, t_batch], axis=0)
    #             # Cond: [Positive(1), Negative(-1)]
    #             g_in = jnp.concatenate([g_pos, g_neg_array], axis=0)
                
    #             v_out = self.network.select('actor_bc_flow')(
    #                 obs_in, x_in, advantage=g_in, times=t_in, is_encoded=True
    #             )
                
    #             v_pos, v_neg = jnp.split(v_out, 2, axis=0)
    #             vel = v_neg + cfg_scale * (v_pos - v_neg)

    #         # Case C: Standard CFG (None to 1) -> Two-pass (since None cannot be batched with Array)
    #         elif self.config['use_cfg']:
    #             # 1. Conditional (g=1)
    #             v_pos = self.network.select('actor_bc_flow')(
    #                 observations, x, advantage=g_pos, times=t_batch, is_encoded=True
    #             )
    #             # 2. Unconditional (g=None -> Null Embedding)
    #             v_uncond = self.network.select('actor_bc_flow')(
    #                 observations, x, advantage=None, times=t_batch, is_encoded=True
    #             )
                
    #             vel = v_uncond + cfg_scale * (v_pos - v_uncond)
                
    #         else:
    #             # Fallback (Should not happen given configs, treat as Case A)
    #             vel = self.network.select('actor_bc_flow')(
    #                 observations, x, advantage=g_pos, times=t_batch, is_encoded=True
    #             )

    #         return x + vel * dt

    #     # Run Loop
    #     actions = jax.lax.fori_loop(0, self.config['flow_steps'], body_fn, actions)
        
    #     # Clip final actions
    #     actions = jnp.clip(actions, -1.0, 1.0)
    #     return actions

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
            num_quantiles=51,
            quantile_bound=0.05,
            use_cfg=False,
            use_bad_to_good_cfg=False,
            cfg=1.0,
            cfg_dropout=0.1
        )
    )
    return config
