import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value, ActorMeanVectorField

SCALE = 1.0
class ACFQLAgent(flax.struct.PyTreeNode):
    """Flow Q-learning (FQL) agent with action chunking. 
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def offline_actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))  # fold in horizon_length together with action_dim
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim)) * SCALE
        
        def stopgrad(x):
            return jax.lax.stop_gradient(x)

        def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
            """
            Adaptive L2 for JAX arrays with shape (B, ...).
            Compute per-sample squared error mean over non-batch dims,
            then weight via w = 1 / (delta_sq + c)^{1-gamma}, stopgrad(w).
            """
            # error: (B, ...)
            if error.ndim == 1:
                delta_sq = error ** 2  # (B,)
            else:
                # mean over non-batch dims
                delta_sq = jnp.square(error)
            p = 1.0 - gamma
            w = 1.0 / jnp.power(delta_sq + c, p)  # (B,)
            loss_per_sample = delta_sq  # (B,)
            return stopgrad(w) * loss_per_sample

            # -------------------------
            # sample t, r with r <= t
        # -------------------------
        def sample_t_r(rng, batch_size):
            """
            JAX version of sample_t_r

            Args:
                rng: jax.random.PRNGKey
                batch_size: int
                time_dist: tuple, e.g. ('uniform',) or ('lognorm', mu, sigma)
                flow_ratio: float, in [0,1], fraction of samples where r=t

            Returns:
                t: (batch_size, 1) jnp.array
                r: (batch_size, 1) jnp.array
            """
            
            mu, sigma = -0.4, 1.0
            rng, subrng = jax.random.split(rng)
            normal_samples = jax.random.normal(subrng, shape=(batch_size, 2)) * sigma + mu
            samples = jax.nn.sigmoid(normal_samples)  # squash to [0,1]

            # Assign t=max, r=min
            t = jnp.maximum(samples[:, 0], samples[:, 1]).reshape(batch_size, 1)
            r = jnp.minimum(samples[:, 0], samples[:, 1]).reshape(batch_size, 1)
            return t, r
            
        B = batch_size//2 # half for MF and half for IVC
        t, r = sample_t_r(t_rng, B)  # t,r are (B,1)
        dual_t = jnp.concatenate([t, t], axis=0)
        dual_r = jnp.concatenate([r, t], axis=0)
        dual_obs = batch['observations']
        dual_actions = batch_actions
        dual_x0 = x_0
        dual_vaild_mask = batch["valid"]

        # interpolation & target velocity
        x_1 = dual_actions
        x_t = (1.0 - dual_t) * x_1 + dual_t * dual_x0  # (B,D)
        v = dual_x0 - x_1  # (B,D) target velocity

        # define function for jvp: takes (zz, tt, rr)
        # It's important the function is pure: no RNG inside
        def func(oo, aa, tt, rr):
            # return apply_fn(params, zz, tt, rr)  # (B,D)
            return self.network.select('actor_mean_flow')(oo, aa, tt, rr, params=grad_params)
        # tangents: (zeros_like(o), v, 1, 0)
        tangents = (jnp.zeros_like(dual_obs), v, jnp.ones_like(dual_t), jnp.zeros_like(dual_r))

        # jvp: returns primal_out, tangent_out
        # jax.jvp supports batching naturally when arrays are batched
        primal, jvp_out = jax.jvp(func, (dual_obs, x_t, dual_t, dual_r), tangents)

        u = primal  # (B,D)
        directional = jvp_out  # (B,D)

        factor = (dual_t - dual_r)  # (B,1)
        # broadcast factor to shape (B,D)
        factor_b = jnp.broadcast_to(factor, (B*2, action_dim))
        u_tgt = v - factor_b * directional  # (B,D)

        err = u - stopgrad(u_tgt)  # (B,D)
        # meanflow_loss = adaptive_l2_loss(err, gamma=0.5, c=1e-3)
        # only bc on the valid chunk indices
            
        bc_flow_loss_MF = jnp.mean(
            jnp.reshape(
                adaptive_l2_loss(err[:B], gamma=0.5, c=1e-3), 
                (B, self.config["horizon_length"], self.config["action_dim"]) 
            ) * dual_vaild_mask[:B]
        )
        bc_flow_loss_IVC = jnp.mean(
            jnp.reshape(
                adaptive_l2_loss(err[B:], gamma=0.5, c=1e-3),
                (B, self.config["horizon_length"], self.config["action_dim"])
            ) * dual_vaild_mask[B:]
        )
        bc_flow_loss = bc_flow_loss_MF + bc_flow_loss_IVC
        actor_loss = bc_flow_loss

        return actor_loss, {
            'actor_loss': actor_loss,
        }

    @jax.jit
    def offline_total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info, next_actions = self.get_critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.offline_actor_loss(batch, grad_params, actor_rng)
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
    def _offline_update(agent, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.offline_total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')

        if agent.config.get('use_smooth_critic', False):
            agent.target_update(new_network, 'smooth_critic')

        return agent.replace(network=new_network, rng=new_rng), info
    
    @jax.jit
    def update(self, batch):
        return self._offline_update(self, batch)
    
    @jax.jit
    def batch_update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        # update_size = batch["observations"].shape[0]
        agent, infos = jax.lax.scan(self._offline_update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)
    
    @jax.jit
    def sample_actions(
        self,
        observations,
        rng=None,
    ):
        actions = self.network.select(f'actor_onestep_flow')(observations, noises)
        actions = jnp.clip(actions, -1, 1)            
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
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def compute_mean_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_mean_flow_encoder')(observations)
        t = jnp.full((*observations.shape[:-1], 1), 1.0)
        r = jnp.full((*observations.shape[:-1], 1), 0.0)
        vels = self.network.select('actor_mean_flow')(observations, noises, t, r, is_encoded=True)
        actions = noises - vels
        actions = jnp.clip(actions, -1, 1)
        return actions

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
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['smooth_critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()
            encoders['actor_mean_flow'] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('critic'),
        )
        
        smooth_critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('smooth_critic'),
        )

        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
            use_fourier_features=config["use_fourier_features"],
            fourier_feature_dim=config["fourier_feature_dim"],
        )

        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )

        actor_mean_flow_def = ActorMeanVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_mean_flow'),
        )

        
        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, full_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, full_actions)),
            actor_mean_flow=(actor_mean_flow_def, (ex_observations, full_actions, ex_times, ex_times)),
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            smooth_critic=(smooth_critic_def, (ex_observations, full_actions)),
            target_smooth_critic=(copy.deepcopy(smooth_critic_def), (ex_observations, full_actions)),
        )
        if encoders.get('actor_bc_flow') is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
        if encoders.get('actor_mean_flow') is not None:
            # Add actor_mean_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_mean_flow_encoder'] = (encoders.get('actor_mean_flow'), (ex_observations,))
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
        params[f'modules_target_smooth_critic'] = params[f'modules_smooth_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():

    config = ml_collections.ConfigDict(
        dict(
            agent_name='acfql',  # Agent name.
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
            q_agg='mean',  # Aggregation method for target Q values.
            alpha=100.0,  # BC coefficient (need to be tuned for each environment).
            num_qs=2, # critic ensemble size
            flow_steps=10,  # Number of flow steps.
            use_mean_flow=True,  # Whether to use mean flow.
            IVC_alpha=1.0,
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            horizon_length=ml_collections.config_dict.placeholder(int), # will be set
            action_chunking=True,  # False means n-step return
            actor_type="distill-ddpg",
            actor_num_samples=32,  # for actor_type="best-of-n" only
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.,
            cql_alpha=1.0,  # CQL regularization coefficient
            use_smooth_critic=False,  # Whether to use smooth critic
            smooth_critic_pev_weight=0.0,
            smooth_critic_pim_weight=0.0,  # Weight for the smooth critic penalty in the loss
            use_coherence_noise=False,  # Whether to use coherence noise
        )
    )
    return config