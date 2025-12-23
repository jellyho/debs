import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorMeanFlowField, Value, ActorVectorField

class FLOWQAgent(flax.struct.PyTreeNode):
    """Don't extract but select! with action chunking. 
    """
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the behavior policy evaluation loss"""
        # Q(s, a) <- R + \gamma V(s')
        batch_size = batch['actions'].shape[0]

        if self.config['rl_method'] == 'iql':
            next_v = self.network.select(f'value')(
                batch['next_observations']
            )

            target_v = batch['rewards'] + \
                (self.config['discount'] ** self.config["horizon_length"]) * \
                batch['masks'] * next_v

            qs = self.network.select('critic')(
                batch['observations'],
                actions=batch['actions'].reshape(batch_size, -1),
                params=grad_params
            ) # N, B, 1

            critic_loss = jnp.square(
                jnp.broadcast_to(target_v, qs.shape) - qs
            ).mean()

            q = qs.mean(axis=0) # For logging

        elif self.config['rl_method'] == 'ddpg':
            ## TBD
            rng, sample_rng = jax.random.split(rng, 2)
            next_actions = self.sample_actions(batch['next_observations'], rng=sample_rng)
            
            next_qs = self.network.select('target_critic')(
                batch['next_observations'], 
                actions=next_actions.reshape(batch_size, -1)
            )

            if self.config['num_critic'] > 1:
                if self.config['critic_agg'] == 'min':
                    next_q = jnp.min(next_qs, axis=0)
                else:
                    next_q = jnp.mean(next_qs, axis=0)
            else:
                next_q = next_qs
            
            qs = self.network.select('critic')(
                batch['observations'], 
                actions=batch['actions'].reshape(batch_size, -1),
                params=grad_params
            )

            target_q = batch['rewards'] + \
                (self.config['discount'] ** self.config["horizon_length"]) * \
            batch['masks'] * next_q

            critic_loss = jnp.square(
                qs - jnp.broadcast_to(target_q, qs.shape)
            ).mean()

            q = qs.mean(axis=0) # For logging

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def value_loss(self, batch, grad_params, rng):
        # Only applied when IQL
        batch_size = batch['actions'].shape[0]

        qs = self.network.select('target_critic')(
            batch['observations'],
            actions=batch['actions'].reshape(batch_size, -1),
        ) # N, B

        if self.config['num_critic'] > 1:
            if self.config['critic_agg'] == 'min':
                q = jnp.min(qs, axis=0)
            else:
                q = jnp.mean(qs, axis=0)
        else:
            q = qs

        v = self.network.select('value')(
            batch['observations'],
            params=grad_params
        ) # B
        g = jnp.where(q >= v, self.config['expectile_tau'], 1.0 - self.config['expectile_tau'])
        value_loss = (g * jnp.square(q - v)).mean()

        metrics = {
            # losses
            'value_loss': value_loss,
            # value stats
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'q_target_hist': q,
        }
        return value_loss, metrics

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))  # fold in horizon_length together with action_dim
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng, drop_rng = jax.random.split(rng, 4)

        # BC mean flow loss.
        x_0 = self.sample_latent_dist(x_rng, batch_size, action_dim)
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(
            batch['observations'], 
            x_t, 
            t, 
            params=grad_params
        )

        loss = jnp.mean(jnp.square(pred-vel))

        return loss, {
            'actor_loss': loss,
        }

    def sample_latent_dist(self, x_rng, batch_size, action_dim):
        if self.config['latent_dist'] == 'normal':
            e = jax.random.normal(x_rng, (batch_size, action_dim))
        elif self.config['latent_dist'] == 'uniform':
            e = jax.random.uniform(x_rng, (batch_size, action_dim), minval=-1.0, maxval=1.0)
        elif self.config['latent_dist'] == 'simplex':
            e = -jnp.log(jax.random.uniform(x_rng, (batch_size, action_dim))) # Exponential
            e = e / jnp.sum(e, axis=-1, keepdims=True) # Sum = 1
        elif self.config['latent_dist'] == 'sphere':
            e = jax.random.normal(x_rng, (batch_size, action_dim))
            sq_sum = jnp.sum(jnp.square(e), axis=-1, keepdims=True)
            norm = jnp.sqrt(sq_sum + 1e-6)
            e = e / norm * jnp.sqrt(action_dim)
        return e


    def latent_actor_loss(self, batch, grad_params, rng):
        observations = batch['observations']
        actions_gt = batch['actions'] # Dataset Actions (Ground Truth)
        batch_size = observations.shape[0]
        latent_dim = self.config['action_dim'] * self.config['horizon_length']
        actions_gt_flat = jnp.reshape(actions_gt, (batch_size, latent_dim))      
        
        ### Query latent actor
        if self.config['noisy_latent_actor']:
            rng, x_rng, l_rng = jax.random.split(rng, 3)
            normal = jax.random.normal(x_rng, (batch_size, latent_dim))
            z_pred = self.network.select('latent_actor')(
                observations, 
                normal,
                rng=l_rng,
                params=grad_params # <--- Gradients flow here
            )
        else:
            rng, l_rng = jax.random.split(rng, 2)
            z_pred = self.network.select('latent_actor')(
                observations, 
                rng=l_rng,
                params=grad_params # <--- Gradients flow here
            )

        if self.config['extract_method'] == 'supervised_awr':
            rng, x_rng = jax.random.split(rng, 2)
            e = self.sample_latent_dist(x_rng, batch_size, latent_dim)
            x_pred = self.compute_flow_actions(observations, e)
        else:
            x_pred = self.compute_flow_actions(observations, z_pred)

        if self.config['mf_method'] == 'mf':
            a_pred_flat = z_pred - jnp.reshape(x_pred, (batch_size, latent_dim))
        elif self.config['mf_method'] == 'jit_mf':
            a_pred_flat = jnp.reshape(x_pred, (batch_size, latent_dim))

        info_dict = {
            'z_norm': jnp.mean(jnp.square(z_pred)),
        }

        if self.config['extract_method'] == 'ddpg':
             # Q(s, a)
            qs = self.network.select('critic')(
                observations,
                a_pred_flat.reshape(batch_size, -1)
            )

            if self.config['num_critic'] > 1:
                q = jnp.mean(qs, axis=0)
            else:
                q = qs

            loss = -q.mean()
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            loss = lam * loss

        elif self.config['extract_method'] == 'awr': # works only IQL
            assert self.config['rl_method'] == 'iql'
             # Q(s, a)
            qs = self.network.select('critic')(
                observations,
                actions_gt_flat
            )

            if self.config['num_critic'] > 1:
                q = jnp.mean(qs, axis=0)
            else:
                q = qs

            v = self.network.select('value')(observations)

            adv = q - v # (B,) 
            tau = self.config.get('tau', 1.0)
            weight = jnp.exp(adv / tau)
            weight = jnp.clip(weight, 0.0, 100.0)
            weight = jax.lax.stop_gradient(weight) # (B,)

            # Flow의 Jacobian을 타고 z_pred로 Gradient가 전달됨
            sq_err = jnp.mean(jnp.square(a_pred_flat - actions_gt_flat), axis=-1)
            loss = jnp.mean(weight * sq_err)

            info_dict['adv_mean'] = adv.mean()
            info_dict['adv_hist'] = adv
            info_dict['weight_mean'] = weight.mean()
        elif self.config['extract_method'] == 'supervised_awr': # works only IQL
            assert self.config['rl_method'] == 'iql'
             # Q(s, a)
            qs = self.network.select('critic')(
                observations,
                a_pred_flat
            )

            if self.config['num_critic'] > 1:
                q = jnp.mean(qs, axis=0)
            else:
                q = qs

            v = self.network.select('value')(observations)

            adv = q - v # (B,) 
            tau = self.config.get('tau', 1.0)
            weight = jnp.exp(adv / tau)
            weight = jnp.clip(weight, 0.0, 100.0)
            weight = jax.lax.stop_gradient(weight) # (B,)

            # Flow의 Jacobian을 타고 z_pred로 Gradient가 전달됨
            sq_err = jnp.mean(jnp.square(e - z_pred), axis=-1)
            loss = jnp.mean(weight * sq_err)

            info_dict['adv_mean'] = adv.mean()
            info_dict['adv_hist'] = adv
            info_dict['weight_mean'] = weight.mean()

        info_dict['latent_loss'] = loss

        return loss, info_dict

    @jax.jit
    def total_latent_actor_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng = jax.random.split(rng, 2)

        actor_loss, actor_info = self.latent_actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'latent/{k}'] = v

        return actor_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng, value_rng = jax.random.split(rng, 4)

        loss = 0

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        loss += critic_loss
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        if self.config['rl_method'] == 'iql':
            value_loss, value_info = self.value_loss(batch, grad_params, value_rng)
            for k, v in value_info.items():
                info[f'value/{k}'] = v
            loss += value_loss

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        loss += actor_loss
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        if not self.config['late_update']:
            latent_loss, latent_info = self.latent_actor_loss(batch, grad_params, actor_rng)
            loss += latent_loss
            for k, v in latent_info.items():
                info[f'latent/{k}'] = v

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
    def _latent_actor_update(agent, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_latent_actor_loss(batch, grad_params, rng=rng)
        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def latent_actor_update(self, batch):
        return self._latent_actor_update(self, batch)

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
        if self.config['rl_method'] == 'iql':
            return self.network.select('value')(observations)
        else:
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
        latent_dim = self.config["horizon_length"] * self.config["action_dim"]

        if self.config['noisy_latent_actor']:
            rng, x_rng = jax.random.split(rng, 2)
            normal = jax.random.normal(x_rng, 
                (*observations.shape[: -len(self.config['ob_dims'])], latent_dim)
            )
            noises = self.network.select('latent_actor')(
                observations, 
                normal,
            )
        else:
            noises = self.network.select('latent_actor')(
                observations, 
            )
        actions = self.compute_flow_actions(observations, noises)
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
        
        # full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        full_actions = jnp.reshape(
            ex_actions,
            (ex_actions.shape[0], -1)
        )
        full_action_dim = full_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['value'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()

        # Define networks.

        if config['rl_method'] == 'iql':
            value_def = Value(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                encoder=encoders.get('value')
            )

        critic_def = Value(
            hidden_dims=config['critic_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_critic'],
            encoder=encoders.get('critic'),
        )

        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )

        latent_actor_def = ActorVectorField(
            hidden_dims=config['latent_actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
            latent_dist=config['latent_dist']
        )

        if config['noisy_latent_actor']:
            latent_actor_input_shape = (ex_observations, full_actions)
        else:
            latent_actor_input_shape = (ex_observations)

        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, full_actions, ex_times)),
            latent_actor=(latent_actor_def, latent_actor_input_shape),
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
        )

        if config['rl_method'] == 'iql':
            network_info['value'] = value_def, (ex_observations)

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
            agent_name='meanflowq',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(256, 256, 256, 256),  # Value network hidden dimensions.
            critic_hidden_dims=(128, 128, 128, 128),  # Value network hidden dimensions.
            latent_actor_hidden_dims=(128, 128),
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
            noisy_latent_actor=False
        )
    )
    return config
