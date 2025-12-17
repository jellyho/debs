import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorMeanFlowField, QuantileValue, ActorVectorField
from utils.hlg import _normal_cdf_log_difference

class HLMEANFLOWAgent(flax.struct.PyTreeNode):
    """Don't extract but select! with action chunking. 
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the behavior policy evaluation loss"""
        target_vs = self._compute_scalar_target(batch) # N, B, 1

        # Aggregate values across ensemble
        if self.config['num_critic'] > 1:
            if self.config['critic_agg'] == 'min':
                target_v = jnp.min(target_vs, axis=0)
            else:
                target_v = jnp.mean(target_v, axis=0)
        else:
            target_v = target_vs

        target_probs = self._scalar_to_prob(target_v) # B, NAtoms

        current_logits = self.network.select('critic')(
            batch['observations'],
            params=grad_params
        ) # N, B, NA

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
            batch['next_observations']
        )

        next_v = self._logit_to_scalar(next_logits)
        
        # 3. Bellman Update (Scalar)
        T = batch['rewards'] + \
            (self.config['discount'] ** self.config["horizon_length"]) * \
            batch['masks'] * next_v
            
        T = T[..., None]
            
        return jax.lax.stop_gradient(T)

    def _compute_dist_loss(self, current_logits, target_probs, valid_mask):
        """
        current_zs: (Batch, Num_Bins)
        target_z: (Batch, Num_Bins)
        """
        if self.config['target_mode'] == 'expectile':
            cv = self._logit_to_scalar(current_logits) # N, B
            tv = jnp.broadcast_to(self._prob_to_scalar(target_probs), cv.shape) # N, B
            g_hard = jnp.where(tv >= cv, self.config['expectile_tau'], 1.0 - self.config['expectile_tau']) # N, B
        else:
            g_hard = jnp.ones_like(valid_mask) # B

        log_probs = jax.nn.log_softmax(current_logits, axis=-1) # N, B, NAtoms
        ce_loss = -jnp.sum(jnp.broadcast_to(target_probs, log_probs.shape) * log_probs, axis=-1) # N, B
        critic_loss = (ce_loss * g_hard).mean() # 
        print(g_hard.shape, ce_loss.shape)
        
        return critic_loss

    ## MF utils
    def adaptive_l2_loss(self, batch_size, error, valid_mask, p=0.5, c=1e-3):
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

        e = jax.random.normal(x_rng, (batch_size, action_dim))
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

        u, dudt = jax.jvp(
            mean_flow_forward, 
            (z, t, r), 
            (v, jnp.ones_like(t), jnp.zeros_like(r))
        )

        u_tgt = v - jnp.clip(t - r, a_min=0.0, a_max=1.0) * dudt
        u_tgt = jax.lax.stop_gradient(u_tgt)

        loss = self.adaptive_l2_loss(batch_size, u - u_tgt, batch['valid'])

        return loss, {
            'actor_loss': loss,
            'mf/u_mean': u.mean(),
            'mf/v_mean': v.mean(),
            'mf/dudt_mean': dudt.mean(),
        }

    def actor_loss_jit(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))  # fold in horizon_length together with action_dim
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng, drop_rng = jax.random.split(rng, 4)

        # BC mean flow loss.
        x = batch_actions

        t, r = self.sample_t_r(batch_size, t_rng)

        e = jax.random.normal(x_rng, (batch_size, action_dim))
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

        x_pred, dxdt = jax.jvp(
            mean_flow_forward, 
            (z, t, r), 
            (v, jnp.ones_like(t), jnp.zeros_like(r))
        )
        u, dudt = z - x_pred, v - dxdt
        u_tgt = v - jnp.clip(t - r, a_min=0.0, a_max=1.0) * dudt
        u_tgt = jax.lax.stop_gradient(u_tgt)

        loss = self.adaptive_l2_loss(batch_size, u - u_tgt, batch['valid'])

        return loss, {
            'actor_loss': loss,
            'mf/u_mean': u.mean(),
            'mf/v_mean': v.mean(),
            'mf/dudt_mean': dudt.mean(),
        }

    def latent_actor_loss(self, batch, grad_params, rng):
        observations = batch['observations']
        actions_gt = batch['actions'] # Dataset Actions (Ground Truth)
        batch_size = actions_gt.shape[0]
        latent_dim = self.config["horizon_length"] * self.config["action_dim"]
        rng, z_rng = jax.random.split(rng, 2)

        v_logits = self.network.select('critic')(observations)
        vs = self._logit_to_scalar(v_logits) # (B,)

        if self.config['num_critic'] > 1:
            if self.config['critic_agg'] == 'min':
                v = jnp.min(vs, axis=0)
            else:
                v = jnp.mean(vs, axis=0)
        else:
            v = vs
        
        # Q(s, a)
        qs = self._compute_scalar_target(batch)

        if self.config['num_critic'] > 1:
            if self.config['critic_agg'] == 'min':
                q = jnp.min(qs, axis=0)
            else:
                q = jnp.mean(qs, axis=0)
        else:
            q = qs

        adv = q.reshape(-1) - v # (B,)
        
        tau = self.config.get('tau', 1.0)
        weight = jnp.exp(adv / tau)
        weight = jnp.clip(weight, 0.0, 100.0)
        weight = jax.lax.stop_gradient(weight) # (B,)

        if self.config['noisy_actor']:
            z_in = jax.random.normal(z_rng, (batch_size, latent_dim))
            z_pred = self.network.select('latent_actor')(
                observations, 
                z_in,
                params=grad_params # <--- Gradients flow here
            )
        else:
            z_pred = self.network.select('latent_actor')(
                observations, 
                params=grad_params # <--- Gradients flow here
            )

        actions_gt_flat = jnp.reshape(actions_gt, (batch_size, latent_dim))
        mf_method = self.config['mf_method'] # 'mf' or 'mf_jit'
        
        # Generation: a_pred approx z - u(z, t=1, dt=1)
        t1 = jnp.ones((batch_size, 1))  # Noise time
        
        # Flow Forward (z -> a 방향)
        # Important: z_pred (Trainable) goes in here!
        x_pred = self.network.select('actor_bc_flow')(
            observations, 
            z_pred,     # Input is Predicted Z
            t1,         # Start at t=1
            t1,         # dt = 1
        )

        if mf_method == 'mf':
            a_pred_flat = z_pred - jnp.reshape(x_pred, (batch_size, latent_dim))
        elif mf_method == 'jit_mf':
            a_pred_flat = jnp.reshape(x_pred, (batch_size, latent_dim))
        
        # Loss in Action Space
        # Flow의 Jacobian을 타고 z_pred로 Gradient가 전달됨
        sq_err = jnp.mean(
            jnp.reshape(
                jnp.square(a_pred_flat - actions_gt_flat),
                (batch_size, self.config["horizon_length"], self.config["action_dim"])
            )
            , axis=(1, 2)
        )
        loss = jnp.mean(weight * sq_err)

        return loss, {
            'latent_loss': loss,
            'adv_mean': adv.mean(),
            'weight_mean': weight.mean(),
            'z_norm': jnp.mean(jnp.square(z_pred)),
        }

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

        if self.config['mf_method'] == 'jit_mf':
            actor_loss, actor_info = self.actor_loss_jit(batch, grad_params, actor_rng)
        elif self.config['mf_method'] == 'mf':
            actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        elif self.config['mf_method'] == 'imf':
            actor_loss, actor_info = self.improved_actor_loss(batch, grad_params, actor_rng)
        elif self.config['mf_method'] == 'jit_imf':
            actor_loss, actor_info = self.improved_actor_loss_jit(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        return actor_loss, info

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

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        if self.config['mf_method'] == 'jit_mf':
            actor_loss, actor_info = self.actor_loss_jit(batch, grad_params, actor_rng)
        elif self.config['mf_method'] == 'mf':
            actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        elif self.config['mf_method'] == 'imf':
            actor_loss, actor_info = self.improved_actor_loss(batch, grad_params, actor_rng)
        elif self.config['mf_method'] == 'jit_imf':
            actor_loss, actor_info = self.improved_actor_loss_jit(batch, grad_params, actor_rng)

        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        latent_loss, latent_info = self.latent_actor_loss(batch, grad_params, actor_rng)

        for k, v in latent_info.items():
            info[f'latent/{k}'] = v

        loss = critic_loss + actor_loss + latent_loss
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

    @staticmethod
    def _latent_actor_update(agent, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        old_latent_params = agent.network.params

        def loss_fn(grad_params):
            return agent.total_latent_actor_loss(batch, grad_params, rng=rng)
        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)

        new_params = new_network.params.unfreeze()
        for key in ['actor_bc_flow', 'critic', 'value']:
            new_params[f'modules_{key}'] = old_latent_params[f'modules_{key}']
        new_network = new_network.replace(params=flax.core.FrozenDict(new_params))

        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def critic_update(self, batch):
        return self._critic_update(self, batch)
    
    @jax.jit
    def actor_update(self, batch):
        return self._actor_update(self, batch)

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
        rng, z_rng = jax.random.split(rng, 2)
        action_dim = self.config["horizon_length"] * self.config["action_dim"]

        if self.config['noisy_actor']:
            z_in = jax.random.normal(z_rng, (action_dim))
            noises = self.network.select('latent_actor')(
                observations, 
                z_in,
            )
        else:
            noises = self.network.select('latent_actor')(
                observations, 
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
        elif self.config['mf_method'] == 'imf':
            action = noise - output
        elif self.config['mf_method'] == 'jit_imf':
            action = output
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
            encoders['actor_bc_flow'] = encoder_module()

        # Define networks.
        critic_def = QuantileValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_critic'],
            encoder=encoders.get('critic'),
            num_quantiles=config['num_bins']
        )

        actor_bc_flow_def = ActorMeanFlowField(
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
        )

        if config['noisy_actor']:
            latent_actor_input_shape = (ex_observations, full_actions)
        else:
            latent_actor_input_shape = (ex_observations)
        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, full_actions, ex_times, ex_times)),
            latent_actor=(latent_actor_def, latent_actor_input_shape),
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
            agent_name='hlmeanflow',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            latent_actor_hidden_dims=(512, 512),
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            critic_agg='min',  # Aggregation method for target Q values.
            num_critic=2, # critic ensemble size
            flow_steps=1,  # Number of flow steps.
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
            noisy_actor=False,
            alpha=1.0,
            cfg=1.0,
            cfg_dropout=0.1,
            sigma=0.75,
            flow_ratio=0.25,
            mf_method='jit_mf'
        )
    )
    return config
