import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.meanflow_utils import adaptive_l2_loss, sample_latent_dist
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorMeanFlowField
from utils.dit import MFDiT_REAL

# Helper functions for Consistency Flow
def f_euler(t_vals, segment_ends, xt, vt):
    """
    Euler step to predict x at segment_ends from xt using velocity vt.
    Formula: x_end = x_t + (t_end - t) * v_t
    """
    return xt + (segment_ends - t_vals) * vt

class ConsistencyFlowAgent(flax.struct.PyTreeNode):
    """
    Consistency Flow Matching Agent implemented in JAX.
    Adapts the logic from FlowPolicy (PyTorch) to the MEANFLOWAgent structure.
    """
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def actor_loss(self, batch, grad_params, rng):
        """Compute the Consistency Flow actor loss."""
        # 1. Prepare Data
        # Flatten action: (B, T, D) -> (B, T*D)
        batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1)) 
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng, r_rng = jax.random.split(rng, 4)

        target = batch_actions  # x_1 (Data)
        
        # 2. Sample Noise (x_0)
        # Assuming latent_dist is 'normal' or similar (Noise)
        e = sample_latent_dist(x_rng, (batch_size, action_dim), self.config['latent_dist'])
        
        # 3. Time Sampling (t and r)
        # Note: Following PyTorch code convention: t=1 is Data, t=0 is Noise.
        eps = self.config['eps']
        delta = self.config['delta']
        
        # t ~ U[eps, 1]
        t = jax.random.uniform(t_rng, (batch_size, 1), minval=eps, maxval=1.0)
        
        # r = clamp(t + delta, max=1.0)
        # We use a slightly perturbed time for consistency constraint
        r = jnp.clip(t + delta, max=1.0)

        # 4. Interpolate State (Forward Process)
        # xt = t * x_1 + (1-t) * x_0 (Rectified Flow / Straight Path)
        xt = t * target + (1 - t) * e
        xr = r * target + (1 - r) * e
        # Masking logic if needed (skipped for pure action generation, but kept generic)
        # In pure action generation, we don't usually mask 'obs' inside 'x', 
        # so xt and xr are just actions.

        # 5. Determine Segments
        num_segments = self.config['num_segments']
        # Define segment boundaries: [0, ..., 1]
        segments = jnp.linspace(0, 1, num_segments + 1)
        
        # Find which segment 't' belongs to. We want the END of the segment.
        # indices: where t would be inserted to maintain order.
        # e.g., if segments=[0, 0.5, 1] and t=0.2, index=1 (0.5)
        seg_indices = jnp.searchsorted(segments, t, side='left')
        # Clamp to ensure we don't access out of bounds, though t <= 1 ensures index <= num_segments
        # We want the value at segments[index].
        # If t=1.0, searchsorted returns num_segments+1 or similar? 
        # jnp.searchsorted for 1.0 in [0, 1] returns index 2 (len). 
        # We clamp index to be at most num_segments (the last element).
        seg_indices = jnp.clip(seg_indices, 1, num_segments)
        
        segment_ends = segments[seg_indices] # (B, 1)

        # 6. Network Forward (Velocity Prediction)
        # Model predicts v(x, t). 
        # We pass t directly. The network should handle embedding.
        
        def consistency_forward(z, time_val):
             return self.network.select('actor_bc_flow')(
                batch['observations'], 
                z, 
                time_val, 
                time_val, # Passing t as r-cond if needed, or ignored by network
                params=grad_params
            )
        
        def consistency_target_forward(z, time_val):
             return self.network.select('target_actor_bc_flow')(
                batch['observations'], 
                z, 
                time_val, 
                time_val, # Passing t as r-cond if needed, or ignored by network
            )

        vt = consistency_forward(xt, t)
        vr = consistency_target_forward(xr, r)
        # 7. Consistency Targets (Euler Step to Segment End)
        # x_ref_t = x_t + (t_ref - t) * v_t
        ft = f_euler(t, segment_ends, xt, vt)
        
        # x_ref_r needs special handling for boundary conditions (if r crosses boundary)
        # But PyTorch code simplifies this by checking if t < boundary.
        # Here we implement the 'threshold_based_f_euler' logic simplified.
        boundary_threshold = self.config['boundary'] # Usually 1.0 or 0
        
        # Calculate x at segment ends using GROUND TRUTH interpolation
        # This acts as an anchor: if we are close to boundary, we should match GT.
        x_at_segment_ends = segment_ends * target + (1 - segment_ends) * e
        
        fr_pred = f_euler(r, segment_ends, xr, vr)
        
        # Logic: If t < threshold, we use prediction fr. Else (t >= threshold), we anchor to GT.
        # But looking at PyTorch: `less_than_threshold = t_expand < threshold`
        # res = less * pred + (~less) * gt
        # Usually boundary=1, so t < 1 is almost always true, so we use prediction.
        use_prediction = t < boundary_threshold
        fr = jnp.where(use_prediction, fr_pred, x_at_segment_ends)

        # 8. Compute Losses
        # A) Consistency Loss: Predictions from t and r should match at segment end
        loss_consistency = jnp.square(ft - jax.lax.stop_gradient(fr))
        loss_consistency = jnp.mean(loss_consistency) # Mean over batch/dim

        # B) Velocity Smoothness Loss
        # Only penalize if we are NOT too close to the segment end
        # PyTorch: (segment_ends - t) > 1.01 * delta
        far_from_end = (segment_ends - t) > (1.01 * delta)
        loss_velocity = jnp.square(vt - jax.lax.stop_gradient(vr))
        
        # Apply masks
        loss_velocity = loss_velocity * use_prediction * far_from_end
        loss_velocity = jnp.mean(loss_velocity)

        alpha = self.config['alpha']
        total_loss = loss_consistency + alpha * loss_velocity

        return total_loss, {
            'actor_loss': total_loss,
            'loss_cons': loss_consistency,
            'loss_vel': loss_velocity
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        rng, actor_rng = jax.random.split(rng, 2)
        loss, info = self.actor_loss(batch, grad_params, actor_rng)
        return loss, info

    @staticmethod
    def _update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)
        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)
        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'actor_bc_flow')
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)
    
    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    # ========= Inference (Sampling) ============
    @jax.jit
    def sample_actions(self, observations, rng=None):
        """
        Sample actions using Consistency Sampling (Segment-wise).
        Flow: t=0 (Noise) -> t=1 (Data)
        """
        rng, x_rng = jax.random.split(rng, 2)
        
        # Dimensions
        horizon = self.config["horizon_length"]
        action_dim = self.config["action_dim"]
        flat_dim = horizon * action_dim
        batch_size = observations.shape[0]

        # 1. Initialize with Noise (t=0)
        z = sample_latent_dist(x_rng, (batch_size, flat_dim), self.config['latent_dist'])
        
        # If encoder is present, encode observations once
        obs_encoded = observations
        if self.config['encoder'] is not None:
             # Assuming we can call this part of the network separately or it's handled inside
             # Ideally, we call the full network and it handles encoding.
             # For efficiency, we might want to cache it, but let's keep it simple.
             pass

        # 2. Define Segments for Sampling
        # We go from 0 to 1.
        num_segments = self.config['num_inference_steps'] # Often same as num_segments or 1
        # PyTorch code uses `sde.sample_N` which corresponds to inference steps.
        # If num_inference_steps == 1, we go 0 -> 1 directly.
        
        # Create time grid: [0, ..., 1]
        # We iterate: start -> end
        # In consistency models, we predict the END of the segment from the START.
        
        # NOTE: The loop in PyTorch code:
        # for i in range(sample_N):
        #    num_t = i / sample_N ... (start time)
        #    sigma_t ... (noise injection if needed)
        #    pred = model(z, t) ...
        #    z = ... Euler update ...
        
        # But wait, true Consistency Models predict x_start directly (or x_end).
        # The PyTorch code implements "Consistency Flow Matching" where the model outputs velocity v.
        # So we simply integrate v over the segments.
        
        def loop_body(i, val):
            z_curr = val
            
            # Current time t
            t_curr = i / num_segments
            
            # Target time t_next
            t_next = (i + 1) / num_segments
            
            # Prepare inputs
            t_batch = jnp.ones((batch_size, 1)) * t_curr
            
            # Predict velocity at current time
            v_pred = self.network.select('actor_bc_flow')(
                observations, z_curr, t_batch, t_batch
            )
            
            # Euler Step: z_next = z_curr + (t_next - t_curr) * v_pred
            dt = t_next - t_curr
            z_next = z_curr + dt * v_pred
            
            # Optional: Noise injection (Langevin) could be added here if sigma > 0
            # but standard Consistency FM usually is deterministic during sampling.
            
            return z_next

        z_final = jax.lax.fori_loop(0, num_segments, loop_body, z)
        
        # Clip final result
        actions = jnp.clip(z_final, -1, 1)
        
        # Reshape to (B, T, D)
        actions = jnp.reshape(actions, (batch_size, horizon, action_dim))
        
        return actions
    
    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a new agent."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        # Dimension setup
        ex_times = jnp.zeros((ex_observations.shape[0], 1))
        action_dim = ex_actions.shape[-1]
        action_len = ex_actions.shape[1]
        full_actions = jnp.reshape(ex_actions, (ex_actions.shape[0], -1))
        full_action_dim = full_actions.shape[-1]

        # Encoders
        encoders = dict()
        if config['encoder'] is not None:
            from utils.encoders import encoder_modules # Delayed import
            encoder_module = encoder_modules[config['encoder']]
            encoders['actor_bc_flow'] = encoder_module()

        # Network Definition (DiT or MLP)
        if config['use_DiT']:
            actor_bc_flow_def = MFDiT_REAL(
                hidden_dim=config.get('dit_hidden_dim', 128),
                depth=config.get('dit_depth', 3),
                num_heads=config.get('dit_heads', 2),
                output_dim=action_dim,  
                output_len=action_len,
                use_r=True, # Network needs to accept r input (even if we pass t)
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

        # Inputs for init: (obs, action, t, r)
        actor_input_shape = (ex_observations, full_actions, ex_times, ex_times)

        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, actor_input_shape),
            target_actor_bc_flow=(copy.deepcopy(actor_bc_flow_def), actor_input_shape),
        )

        if encoders.get('actor_bc_flow') is not None:
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
        params[f'modules_target_actor_bc_flow'] = params[f'modules_actor_bc_flow']

        config['ob_dims'] = ex_observations.shape[-1:]
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='cfm',
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=False,
            encoder=ml_collections.config_dict.placeholder(str),
            horizon_length=ml_collections.config_dict.placeholder(int),
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.,
            latent_dist='normal',
            use_DiT=False,
            mf_method='mfql',
            tau=0.005,
            # Consistency specific params
            eps=1e-2,           # Min time t
            delta=1e-2,         # Time perturbation for r = t + delta
            num_segments=2,     # Number of training segments
            alpha=1e-5,         # Weight for velocity smoothness loss
            boundary=1.0,       # Boundary time (usually 1.0)
            
            # Inference
            num_inference_steps=1, # Number of steps during sampling (1 or 2 is common)
        )
    )
    return config