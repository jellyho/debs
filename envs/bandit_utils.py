import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
import io
from utils.datasets import Dataset

# ==========================================
# 1. Data Generators (Manifolds)
# ==========================================

def generate_bandit_1(n):
    # Lv 1: 4-Gaussians (Easy)
    centers = [(-0.7, -0.7), (-0.7, 0.7), (0.7, -0.7), (0.7, 0.7)]
    data = []
    for _ in range(n):
        c = centers[np.random.randint(4)]
        data.append(np.array(c) + 0.1 * np.random.randn(2))
    return np.array(data).astype(np.float32)

def generate_bandit_2(n):
    # Lv 2: Checkerboard (Medium)
    data = []
    while len(data) < n:
        x, y = np.random.uniform(-1, 1, 2)
        if (np.floor(x * 2) % 2 + np.floor(y * 2) % 2) % 2 == 0:
             data.append([x, y])
    return np.array(data[:n]).astype(np.float32)

def generate_bandit_3(n):
    # Lv 3: Two Moons (Hard)
    n_out = n // 2
    n_in = n - n_out
    outer_x = np.cos(np.linspace(0, np.pi, n_out))
    outer_y = np.sin(np.linspace(0, np.pi, n_out))
    inner_x = 1 - np.cos(np.linspace(0, np.pi, n_in))
    inner_y = 1 - np.sin(np.linspace(0, np.pi, n_in)) - 0.5
    X = np.vstack([np.append(outer_x, inner_x), np.append(outer_y, inner_y)]).T
    X = X * 0.8 - 0.4
    X += 0.05 * np.random.randn(*X.shape)
    return np.clip(X, -1, 1).astype(np.float32)

def generate_bandit_4(n):
    # Lv 4: Rings (Very Hard - Topological hole)
    noisy_circles = []
    for _ in range(n):
        radius = 0.8 if np.random.rand() > 0.5 else 0.3
        angle = np.random.rand() * 2 * np.pi
        noise = 0.05 * np.random.randn(2)
        point = np.array([np.cos(angle) * radius, np.sin(angle) * radius]) + noise
        noisy_circles.append(point)
    return np.array(noisy_circles).astype(np.float32)

def generate_bandit_5(n):
    # Lv 5: Spiral (Extreme)
    t = np.sqrt(np.random.uniform(0, 1, n)) * 540 * (2 * np.pi) / 360
    x = -np.cos(t) * t + np.random.rand(n) * 0.05
    y = np.sin(t) * t + np.random.rand(n) * 0.05
    data = np.vstack([x, y]).T
    data = data / (np.max(np.abs(data)) + 1e-6)
    return data.astype(np.float32)

# 이름 매핑
GENERATORS = {
    'bandit-1': generate_bandit_1,
    'bandit-2': generate_bandit_2,
    'bandit-3': generate_bandit_3,
    'bandit-4': generate_bandit_4,
    'bandit-5': generate_bandit_5,
}

# ==========================================
# 2. Reward Functions
# ==========================================

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
import io

# 보상 함수 (벡터화 지원)
def get_reward_batch(env_name, actions):
    """
    actions: (N, 2) or (2,)
    returns: (N,) or scalar
    """
    actions = np.atleast_2d(actions)
    x = actions[:, 0]
    y = actions[:, 1]
    rewards = np.zeros(len(actions), dtype=np.float32)

    if env_name == 'bandit-1': # 4-Gaussians (Top-Right)
        # Dist to (0.7, 0.7) < 0.2
        dists = np.linalg.norm(actions - np.array([[0.7, 0.7]]), axis=1)
        rewards = (dists < 0.2).astype(np.float32)

    elif env_name == 'bandit-2': # Checkerboard (Center-Right-Top)
        # 0.5 < x < 1.0 AND 0.5 < y < 1.0
        mask = (x > 0.5) & (x < 1.0) & (y > 0.5) & (y < 1.0)
        rewards = mask.astype(np.float32)

    elif env_name == 'bandit-3': # Two Moons (Upper Moon Right tip)
        # x > 0.5 AND y > 0.0 (Simplified region for moon tip)
        mask = (x > 0.5) & (y > 0.0)
        rewards = mask.astype(np.float32)

    elif env_name == 'bandit-4': # Rings (Outer Ring 1 o'clock)
        r = np.linalg.norm(actions, axis=1)
        # 0.7 < r < 0.9 AND x > 0 AND y > 0
        mask = (r > 0.7) & (r < 0.9) & (x > 0) & (y > 0)
        rewards = mask.astype(np.float32)

    elif env_name == 'bandit-5': # Spiral (Tail)
        r = np.linalg.norm(actions, axis=1)
        # r > 0.8 AND x > 0
        mask = (r > 0.8) & (x > 0)
        rewards = mask.astype(np.float32)
        
    return rewards


class ToyBanditEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, env_name, seed=0, render_mode=None):
        super().__init__()
        self.env_name = env_name
        self.action_dim = 2
        self.obs_dim = 2 
        
        self.observation_space = spaces.Box(-1, 1, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(self.action_dim,), dtype=np.float32)
        
        self.render_mode = render_mode
        self.fig, self.ax = None, None
        
        # 배치 시각화를 위해 리스트나 배열로 저장
        self.last_actions = None
        self.last_rewards = None
        
        # GT Data for visualization (배경용)
        self.gt_data = GENERATORS[env_name](2000)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.obs_dim, dtype=np.float32), {}

    def step(self, action):
        # 1. Action Clipping & Batch Handling
        # 입력이 (2,) 일 수도 있고 (Batch, 2) 일 수도 있음
        action = np.array(action)
        is_batch = action.ndim > 1
        
        # Clip
        action = np.clip(action, -1.0, 1.0)
        
        # 2. Reward Calculation (Vectorized)
        # 배치 처리가 가능한 get_reward_batch 사용
        reward = get_reward_batch(self.env_name, action)
        
        # 3. Save for Rendering
        # 나중에 그릴 때 쓰기 위해 저장
        self.last_actions = action
        self.last_rewards = reward
        
        # 4. Return Construction
        # One-step termination
        # 배치 크기에 맞춰서 리턴
        if is_batch:
            batch_size = len(action)
            obs = np.zeros((batch_size, self.obs_dim), dtype=np.float32)
            terminated = np.ones(batch_size, dtype=bool)
            truncated = np.zeros(batch_size, dtype=bool)
            return obs, reward.astype(float), terminated, truncated, {}
        else:
            return np.zeros(self.obs_dim, dtype=np.float32), float(reward), True, False, {}

    def render(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(5, 5))
            
        self.ax.clear()
        self.ax.set_xlim(-1.1, 1.1)
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_title(f"{self.env_name} (Last Batch)")
        
        # 1. Background: Data Manifold (Ground Truth)
        self.ax.scatter(self.gt_data[:, 0], self.gt_data[:, 1], c='gray', s=5, alpha=0.1, label='GT Data')
        
        # 2. Background: Reward Region (Goal)
        grid = np.linspace(-1, 1, 50)
        gx, gy = np.meshgrid(grid, grid)
        gpoints = np.vstack([gx.ravel(), gy.ravel()]).T
        grewards = get_reward_batch(self.env_name, gpoints)
        if grewards.sum() > 0:
             self.ax.scatter(gpoints[grewards>0, 0], gpoints[grewards>0, 1], c='green', s=5, alpha=0.1, label='Goal')

        # 3. Foreground: Agent Actions (Batch Scatter)
        if self.last_actions is not None:
            # 배치 차원 보정
            actions = np.atleast_2d(self.last_actions)
            rewards = np.atleast_1d(self.last_rewards)
            
            # 보상에 따라 색상 구분 (성공: 빨강, 실패: 파랑)
            # matplotlib scatter는 색상 배열을 받을 수 있음
            colors = np.where(rewards > 0, 'red', 'blue')
            
            self.ax.scatter(
                actions[:, 0], actions[:, 1], 
                c=colors, 
                s=30,       # 점 크기
                marker='x', # 마커 모양
                alpha=0.6,  # 투명도
                label='Agent'
            )

        # self.ax.legend(loc='upper right')

        io_buf = io.BytesIO()
        self.fig.savefig(io_buf, format='raw', dpi=100)
        io_buf.seek(0)
        img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                                newshape=(int(self.fig.bbox.bounds[3]), int(self.fig.bbox.bounds[2]), -1))
        io_buf.close()
        return img_arr
            
    def close(self):
        if self.fig: plt.close(self.fig)

def make_bandit_datasets(env_name, dataset_size=100000, seed=0):
    """
    Toy Bandit 데이터셋 생성 함수.
    get_reward_batch를 사용하여 고속으로 보상을 계산합니다.
    """
    # 환경 생성 (메타데이터 및 스페이스 확인용)  
    
    # 1. Actions 생성 (Manifold Generator 사용)
    # Shape: (dataset_size, 2)
    actions = GENERATORS[env_name](dataset_size)
    
    # 2. Rewards 계산 (Vectorized)
    # [수정] 리스트 컴프리헨션 제거 -> 배치 함수 사용
    rewards = get_reward_batch(env_name, actions)

    # 3. Dummy fields (Offline RL 포맷 맞추기)
    # Observation은 의미 없으므로 0으로 채움
    observations = np.zeros((dataset_size, 2), dtype=np.float32)
    next_observations = np.zeros_like(observations)
    
    # Bandit이므로 매 스텝 종료 (Terminal=1, Mask=0)
    terminals = np.ones(dataset_size, dtype=np.float32)
    masks = np.zeros(dataset_size, dtype=np.float32)
    
    data_dict = {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'next_observations': next_observations,
        'terminals': terminals,
        'masks': masks
    }
    
    # Dataset 객체 반환
    dataset = Dataset.create(**data_dict)
    
    print(f"Dataset ({env_name}) Created: {dataset_size} samples")
    print(f" - Success Rate: {rewards.mean()*100:.2f}%")
    
    # main.py의 구조에 맞춰 env, eval_env, train_dataset, val_dataset(None) 반환
    return dataset


import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, rng=key, **kwargs)

    return wrapped

def evaluate(
    agent,
    env,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
    action_shape=None,
    observation_shape=None,
    action_dim=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))

    observation, info = env.reset()
    observations = np.zeros((1000, 2), dtype=np.float32)
    action = actor_fn(observations=observation)
    action = np.array(action).reshape(1000, -1)
    _, rewards, _, _, _ = env.step(actions)

    # 4. Calculate Metrics
    success_rate = np.mean(rewards) # Reward는 0 or 1
    avg_reward = np.mean(rewards)
    
    stats = {
        "eval/success_rate": success_rate,
        "eval/avg_reward": avg_reward,
    }
    img_array = env.render()

    if img_array is not None:
        # WandB Image 로깅 (채널 순서: HWC -> CHW 필요 없음, wandb.Image가 알아서 함)
        # 다만 보통 PyTorch/Tensorboard 스타일인 CHW로 변환해서 넣기도 함.
        # 여기서는 HWC 그대로 넣어도 됩니다.
        stats["eval/policy_distribution"] = wandb.Image(
            img_array, 
            caption=f"Success Rate {success_rate*100:.1f}%"
        )

    return stats, _, _

