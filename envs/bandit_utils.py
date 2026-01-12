import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
import io
from utils.datasets import Dataset
import wandb
from sklearn.decomposition import PCA  # [New] PCA for visualization
# ==========================================
# 1. Data Generators (Manifolds) - Deterministic Version
# ==========================================

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
import io
from utils.datasets import Dataset
import wandb
from sklearn.decomposition import PCA  # [New] PCA for visualization
# ==========================================
# 1. Data Generators (Manifolds) - Deterministic Version
# ==========================================

def _get_bandit6_centers(rng=None):
    """
    50차원 공간에 9개의 클러스터 중심을 고정적으로 생성합니다.
    첫 번째 중심(Index 0)을 Optimal Target으로 설정합니다.
    """
    # 시드가 달라도 항상 같은 위치에 중심을 만들기 위해 고정된 시드 사용
    local_rng = np.random.RandomState(42) 
    
    # 50차원, 9개 클러스터
    # [-0.8, 0.8] 범위 내에 랜덤하게 배치
    centers = local_rng.uniform(-0.8, 0.8, size=(9, 50))
    return centers

def generate_bandit_1(n, rng):
    # Lv 1: 4-Gaussians
    centers = [(-0.7, -0.7), (-0.7, 0.7), (0.7, -0.7), (0.7, 0.7)]
    data = []
    for _ in range(n):
        c = centers[rng.randint(4)]
        data.append(np.array(c) + 0.1 * rng.randn(2))
    return np.array(data).astype(np.float32)

def generate_bandit_2(n, rng):
    # Lv 2: Checkerboard
    data = []
    while len(data) < n:
        x, y = rng.uniform(-1, 1, 2)
        if (np.floor(x * 2) % 2 + np.floor(y * 2) % 2) % 2 == 0:
             data.append([x, y])
    return np.array(data[:n]).astype(np.float32)

def generate_bandit_3(n, rng):
    # Lv 3: Two Moons (Hard) - Corrected
    # 정확한 Interleaving Moons 구현
    n_upper = n // 2
    n_lower = n - n_upper
    
    # 1. Upper Moon (t: 0 ~ pi) -> 중심 (0, 0)
    t_upper = np.linspace(0, np.pi, n_upper)
    x_upper = np.cos(t_upper)
    y_upper = np.sin(t_upper)
    upper_moon = np.stack([x_upper, y_upper], axis=1)
    
    # 2. Lower Moon (t: 0 ~ pi) -> 중심 (1, -0.5)
    t_lower = np.linspace(0, np.pi, n_lower)
    x_lower = 1 - np.cos(t_lower)
    y_lower = 1 - np.sin(t_lower) - 0.5
    lower_moon = np.stack([x_lower, y_lower], axis=1)
    
    # 3. Combine
    data = np.vstack([upper_moon, lower_moon])
    
    # 4. Normalize to [-1, 1]
    # 원래 범위: x[-1, 2], y[-0.5, 1] -> 중심 (0.5, 0.25)
    # 중심을 0으로 이동하고 스케일링
    data[:, 0] -= 0.5
    data[:, 1] -= 0.25
    data *= 0.6 # [-1, 1] 안쪽으로 안전하게 들어오도록 축소
    
    # 5. Add Noise
    data += 0.05 * rng.randn(*data.shape)
    
    return np.clip(data, -1.0, 1.0).astype(np.float32)

def generate_bandit_4(n, rng):
    # Lv 4: Rings
    noisy_circles = []
    for _ in range(n):
        radius = 0.8 if rng.rand() > 0.5 else 0.3
        angle = rng.rand() * 2 * np.pi
        noise = 0.05 * rng.randn(2)
        point = np.array([np.cos(angle) * radius, np.sin(angle) * radius]) + noise
        noisy_circles.append(point)
    return np.array(noisy_circles).astype(np.float32)

def generate_bandit_5(n, rng):
    # Lv 5: Spiral
    t = np.sqrt(rng.uniform(0, 1, n)) * 540 * (2 * np.pi) / 360
    x = -np.cos(t) * t + rng.rand(n) * 0.05
    y = np.sin(t) * t + rng.rand(n) * 0.05
    data = np.vstack([x, y]).T
    data = data / (np.max(np.abs(data)) + 1e-6)
    return data.astype(np.float32)

def generate_bandit_6(n, rng):
    """
    Lv 6: High-Dim Clusters (Action Chunking Sim)
    - Dim: 50
    - Clusters: 9 (Mixture of Gaussians)
    - Reward: Center 0 근처만 1.0, 나머지는 0.0
    """
    centers = _get_bandit6_centers()
    n_clusters = len(centers)
    
    data = []
    for _ in range(n):
        # 9개 클러스터 중 하나 선택 (Uniform)
        # 즉, 데이터셋에는 정답(1/9)과 오답(8/9)이 섞여 있음
        c_idx = rng.randint(n_clusters)
        center = centers[c_idx]
        
        # Noise (Cluster Spread)
        # 50차원이므로 노이즈가 너무 크면 겹칠 수 있음. 작게 설정.
        noise = 0.05 * rng.randn(50)
        data.append(center + noise)
        
    return np.array(data).astype(np.float32)

# 이름 매핑
GENERATORS = {
    'bandit-1': generate_bandit_1,
    'bandit-2': generate_bandit_2,
    'bandit-3': generate_bandit_3,
    'bandit-4': generate_bandit_4,
    'bandit-5': generate_bandit_5,
    'bandit-6': generate_bandit_6,
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
    Multi-modal Reward with Hard Peaks.
    Ensures Optimal Regions get exactly 1.0 reward.
    """
    actions = np.atleast_2d(actions)
    rewards = np.zeros(len(actions), dtype=np.float32)
    
    x = actions[:, 0]
    y = actions[:, 1]

    if env_name == 'bandit-1': # 4-Gaussians
        # Centers
        c1 = np.array([0.7, 0.7])   # Optimal
        c2 = np.array([0.7, -0.7])  # Trap 1
        c3 = np.array([-0.7, 0.7])  # Trap 2
        c4 = np.array([-0.7, -0.7]) # Trap 3
        
        # Distances
        d1 = np.linalg.norm(actions - c1, axis=1)
        d2 = np.linalg.norm(actions - c2, axis=1)
        d3 = np.linalg.norm(actions - c3, axis=1)
        d4 = np.linalg.norm(actions - c4, axis=1)
        
        # Thresholding (반지름 0.3 이내)
        # 겹칠 일은 거의 없지만 우선순위 적용
        rewards[d4 < 0.3] = 0.2
        rewards[d3 < 0.3] = 0.4
        rewards[d2 < 0.3] = 0.7
        rewards[d1 < 0.3] = 1.0 # Optimal은 무조건 1.0

    elif env_name == 'bandit-2': # Checkerboard
        # Check Pattern
        in_checker = (np.floor(x * 2) % 2 + np.floor(y * 2) % 2) % 2 == 0
        
        # 1. Optimal Zone: 우상단 박스 (0.5~1.0, 0.5~1.0)
        is_optimal = (x > 0.5) & (x < 1.0) & (y > 0.5) & (y < 1.0)
        
        # 2. Sub-optimal: 나머지 체스판
        is_subopt = in_checker & (~is_optimal)
        
        rewards[is_subopt] = 0.5
        rewards[is_optimal] = 1.0

    elif env_name == 'bandit-3': # Two Moons
        # 1. Upper Moon Tip (Optimal)
        t_tip = np.linspace(0, np.pi * 0.2, 50)
        tip_x = (np.cos(t_tip) - 0.5) * 0.6
        tip_y = (np.sin(t_tip) - 0.25) * 0.6
        tip_arc = np.stack([tip_x, tip_y], axis=1)
        dist_tip = np.min(np.linalg.norm(actions[:, None, :] - tip_arc[None, :, :], axis=2), axis=1)
        
        # 2. Rest of Upper Moon (Sub-optimal)
        t_up = np.linspace(np.pi * 0.2, np.pi, 100)
        up_x = (np.cos(t_up) - 0.5) * 0.6
        up_y = (np.sin(t_up) - 0.25) * 0.6
        up_arc = np.stack([up_x, up_y], axis=1)
        dist_up = np.min(np.linalg.norm(actions[:, None, :] - up_arc[None, :, :], axis=2), axis=1)

        # 3. Lower Moon (Bad Trap)
        t_low = np.linspace(0, np.pi, 100)
        low_x = (1 - np.cos(t_low) - 0.5) * 0.6
        low_y = (1 - np.sin(t_low) - 0.5 - 0.25) * 0.6
        low_arc = np.stack([low_x, low_y], axis=1)
        dist_low = np.min(np.linalg.norm(actions[:, None, :] - low_arc[None, :, :], axis=2), axis=1)
        
        # Assign Rewards (Threshold 0.1)
        rewards[dist_low < 0.1] = 0.2
        rewards[dist_up < 0.1] = 0.6
        rewards[dist_tip < 0.1] = 1.0 # Tip은 무조건 1.0

    elif env_name == 'bandit-4': # Rings
        r = np.linalg.norm(actions, axis=1)
        theta = np.arctan2(actions[:, 1], actions[:, 0])
        
        # Conditions
        is_inner = (r > 0.2) & (r < 0.4) # Inner Ring (r~0.3)
        is_outer = (r > 0.7) & (r < 0.9) # Outer Ring (r~0.8)
        is_quad1 = (theta > np.pi*2/6) & (theta < np.pi*3/6) # 1사분면
        
        rewards[is_inner] = 0.3
        rewards[is_outer] = 0.6 # 일단 전체 Outer는 0.6
        rewards[is_outer & is_quad1] = 1.0 # 1사분면 Outer만 1.0

    elif env_name == 'bandit-5': # Spiral
        # 1. Optimal Tip (t: 2.5 ~ 3.0)
        t_tip = np.linspace(2.5, 3.0, 100)
        tx = t_tip * np.cos(3 * t_tip) * 0.3
        ty = t_tip * np.sin(3 * t_tip) * 0.3
        tip_arc = np.stack([tx, ty], axis=1)
        dist_tip = np.min(np.linalg.norm(actions[:, None, :] - tip_arc[None, :, :], axis=2), axis=1)
        
        # 2. Body (t: 0.5 ~ 2.5)
        t_body = np.linspace(0.5, 2.5, 200)
        bx = t_body * np.cos(3 * t_body) * 0.3
        by = t_body * np.sin(3 * t_body) * 0.3
        body_arc = np.stack([bx, by], axis=1)
        dist_body = np.min(np.linalg.norm(actions[:, None, :] - body_arc[None, :, :], axis=2), axis=1)
        
        # Spiral 궤적 위에 있으면 점수 부여 (Continuous Gradient 느낌)
        # 하지만 1.0은 Tip에만 부여
        rewards[dist_body < 0.1] = 0.5 # 몸통은 0.5
        rewards[dist_tip < 0.1] = 1.0  # 끝은 1.0

    elif env_name == 'bandit-6': # High-Dim Clusters
        centers = _get_bandit6_centers()
        dists = np.linalg.norm(actions[:, None, :] - centers[None, :, :], axis=2) # (N, 9)
        min_dists = np.min(dists, axis=1)
        closest_idx = np.argmin(dists, axis=1)
        
        # Threshold 0.3 이내인 경우에만 점수 부여
        valid_mask = min_dists < 0.3
        
        # 기본 점수 (Noise)
        scores = np.full(len(actions), 0.0, dtype=np.float32)
        
        # 클러스터별 점수 매핑
        # 0번: 1.0 (Optimal)
        # 1번: 0.8
        # 나머지: 0.1
        cluster_scores = np.full(9, 0.1, dtype=np.float32)
        cluster_scores[0] = 1.0
        cluster_scores[1] = 0.8
        
        # 할당
        # valid한 애들만 점수 줌
        assigned_scores = cluster_scores[closest_idx]
        rewards = np.where(valid_mask, assigned_scores, 0.0)

    return rewards


class ToyBanditEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, env_name, seed=0, render_mode=None):
        super().__init__()
        self.env_name = env_name
        if env_name == 'bandit-6':
            self.action_dim = 50
            self.obs_dim = 2
        else:
            self.action_dim = 2
            self.obs_dim = 2
        
        self.observation_space = spaces.Box(-1, 1, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(self.action_dim,), dtype=np.float32)
        
        self.render_mode = render_mode
        self.fig, self.ax = None, None
        
        # 배치 시각화를 위해 리스트나 배열로 저장
        self.last_actions = None
        self.last_rewards = None
        
        rng = np.random.RandomState(seed)
        self.gt_data = GENERATORS[env_name](2000, rng)

        # 2. [New] PCA Initialization (Fixed Mapping)
        self.pca = None
        if self.action_dim > 2:
            # GT 데이터를 기준으로 PCA 축을 학습(Fit)하고 고정합니다.
            # 이렇게 하면 나중에 Agent가 생성한 데이터도 이 '고정된 축'으로 투영됩니다.
            self.pca = PCA(n_components=2)
            self.pca.fit(self.gt_data)
            
            # GT 데이터도 2D로 변환해 둡니다 (시각화용)
            self.gt_data_2d = self.pca.transform(self.gt_data)
        else:
            self.gt_data_2d = self.gt_data

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
        # 1. 스타일 및 캔버스 설정
        plt.style.use('default')
        
        # 3개의 서브플롯 생성 (가로로 긴 형태)
        # if self.fig is None:
            # 1행 3열, 전체 크기 (18, 6) -> 각 플롯당 (6, 6)
        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
        
        # 배경색: 아이보리
        bg_color = '#FFFDF5'
        self.fig.patch.set_facecolor(bg_color)
        for ax in self.axes:
            ax.set_facecolor(bg_color)
                
        # 매번 그리기 위해 클리어
        for ax in self.axes:
            ax.clear()
            ax.axis('off')
            
        # 줌 설정 (공통)
        limit = 1.2 if not self.pca else 2.5
        for ax in self.axes:
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)

        # 타이틀 설정
        self.axes[0].set_title("Dataset Density (Offline Data)", fontsize=14, fontweight='bold', color='#8B8000')
        # self.axes[1].set_title("Reward Landscape (Ground Truth)", fontsize=14, fontweight='bold', color='#800080')
        self.axes[1].set_title("Current Policy (Agent)", fontsize=14, fontweight='bold', color='#00008B')

        # 필요한 도구 임포트
        from scipy.stats import gaussian_kde
        import matplotlib.cm as cm

        # 공통 그리드 생성 (등고선용)
        grid_res = 100
        x_grid = np.linspace(-limit, limit, grid_res)
        y_grid = np.linspace(-limit, limit, grid_res)
        gx, gy = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([gx.ravel(), gy.ravel()]) # (2, N)

        # ==========================================
        # Plot 1: Dataset Density (Yellow ~ Orange)
        # ==========================================
        ax_data = self.axes[0]
        
        # GT Data가 있으면 KDE 계산
        if self.gt_data_2d is not None:
            try:
                # KDE 계산 (Dataset은 고정되어 있으므로 사실 캐싱하면 더 빠르지만, 여기선 매번 계산)
                # 데이터가 많으면 subsample을 쓰는 것이 좋음 (여기선 2000개라 가정)
                data_kde = gaussian_kde(self.gt_data_2d.T)
                z_data = data_kde(positions).reshape(grid_res, grid_res)
                z_data_norm = z_data / (z_data.max() + 1e-8)
                
                # YlOrBr: Yellow -> Orange -> Brown
                # 0.1 이하는 투명하게
                ax_data.contourf(gx, gy, z_data_norm, levels=np.linspace(0.1, 1.0, 12), cmap='YlOrBr', alpha=0.8)
                # ax_data.contour(gx, gy, z_data_norm, levels=[0.1], colors=['#B8860B'], linewidths=0.5)
                
            except Exception:
                # KDE 실패 시 그냥 점으로 찍기
                ax_data.scatter(self.gt_data_2d[:, 0], self.gt_data_2d[:, 1], c='orange', s=5, alpha=0.3)

        # ==========================================
        # Plot 2: Reward Landscape (Purple ~ Red) + Colorbar
        # ==========================================
        # ax_reward = self.axes[1]
        
        # if self.action_dim == 2:
        #     # 2D 환경: 실제 보상 함수 계산
        #     points = positions.T # (N, 2)
        #     rewards = get_reward_batch(self.env_name, points).reshape(grid_res, grid_res)
            
        #     # PuRd: Purple -> Red (낮음 -> 높음)
        #     # vmin=0, vmax=1.2로 고정하여 색상 일관성 유지
        #     # levels를 0.0부터 시작하여 전체 배경을 칠해줌 (Landscape 느낌)
        #     cf = ax_reward.contourf(gx, gy, rewards, levels=np.linspace(0.0, 1.0, 12), cmap='PuRd', alpha=0.8)
            
        #     # Colorbar 추가 (Plot 내부에 작게 넣거나 오른쪽에 붙임)
        #     # 여기서는 Plot 오른쪽에 깔끔하게 붙임
        #     cbar = self.fig.colorbar(cf, ax=ax_reward, fraction=0.046, pad=0.04)
        #     cbar.ax.tick_params(labelsize=8)
        #     cbar.outline.set_visible(False)
            
        # else:
        #     # High-Dim (Bandit-6): PCA 공간에서는 전체 Reward Grid를 그릴 수 없음
        #     # 대신 Cluster Center들의 보상을 시각화
        #     centers = _get_bandit6_centers()
        #     centers_2d = self.pca.transform(centers)
            
        #     # 각 센터의 보상값 계산
        #     center_rewards = get_reward_batch(self.env_name, centers) # (9,)
            
        #     # Scatter로 표현하되 색상을 Reward에 매핑
        #     sc = ax_reward.scatter(centers_2d[:, 0], centers_2d[:, 1], c=center_rewards, 
        #                            cmap='PuRd', s=500, edgecolors='black', vmin=0, vmax=1.0)
            
        #     cbar = self.fig.colorbar(sc, ax=ax_reward, fraction=0.046, pad=0.04)
        #     cbar.ax.tick_params(labelsize=8)

        # ==========================================
        # Plot 3: Current Action Density (Blue)
        # ==========================================
        ax_policy = self.axes[1]
        
        # Ground Truth Context (옅게 깔기) - 위치 파악용
        # if self.action_dim == 2:
        #      # Reward 2D Map을 옅은 회색으로 바닥에 깔아줌
        #      points_bg = np.vstack([gx.ravel(), gy.ravel()]).T
        #      rewards_bg = get_reward_batch(self.env_name, points_bg).reshape(grid_res, grid_res)
        #      ax_policy.contourf(gx, gy, rewards_bg, levels=[0.5, 2.0], colors=['#E0E0E0'], alpha=0.3)

        # Action Density (KDE)
        if self.last_actions is not None:
            actions = np.atleast_2d(self.last_actions)
            if self.pca:
                actions_2d = self.pca.transform(actions)
            else:
                actions_2d = actions
            
            try:
                kde = gaussian_kde(actions_2d.T)
                z_act = kde(positions).reshape(grid_res, grid_res)
                z_act_norm = z_act / (z_act.max() + 1e-8)
                
                # Blues: Light Blue -> Dark Blue
                ax_policy.contourf(gx, gy, z_act_norm, levels=np.linspace(0.1, 1.0, 10), cmap='Blues', alpha=0.8)
                # ax_policy.contour(gx, gy, z_act_norm, levels=[0.1], colors=['darkblue'], linewidths=0.5, alpha=0.5)
                
            except Exception:
                pass
            
            # 최근 샘플 점찍기 (확인용)
            # ax_policy.scatter(actions_2d[:, 0], actions_2d[:, 1], c='black', s=5, alpha=0.2)

        # ==========================================
        # 캡처 로직 (buffer_rgba)
        # ==========================================
        self.fig.canvas.draw()
        img_rgba = np.asarray(self.fig.canvas.buffer_rgba())
        img_arr = np.array(img_rgba[:, :, :3], dtype=np.uint8)
            
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
    # [수정] RandomState 객체 생성 (Local RNG)
    rng = np.random.RandomState(seed)
    
    # 1. Actions 생성 (Manifold Generator 사용)
    actions = GENERATORS[env_name](dataset_size, rng)
    
    # 2. Rewards 계산 (Vectorized)
    # [수정] 리스트 컴프리헨션 제거 -> 배치 함수 사용
    rewards = get_reward_batch(env_name, actions)

    # 3. Dummy fields (Offline RL 포맷 맞추기)
    # Observation은 의미 없으므로 0으로 채움
    observations = np.zeros((dataset_size, 2), dtype=np.float32)
    next_observations = np.zeros_like(observations)
    
    # Bandit이므로 매 스텝 종료 (Terminal=1, Mask=0)
    terminals = np.zeros(dataset_size, dtype=np.float32)
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

    sr = (rewards == 1.0).sum() / (rewards > -1).sum()
    
    print(f"Dataset ({env_name}) Created: {dataset_size} samples")
    print(f"SR - : {sr*100:.2f}%")
    
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
    actions = actor_fn(observations=observations)
    actions = np.array(actions).reshape(1000, -1)
    _, rewards, _, _, _ = env.step(actions)

    # 4. Calculate Metrics
    success_rate = np.sum(rewards==1.0) / np.ones_like(rewards).sum() # Reward는 0 or 1
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

