import dataclasses
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
from tqdm import tqdm

## Dataset code from DEAS
def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)

class Dataset(FrozenDict):
    """Dataset class."""

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.frame_stack = None  # Number of frames to stack; set outside the class.
        self.p_aug = None  # Image augmentation probability; set outside the class.
        self.return_next_actions = False  # Whether to additionally return next actions; set outside the class.
        self.brc_normalize_rewards = False  # Whether to normalize rewards according to brc paper.
        self.v_max = None  # Maximum value of the reward; set outside the class.
        self.actor_action_sequence = 1  # Actor action sequence; set outside the class. Used for outputting action sequence.
        self.critic_action_sequence = 1  # Critic action sequence; set outside the class. Used for options framework.
        self.nstep = 1  # Number of steps for n-step return; set outside the class.
        self.discount = 1.0  # Discount factor; set outside the class.
        self.discount2 = 1.0  # Discount factor for actor; set outside the class.

        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        if len(self.terminal_locs) == 0:
            self.terminal_locs = np.array([self.size - 1])
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

        self._raw_rtgs = None
        self._valid_indices = None
        self._stats = None

    def update_locs(self):
        self._valid_indices = None
        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        if len(self.terminal_locs) == 0:
            self.terminal_locs = np.array([self.size - 1])
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    @property
    def valid_indices(self):
        """Get valid indices."""
        if self._valid_indices is None:
            valid_indices = []
            for start, end in zip(self.initial_locs, np.append(self.terminal_locs, self.size)):
                traj_len = end - start + 1
                valid_end = start + max(0, traj_len - (self.actor_action_sequence * self.nstep - 1))
                if valid_end > start:
                    valid_indices.extend(range(start, valid_end))
            self._valid_indices = np.array(valid_indices)
        return self._valid_indices

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        valid_indices = self.valid_indices
        assert len(valid_indices) > 0, 'No valid indices found'

        # Sample from valid indices
        rand_idx = np.random.randint(len(valid_indices), size=num_idxs)
        return valid_indices[rand_idx]

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)

        if (self.nstep > 1 or self.critic_action_sequence > 1) and self._dict.get('rewards') is not None:
            # Find next episode start for each idx to avoid crossing episode boundaries
            search_idxs = np.searchsorted(self.terminal_locs, idxs)
            # Clip search_idxs to valid range before indexing terminal_locs
            clipped_search_idxs = np.minimum(search_idxs, len(self.terminal_locs) - 1)
            next_episode_starts = np.where(
                search_idxs < len(self.terminal_locs), self.terminal_locs[clipped_search_idxs], self.size - 1
            )

            # Double loop case - outer nstep, inner action_sequence
            outer_indices = np.arange(self.nstep)[None, :, None]  # [1, nstep, 1]
            inner_indices = np.arange(self.critic_action_sequence)[None, None, :]  # [1, 1, action_sequence]
            idxs_expanded = np.expand_dims(idxs, (1, 2))  # [batch, 1, 1]

            # Combined indices for both loops [batch, nstep, action_sequence]
            seq_indices = idxs_expanded + (outer_indices * self.critic_action_sequence) + inner_indices
            seq_indices = np.minimum(seq_indices, np.expand_dims(next_episode_starts, (1, 2)))

            # Get rewards and masks for all steps
            rewards = self._dict['rewards'][seq_indices]  # [batch, nstep, action_sequence]
            masks = self._dict['masks'][seq_indices]  # [batch, nstep, action_sequence]

            # First compute inner action_sequence discounts
            inner_discount_powers = self.discount ** np.arange(self.critic_action_sequence)[None, None, :]
            inner_returns = np.sum(rewards * inner_discount_powers, axis=2)  # [batch, nstep]

            # Then compute outer nstep discounts
            outer_discount_powers = self.discount2 ** np.arange(self.nstep)[None, :]
            batch['rewards'] = np.sum(inner_returns * outer_discount_powers, axis=1)  # [batch]

            # Compute combined masks
            batch['masks'] = np.prod(np.prod(masks, axis=2), axis=1)
            total_steps = self.nstep * self.critic_action_sequence

            # Get final next observations
            next_obs_idxs = np.minimum(idxs + total_steps - 1, next_episode_starts)
            batch['next_observations'] = jax.tree_util.tree_map(
                lambda arr: arr[next_obs_idxs], 
                self._dict['next_observations']
            )
            # batch['next_observations'] = self._dict['next_observations'][
            #     np.minimum(idxs + total_steps - 1, next_episode_starts)
            # ]

        batch['valid'] = np.ones_like(batch['masks'])

        if self.frame_stack is not None:
            # Stack frames.
            initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
            obs = []  # Will be [ob[t - frame_stack + 1], ..., ob[t]].
            next_obs = []  # Will be [ob[t - frame_stack + 1 + N], ..., ob[t + N]].

            # Get current observation stack
            for i in reversed(range(self.frame_stack)):
                # Use the initial state if the index is out of bounds.
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))

            # Get next observation stack, shifted by nstep
            for i in reversed(range(self.frame_stack)):
                # Use the initial state if the index is out of bounds.
                # next_idxs = np.maximum(idxs + (self.nstep * self.action_sequence - 1) - i, initial_state_idxs)
                next_idxs = np.maximum(idxs + (self.nstep * self.critic_action_sequence - 1) - i, initial_state_idxs)
                next_obs.append(jax.tree_util.tree_map(lambda arr: arr[next_idxs], self['observations']))

            batch['observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *obs)
            batch['next_observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *next_obs)
        if self.p_aug is not None:
            # Apply random-crop image augmentation.
            if np.random.rand() < self.p_aug:
                self.augment(batch, ['observations', 'next_observations'])
        return batch

    def _get_action_sequences(self, idxs):
        # Find next episode start for each idx to avoid crossing episode boundaries
        search_idxs = np.searchsorted(self.terminal_locs, idxs)
        # Clip search_idxs to valid range before indexing terminal_locs
        clipped_search_idxs = np.minimum(search_idxs, len(self.terminal_locs) - 1)
        next_episode_starts = np.where(
            search_idxs < len(self.terminal_locs), self.terminal_locs[clipped_search_idxs], self.size - 1
        )

        # Create sequence indices for each idx
        seq_indices = np.expand_dims(idxs, 1) + np.arange(self.actor_action_sequence)[None, :]

        # Clip sequence indices to stay within episodes
        seq_indices = np.minimum(seq_indices, np.expand_dims(next_episode_starts, 1))

        # Get action sequences
        action_sequences = self._dict['actions'][seq_indices]
        return action_sequences

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
            # WARNING: This is incorrect at the end of the trajectory. Use with caution.
            result['next_actions'] = self._dict['actions'][np.minimum(idxs + 1, self.size - 1)]

        result['actions'] = self._get_action_sequences(idxs)
        if self.return_next_actions:
            result['next_actions'] = self._get_action_sequences(np.minimum(idxs + 1, self.size - 1))
        return result

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = get_size(batch[keys[0]])
        # batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition."""
        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset."""
        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """
        Add a transition to the replay buffer.
        Assumes: No circular buffer (infinite size), simply appends data.
        """
        
        # 1. 초기화: valid_indices가 없으면 생성 (첫 실행 시)
        if self._valid_indices is None:
             _ = self.valid_indices  # 프로퍼티 호출해서 초기화

        # 2. 데이터 저장 (In-place update)
        # 꽉 차지 않는다고 가정하므로 덮어쓰기 걱정 없이 현재 포인터에 넣습니다.
        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element
        
        jax.tree_util.tree_map(set_idx, self._dict, transition)

        # 3. Terminal 정보 업데이트
        # 덮어쓰기가 없으므로 '기존 데이터가 터미널이었는지' 체크할 필요 없음.
        # 이번에 들어온 데이터가 터미널인지만 보면 됩니다.
        if transition['terminals'] > 0:
            # 순서대로 쌓이므로 그냥 append 하면 정렬 상태 유지됨
            self.terminal_locs = np.append(self.terminal_locs, self.pointer)
            
            # initial_locs 업데이트 (다음 에피소드의 시작점 미리 등록)
            # 현재 pointer가 터미널이면, 다음 데이터(pointer+1)는 새 에피소드 시작임
            self.initial_locs = np.append(self.initial_locs, self.pointer + 1)

        # 4. Valid Indices 업데이트 (핵심: Incremental Update)
        # 현재 추가된 데이터(self.pointer)로 인해 새로운 유효 시퀀스가 생겼는지 확인
        
        # 시퀀스 길이 조건
        seq_len = self.actor_action_sequence * self.nstep
        
        # 현재 에피소드의 시작점 찾기
        # initial_locs의 마지막 원소가 현재 진행 중인 에피소드의 시작점입니다.
        # (위에서 terminal일 경우 pointer+1을 추가했으므로, 그 전까진 마지막 원소가 시작점)
        current_ep_start = self.initial_locs[-1]
        
        # 만약 방금 terminal을 추가해서 initial_locs가 갱신되었다면?
        # -> 방금 끝난 에피소드의 시작점은 initial_locs[-2]가 됩니다.
        if transition['terminals'] > 0:
             current_ep_start = self.initial_locs[-2]
        
        # 현재 에피소드의 길이 계산 (start ~ current pointer)
        current_ep_len = self.pointer - current_ep_start + 1
        
        # 길이가 조건보다 길거나 같다면, 유효 인덱스 추가
        if current_ep_len >= seq_len:
            # 새로 유효해진 인덱스: (현재 위치) - (시퀀스 길이) + 1
            # 예: 길이 3 필요. 0,1,2 들어옴(idx 2). 2-3+1 = 0번 인덱스가 유효해짐.
            new_valid_idx = self.pointer - seq_len + 1
            
            # 단순히 끝에 추가 (Append)
            self._valid_indices = np.append(self._valid_indices, new_valid_idx)

        # 5. 포인터 이동 (단순 증가)
        self.pointer += 1
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0
        # locs 정보도 초기화
        self.terminal_locs = np.array([], dtype=int)
        self.initial_locs = np.array([0], dtype=int)