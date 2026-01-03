import os
# 환경변수가 파이썬 내부에서도 잘 잡혔는지 확인 (스크립트에서 설정 안했을 경우)
# os.environ['MUJOCO_GL'] = 'egl'

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

import robosuite

env = robosuite.make('Lift', robots='Panda')
obs = env.reset()
print(obs['agentview_image'].shape)
