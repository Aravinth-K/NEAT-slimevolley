import numpy as np
import gym
from matplotlib.pyplot import imread


def make_env(env_name, seed=-1, render_mode=False):

  # -- Cart Pole Swing up -------------------------------------------- -- #
  if (env_name.startswith("CartPoleSwingUp")):
    from domain.cartpole_swingup import CartPoleSwingUpEnv
    env = CartPoleSwingUpEnv()
    if (env_name.startswith("CartPoleSwingUp_Hard")):
      env.dt = 0.01
      env.t_limit = 200

  # -- Slime Volley Ball -------------------------------------------- -- #
  elif (env_name.startswith("Slime")):
    if env_name.startswith("SlimeVolley"):
      from slimevolleygym import SlimeVolleyEnv
      env = SlimeVolleyEnv()
    elif env_name.startswith("SlimeSurvival"):
      from slimevolleygym import SlimeVolleyEnv, SurvivalRewardEnv
      base_env = SlimeVolleyEnv()
      env = SurvivalRewardEnv(base_env)

  # -- Other  -------------------------------------------------------- -- #
  else:
    env = gym.make(env_name)

  if (seed >= 0):
    domain.seed(seed)

  return env