from collections import namedtuple
import numpy as np

Game = namedtuple('Game', ['env_name', 'time_factor', 'actionSelect',
  'input_size', 'output_size', 'layers', 'i_act', 'h_act',
  'o_act', 'weightCap','noise_bias','output_noise','max_episode_length','in_out_labels'])

games = {}


# -- Cart-pole Swingup --------------------------------------------------- -- #

# > Slower reaction speed
cartpole_swingup = Game(env_name='CartPoleSwingUp_Hard',
  actionSelect='weighted', # all, soft, hard
  input_size=5,
  output_size=1,
  time_factor=0,
  layers=[5, 5],
  i_act=np.full(5,1),
  h_act=[1,2,3,4,5,6,7,8,9,10],
  o_act=np.full(1,1),
  weightCap = 2.0,
  noise_bias=0.0,
  output_noise=[False, False, False],
  max_episode_length = 200,
  in_out_labels = ['x','x_dot','cos(theta)','sin(theta)','theta_dot',
                   'force']
)
games['swingup_hard'] = cartpole_swingup

# > Normal reaction speed
cartpole_swingup = cartpole_swingup._replace(\
    env_name='CartPoleSwingUp', max_episode_length=1000)
games['swingup'] = cartpole_swingup


# -- Slime Volley Ball --------------------------------------------------- -- #
slimevolley = Game(env_name='SlimeVolley-v0',
  actionSelect='tanh', # all, soft, hard
  input_size=12,
  output_size=3,
  time_factor=0,
  layers=[32, 9],
  i_act=np.full(12,1),
  h_act=[1,2,3,4,5,6,7,8,9,10],
  o_act=np.full(3,1),
  weightCap = 2.0,
  noise_bias=0.0,
  output_noise=[False, False, False],
  max_episode_length = 3000,
  in_out_labels = ['x', 'y', 'vx', 'vy',
                    'bx', 'by', 'bvx', 'bvy',
                    'ox', 'oy', 'ovx', 'ovy']
)
games['slimevolley'] = slimevolley

slimesurvival = slimevolley._replace(env_name='SlimeSurvival-v0')
games['slimesurvival'] = slimesurvival
