#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gym, gym_poine

#from baselines import ddpg
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
import baselines.logger
import baselines.ddpg.ddpg as ddpg
import pdb
#pdb.set_trace()

# python -m baselines.run --alg=ddpg --env=Pendulum-v0 --num_timesteps=1e5 --play  #  --save_path=pendulum_model_ddpg.pkl
# python -m baselines.run --alg=ppo2 --env=Pendulum-v0 --num_timesteps=0 --load_path=pendulum_model_ddpg.pkl --play

def main():
    baselines.logger.configure(dir='/tmp/pendulum_ddpg', format_strs=['stdout', 'log', 'csv', 'tensorboard'])
    #env = gym.make("pendulum-legacy-v0")
    #env = gym.make("Pendulum-v0")
    env_id = "Pendulum-v0"
    env_type = "classic_control"
    num_env = 1
    seed = 1234
    reward_scale = 1.
    flatten_dict_observations = False
    env = make_vec_env(env_id, env_type, num_env, seed, reward_scale, flatten_dict_observations)

    act = ddpg.learn(
        env=env,
        network='mlp',
        seed=seed,
        nb_epoch_cycles=20,
        nb_rollout_steps=100,
        reward_scale=1.0,
        render=False,#True,
        render_eval=False,
        noise_type='adaptive-param_0.2',
        normalize_returns=False,
        normalize_observations=True,
        critic_l2_reg=1e-2,
        actor_lr=1e-4,
        critic_lr=1e-3,
        popart=False,
        gamma=0.99,
        clip_norm=None,
        nb_train_steps=50, # per epoch cycle and MPI worker,
        nb_eval_steps=100,
        batch_size=64, # per MPI worker
        tau=0.01,
        eval_env=None,
        param_noise_adaption_interval=50
    )
    print("Saving model to pendulum_model_ddpg.pkl")
    act.save("pendulum_model_ddpg.pkl")
    

if __name__ == '__main__':
    main()
