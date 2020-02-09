#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gym, gym_poine
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
import baselines.ppo2.ppo2 as ppo2
# python -m baselines.run --alg=ppo2 --env=Pendulum-v0 --num_timesteps=0 --load_path=pendulum_model_ppo2.pkl --play
def main():
    if 1:
        env = gym.make("Pendulum-v0")
        env.num_envs = 1
        act = ppo2.learn(env=env, network='mlp', total_timesteps=0, load_path="pendulum_model_ppo2.pkl")
    else:
        env_id = "pendulum-legacy-v0"
        env_type = "gym_poine"
        num_env = 1
        seed = 1234
        reward_scale = 1.
        flatten_dict_observations = False
        env = make_vec_env(env_id, env_type, num_env, seed, reward_scale, flatten_dict_observations)
       
        act = ppo2.learn(
            env=env,
            network='mlp',
            total_timesteps = 0,
            eval_env = None,
            seed=None, nsteps=2048, ent_coef=0.0,
            #lr=lambda f : f * 2.5e-4,
            lr=3e-4,
            vf_coef=0.5,
            max_grad_norm=0.5,
            gamma=0.9, # default 0.99
            lam=0.95,
            log_interval=10,
            nminibatches=32, # default 4
            noptepochs=10, cliprange=0.2,
            save_interval=0, load_path="pendulum_model_ppo2.pkl", model_fn=None, update_fn=None, init_fn=None, mpi_rank_weight=1, comm=None,
        )
 
    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
