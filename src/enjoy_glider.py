#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gym, gym_poine

import baselines.ppo2.ppo2 as ppo2

# python -m baselines.run --alg=ppo2 --env=glider-v0 --num_timesteps=0 --load_path=glider_model_ppo2.pkl --play

def main():
    env = gym.make("CartPole-v0")
    #env = gym.make("glider-v0")
    env.num_envs = 1
    #act = ppo2.learn(env, network='mlp', total_timesteps=0, load_path="glider_model_ppo2.pkl")
    act = ppo2.learn(
        env=env,
        network='mlp',
        total_timesteps = 5e6,
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
        save_interval=0, load_path="glider_model_ppo2.pkl", model_fn=None, update_fn=None, init_fn=None, mpi_rank_weight=1, comm=None,
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
