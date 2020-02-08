#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gym#, gym_poine

from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
import baselines.logger
import baselines.ppo2.ppo2 as ppo2
import pdb
#pdb.set_trace()
# export OPENAI_LOGDIR=/tmp/blaa
# export OPENAI_LOG_FORMAT='stdout,tensorboard'
# python -m baselines.run --alg=ppo2 --env=Pendulum-v0 --num_timesteps=1e6 --save_path=pendulum_model_ppo2.pkl
# python -m baselines.run --alg=ppo2 --env=Pendulum-v0 --nminibatches=32 --noptepochs=10 --num_env=12 --num_timesteps=4e6 --save_path=pendulum_model_ppo2.pkl --play
# python -m baselines.run --alg=ppo2 --env=Pendulum-v0 --num_timesteps=0 --load_path=pendulum_model_ppo2.pkl --play
def main():
    baselines.logger.configure(dir='/tmp/pendulum_ppo2', format_strs=['stdout', 'log', 'csv', 'tensorboard'])
    #env = gym.make("pendulum-legacy-v0")
    env = gym.make("Pendulum-v0")
    env_id = "Pendulum-v0"
    env_type = "classic-control"
    num_env = 1
    seed = 1234
    reward_scale = 1.
    flatten_dict_observations = False
    env = make_vec_env(env_id, env_type, num_env, seed, reward_scale, flatten_dict_observations)
    
    # nsteps=128, nminibatches=4,
    #     lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
    #     ent_coef=.01,
    #     lr=lambda f : f * 2.5e-4,
    #     cliprange=0.1,

    act = ppo2.learn(
        env=env,
        network='mlp',
        total_timesteps = 3e6,
        eval_env = None, seed=seed, nsteps=2048, ent_coef=0.0,
        lr=lambda f : f * 2.5e-4,
        #lr=3e-4,
        vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
        log_interval=10, nminibatches=4,
        noptepochs=10, cliprange=0.2,
        save_interval=0, load_path=None, model_fn=None, update_fn=None, init_fn=None, mpi_rank_weight=1, comm=None,
    )
    print("Saving model to pendulum_model_ppo2.pkl")
    act.save("pendulum_model_ppo2.pkl")
    

if __name__ == '__main__':
    main()
