# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train an agent to solve the pheromone task.

Example command to run this script: `python train_pheromone.py --gpu-id=0`
"""

import sys
import argparse
import os
import shutil
import jax

from evojax.task.pheromone import Pheromone
from evojax.policy.mlp import MLPPolicy
from evojax.algo import PGPE
from evojax import Trainer
from evojax import util


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pop-size', type=int, default=256, help='ES population size.')
    parser.add_argument(
        '--hidden-size', type=int, default=100, help='Policy hidden size.')
    parser.add_argument(
        '--num-tests', type=int, default=100, help='Number of test rollouts.')
    parser.add_argument(
        '--n-repeats', type=int, default=32, help='Training repetitions.')
    parser.add_argument(
        '--max-iter', type=int, default=500, help='Max training iterations.')
    parser.add_argument(
        '--test-interval', type=int, default=50, help='Test interval.')
    parser.add_argument(
        '--log-interval', type=int, default=10, help='Logging interval.')
    parser.add_argument(
        '--seed', type=int, default=42, help='Random seed for training.')
    parser.add_argument(
        '--center-lr', type=float, default=0.014, help='Center learning rate.')
    parser.add_argument(
        '--std-lr', type=float, default=0.088, help='Std learning rate.')
    parser.add_argument(
        '--init-std', type=float, default=0.069, help='Initial std.')
    parser.add_argument(
        '--gpu-id', type=str, help='GPU(s) to use.')
    parser.add_argument(
        '--debug', action='store_true', help='Debug mode.')
    config, _ = parser.parse_known_args()
    return config


def main(config):
    log_dir = './log/pheromone'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(
        name='Pheromone', log_dir=log_dir, debug=config.debug)
    logger.info('EvoJAX Pheromone')
    logger.info('=' * 30)

    max_steps = 100
    num_agents = 10
    train_task = Pheromone(test=False, max_steps=max_steps, num_agents=num_agents)
    test_task = Pheromone(test=True, max_steps=max_steps, num_agents=num_agents)
    policy = MLPPolicy(
        input_dim=train_task.obs_shape[-1],
        hidden_dims=[5],
        output_dim=train_task.act_shape[-1],
        output_act_fn='softmax',
    )
    solver = PGPE(
        pop_size=config.pop_size,
        param_size=policy.num_params,
        optimizer='adam',
        center_learning_rate=config.center_lr,
        stdev_learning_rate=config.std_lr,
        init_stdev=config.init_std,
        logger=logger,
        seed=config.seed,
    )

    # Train.
    trainer = Trainer(
        policy=policy,
        solver=solver,
        train_task=train_task,
        test_task=test_task,
        max_iter=config.max_iter,
        log_interval=config.log_interval,
        test_interval=config.test_interval,
        n_repeats=config.n_repeats,
        n_evaluations=num_agents,
        seed=config.seed,
        log_dir=log_dir,
        logger=logger,
    )
    #trainer.run(demo_mode=False)

    # Test the final model.
    src_file = os.path.join(log_dir, 'best.npz')
    tar_file = os.path.join(log_dir, 'model.npz')
    shutil.copy(src_file, tar_file)
    trainer.model_dir = log_dir
    trainer.run(demo_mode=True)

    # Visualize the policy.
    task_reset_fn = jax.jit(test_task.reset)
    policy_reset_fn = jax.jit(policy.reset)
    step_fn = jax.jit(test_task.step)
    action_fn = jax.jit(policy.get_actions)
    best_params = trainer.solver.best_params[None, :]
    #key = jax.random.PRNGKey(0)[None, :]
    #seed = 0
    #key = jax.random.PRNGKey(seed)[None, :]
    #key, subkey = jax.random.split(key)
    task_state = test_task.reset()
    policy_state = policy.reset(task_state)
    screens = []

    for _ in range(max_steps):
        action, policy_state = action_fn(task_state, best_params, policy_state)
        task_states[i], reward, done = step_fn(task_state, action)
        screens.append(Pheromone.render(task_state))

    gif_file = os.path.join(log_dir, 'pheromone.gif')
    screens[0].save(
        gif_file, save_all=True, append_images=screens[1:], duration=40, loop=0)
    logger.info('GIF saved to {}.'.format(gif_file))

if __name__ == '__main__':
    configs = parse_args()
    if configs.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = configs.gpu_id
    main(configs)
