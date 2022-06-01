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

"""Implementation of the pheromone task in JAX.

The pheromone task is a model of artificial life that references the behavior of ants.
https://github.com/alifelab/alife_book_src/blob/master/alifebook_lib/simulators/ant_simulator.py
"""

from typing import Tuple
from functools import partial
from PIL import Image
from PIL import ImageDraw
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import random
from jax import tree_util
from flax.struct import dataclass

from evojax.task.base import TaskState
from evojax.task.base import VectorizedTask

SCREEN_W = 600
SCREEN_H = 600
ANT_RADIUS = 20
SENSOR_RADIUS = 2
MIN_DIST = 2 * ANT_RADIUS
NUM_RANGE_SENSORS = 7
DELTA_ANG = 2 * 3.14 / NUM_RANGE_SENSORS

TYPE_VOID = 0
TYPE_WALL = 1
TYPE_FOOD = 2
TYPE_POISON = 3
TYPE_AGENT = 4

@dataclass
class AntStatus(object):
    pos_x: jnp.float32
    pos_y: jnp.float32
    angle: jnp.float32
    vel: jnp.float32
    ang_vel: jnp.float32

@dataclass
class State(TaskState):
    agent_state: AntStatus
    field: jnp.ndarray
    obs: jnp.ndarray
    steps: jnp.int32
    key: jnp.ndarray


#@partial(jax.vmap, in_axes=(0, None))
def create_ants(key: jnp.ndarray) -> AntStatus:
    k_pos_x, k_pos_y, k_angle = random.split(key, 3)
    vel = ang_vel = 0.
    return AntStatus(
        pos_x=random.uniform(
            k_pos_x, shape=(), dtype=jnp.float32,
            minval=MIN_DIST, maxval=SCREEN_W - MIN_DIST),
        pos_y=random.uniform(
            k_pos_y, shape=(), dtype=jnp.float32,
            minval=MIN_DIST, maxval=SCREEN_H - MIN_DIST),
        angle=random.uniform(
            k_angle, shape=(), dtype=jnp.float32),
        vel=vel, ang_vel=ang_vel)


def get_reward(field: jnp.ndarray, agent: AntStatus) -> jnp.float32:
    x = agent.pos_x.astype(jnp.int32)
    y = agent.pos_y.astype(jnp.int32)
    reward = field[y, x]
    return reward

def update_agent_state(agent: AntStatus, action) -> AntStatus:
    vel = agent.vel - action[0] * 0.1
    ang_vel = agent.ang_vel + action[1] * 0.001

    angle = agent.angle + ang_vel
    pos_x = agent.pos_x + vel * jnp.cos(angle)
    pos_y = agent.pos_y + vel * jnp.sin(angle)

    vel *= 0.95
    ang_vel *= 0.95

    return AntStatus(pos_x=pos_x, pos_y=pos_y, angle=angle, vel=vel, ang_vel=ang_vel)

def update_field(field: jnp.ndarray, agent: AntStatus) -> jnp.ndarray:
    x = agent.pos_x.astype(jnp.int32)
    y = agent.pos_y.astype(jnp.int32)
    field = field.at[y, x].set(1.0)
    return field

def get_obs(agent: AntStatus) -> jnp.ndarray:
    x = agent.pos_x
    y = agent.pos_y
    theta = agent.angle
    return jnp.array([x, y, theta])

class Pheromone(VectorizedTask):
    def __init__(self,
                 max_steps: int = 1000,
                 test: bool = False):

        self.max_steps = max_steps
        self.test = test
        self.obs_shape = tuple([3, ])
        self.act_shape = tuple([2, ])

        def reset_fn(key):
            next_key, key = random.split(key)
            agent = create_ants(key)
            field = jnp.zeros((SCREEN_H, SCREEN_W), dtype=jnp.float32)
            obs = get_obs(agent)
            return State(agent_state=agent, field=field, obs=obs,
                         steps=jnp.zeros((), dtype=jnp.int32), key=next_key)
        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action):
            next_key, _ = random.split(state.key)
            agent = update_agent_state(state.agent_state, action)
            field = update_field(state.field, agent)
            reward = get_reward(field, agent)
            steps = state.steps + 1
            done = jnp.where(steps >= max_steps, 1, 0)
            obs = get_obs(agent)
            return State(agent_state=agent, field=field, obs=obs,
                         steps=steps, key=next_key), reward, done
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.array) -> State:
        return self._reset_fn(key)

    def step(self,
             state: State,
             action: jnp.ndarray) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)

    @staticmethod
    def render(state: State, task_id: int = 0) -> Image:
        # Draw background
        cmap = plt.get_cmap("viridis")
        img = cmap(state.field, bytes=True)
        img = Image.fromarray((img[0]).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        state = tree_util.tree_map(lambda s: s[task_id], state)

        # Draw the agent
        agent = state.agent_state
        x, y, angle = agent.pos_x, agent.pos_y, agent.angle
        x_top = x - ANT_RADIUS * jnp.cos(angle)
        y_top = y - ANT_RADIUS * jnp.sin(angle)
        sensor_data = jnp.array(state.obs)
        draw.ellipse(
            (x - ANT_RADIUS, y - ANT_RADIUS,
             x + ANT_RADIUS, y + ANT_RADIUS),
             outline=(255, 0, 0))
        draw.line((x, y, x_top, y_top), fill=(255, 0, 0), width=1)
        draw.ellipse((x - SENSOR_RADIUS, y - SENSOR_RADIUS,
                      x + SENSOR_RADIUS, y + SENSOR_RADIUS),
                      fill=(0, 0, 0), outline=(0, 0, 0))
        for i in range(NUM_RANGE_SENSORS):
            ang = i * DELTA_ANG + agent.angle
            x_sensor = x + ANT_RADIUS * np.cos(ang)
            y_sensor = y + ANT_RADIUS * np.sin(ang)
            draw.ellipse(
                (x_sensor - SENSOR_RADIUS, y_sensor - SENSOR_RADIUS,
                 x_sensor + SENSOR_RADIUS, y_sensor + SENSOR_RADIUS),
                 fill=(0, 255, 0))
        return img
