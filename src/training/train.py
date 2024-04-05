# Copyright 2023 The Pgx Authors. All Rights Reserved.
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

import datetime
import os
import pickle
import time
from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import mctx
import optax
import pgx
import wandb
from flax.training import train_state
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from pgx.experimental import auto_reset
from pydantic import BaseModel

from src.architectures.azresnet import AZResnet, AZResnetConfig

devices = jax.local_devices()
num_devices = len(devices)
print("Number of devices:", num_devices)

class Config(BaseModel):
    env_id: pgx.EnvId = "bughouse"
    seed: int = 0
    max_num_iters: int = 400
    # network params
    num_channels: int = 256
    num_layers: int = 15
    # selfplay params
    selfplay_batch_size: int = 16
    num_simulations: int = 32
    max_num_steps: int = 256
    # training params
    training_batch_size: int = 16
    learning_rate: float = 0.001
    # eval params
    eval_interval: int = 5

    class Config:
        extra = "forbid"


conf_dict = OmegaConf.from_cli()
config: Config = Config(**conf_dict)

class TrainState(train_state.TrainState):
    batch_stats: chex.ArrayTree


env = pgx.make(config.env_id)

optimizer = optax.adam(learning_rate=config.learning_rate)
net = AZResnet(
    AZResnetConfig(
        num_blocks=15,
        channels=256,
        policy_channels=4,
        value_channels=8,
        num_policy_labels=2*64*78+1,
    )
)

def recurrent_fn(model_state, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
    del rng_key

    current_player = state.current_player
    state = jax.vmap(env.step)(state, action)

    model_params = {'params': model_state.params, 'batch_stats': model_state.batch_stats}
    logits, value = model_state.apply_fn(
        model_params, state.observation, train=False
    )

    # mask invalid actions
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
    value = jnp.where(state.terminated, 0.0, value)
    discount = -1.0 * jnp.ones_like(value)
    discount = jnp.where(state.terminated, 0.0, discount)

    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=value,
    )
    return recurrent_fn_output, state


class SelfplayOutput(NamedTuple):
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray


@jax.pmap
def selfplay(model_state, rng_key: jnp.ndarray) -> SelfplayOutput:
    batch_size = config.selfplay_batch_size // num_devices
    model_params = {'params': model_state.params, 'batch_stats': model_state.batch_stats}

    def step_fn(state, key) -> SelfplayOutput:
        key1, key2 = jax.random.split(key)
        observation = state.observation

        logits, value = model_state.apply_fn(
            model_params, state.observation, train=False
        )

        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

        policy_output = mctx.gumbel_muzero_policy(
            params=model_state,
            rng_key=key1,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            max_num_considered_actions=512,
            gumbel_scale=1.0,
        )
        actor = state.current_player
        keys = jax.random.split(key2, batch_size)
        state = jax.vmap(auto_reset(env.step, env.init))(state, policy_output.action, keys)
        discount = -1.0 * jnp.ones_like(value)
        discount = jnp.where(state.terminated, 0.0, discount)
        return state, SelfplayOutput(
            obs=observation,
            action_weights=policy_output.action_weights,
            reward=state.rewards[jnp.arange(state.rewards.shape[0]), actor],
            terminated=state.terminated,
            discount=discount,
        )

    # Run selfplay for max_num_steps by batch
    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    state = jax.vmap(env.init)(keys)
    key_seq = jax.random.split(rng_key, config.max_num_steps)
    _, data = jax.lax.scan(step_fn, state, key_seq)

    return data


class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray


@jax.pmap
def compute_loss_input(data: SelfplayOutput) -> Sample:
    batch_size = config.selfplay_batch_size // num_devices
    # If episode is truncated, there is no value target
    # So when we compute value loss, we need to mask it
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1

    # Compute value target
    def body_fn(carry, i):
        ix = config.max_num_steps - i - 1
        v = data.reward[ix] + data.discount[ix] * carry
        return v, v

    _, value_tgt = jax.lax.scan(
        body_fn,
        jnp.zeros(batch_size),
        jnp.arange(config.max_num_steps),
    )
    value_tgt = value_tgt[::-1, :]

    return Sample(
        obs=data.obs,
        policy_tgt=data.action_weights,
        value_tgt=value_tgt,
        mask=value_mask,
    )


@partial(jax.pmap, axis_name="i")
def train(model_state, data: Sample):
    def calculate_loss(model_params, batch_stats, samples: Sample):
        (logits, value), new_model_state = model_state.apply_fn(
            {'params': model_params, 'batch_stats': batch_stats},
            samples.obs, train=True, mutable=['batch_stats']
        )

        policy_loss = optax.softmax_cross_entropy(logits, samples.policy_tgt)
        policy_loss = jnp.mean(policy_loss)

        value_loss = optax.l2_loss(value, samples.value_tgt)
        value_loss = jnp.mean(value_loss * samples.mask)  # mask if the episode is truncated

        return policy_loss + value_loss, (new_model_state, policy_loss, value_loss)
    
    loss_fn = lambda params: calculate_loss(
        params, model_state.batch_stats, data
    )
    (loss, (new_model_state, policy_loss, value_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        model_state.params
    )
    grads = jax.lax.pmean(grads, axis_name="i")
    model_state = model_state.apply_gradients(
        grads=grads, batch_stats=new_model_state['batch_stats']
    )
    return model_state, policy_loss, value_loss


if __name__ == "__main__":
    #wandb.init(project="pgx-az", config=config.model_dump())

    # Initialize model and opt_state
    dummy_state = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 2))
    dummy_input = dummy_state.observation
    
    model_parameters = net.init(jax.random.key(0), dummy_input, train=True)
    model_state = TrainState.create(
        apply_fn=net.apply,
        params=model_parameters['params'],
        batch_stats=model_parameters['batch_stats'],
        tx=optimizer,
    )

    # replicates to all devices
    model_state  = jax.device_put_replicated(model_state, devices)

    # Prepare checkpoint dir
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    now = now.strftime("%Y%m%d%H%M%S")
    ckpt_dir = os.path.join("checkpoints", f"{config.env_id}_{now}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Initialize logging dict
    iteration: int = 0
    hours: float = 0.0
    frames: int = 0
    log = {"iteration": iteration, "hours": hours, "frames": frames}

    rng_key = jax.random.PRNGKey(config.seed)
    while True:
        '''if iteration % config.eval_interval == 0:
            # Evaluation
            rng_key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, num_devices)

            # Store checkpoints
            model_0, opt_state_0 = jax.tree_util.tree_map(lambda x: x[0], (model, opt_state))
            with open(os.path.join(ckpt_dir, f"{iteration:06d}.ckpt"), "wb") as f:
                dic = {
                    "config": config,
                    "rng_key": rng_key,
                    "model": jax.device_get(model_0),
                    "opt_state": jax.device_get(opt_state_0),
                    "iteration": iteration,
                    "frames": frames,
                    "hours": hours,
                    "pgx.__version__": pgx.__version__,
                    "env_id": env.id,
                    "env_version": env.version,
                }
                pickle.dump(dic, f)'''

        print(log)
        #wandb.log(log)

        if iteration >= config.max_num_iters:
            break

        iteration += 1
        log = {"iteration": iteration}
        st = time.time()

        # Selfplay
        rng_key, subkey = jax.random.split(rng_key)
        keys = jax.random.split(subkey, num_devices)
        data: SelfplayOutput = selfplay(model_state, keys)
        samples: Sample = compute_loss_input(data)

        # Shuffle samples and make minibatches
        samples = jax.device_get(samples)  # (#devices, batch, max_num_steps, ...)
        frames += samples.obs.shape[0] * samples.obs.shape[1] * samples.obs.shape[2]
        samples = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[3:])), samples)
        rng_key, subkey = jax.random.split(rng_key)
        ixs = jax.random.permutation(subkey, jnp.arange(samples.obs.shape[0]))
        samples = jax.tree_map(lambda x: x[ixs], samples)  # shuffle
        num_updates = samples.obs.shape[0] // config.training_batch_size
        minibatches = jax.tree_map(
            lambda x: x.reshape((num_updates, num_devices, -1) + x.shape[1:]), samples
        )

        # Training
        policy_losses, value_losses = [], []
        for i in range(num_updates):
            minibatch: Sample = jax.tree_map(lambda x: x[i], minibatches)
            model_state, policy_loss, value_loss = train(model_state, minibatch)
            policy_losses.append(policy_loss.mean().item())
            value_losses.append(value_loss.mean().item())
        policy_loss = sum(policy_losses) / len(policy_losses)
        value_loss = sum(value_losses) / len(value_losses)

        et = time.time()
        hours += (et - st) / 3600
        log.update(
            {
                "train/policy_loss": policy_loss,
                "train/value_loss": value_loss,
                "hours": hours,
                "frames": frames,
            }
        )