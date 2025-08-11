#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from standalone.rsl_rl.ext.storage import RolloutStorage

class PPOLCP:
    policy: ActorCritic
    def __init__(self,
                 policy,
                 env,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 grad_penalty_coef_schedule = None,
                 ):
        self.env = env
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Adaptation
        assert grad_penalty_coef_schedule is not None, f"Grad penalty coefficents schedule should be set."
        self.gradient_penalty_coef_schedule = grad_penalty_coef_schedule
        self.counter = 0
    
    def init_storage(self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(
            training_type, num_envs, num_transitions_per_env, actor_obs_shape,  critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.policy.test()
    
    def train_mode(self):
        self.policy.train()

    def act(self, obs, critic_obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs

        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)
        
        return rewards
    
    def compute_returns(self, last_critic_obs):
        last_values= self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def _calc_grad_penalty(self, obs_batch, actions_log_prob_batch):
        grad_log_prob = torch.autograd.grad(actions_log_prob_batch.sum(), obs_batch, create_graph=True)[0]
        gradient_penalty_loss = torch.sum(torch.square(grad_log_prob), dim=-1).mean()
        return gradient_penalty_loss
    
    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_grad_penalty_loss = 0

        if self.policy.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for sample in generator:
                obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch = sample
                obs_est_batch = obs_batch.clone()
                obs_est_batch.requires_grad_()
                
                self.policy.act(obs_est_batch, masks=masks_batch, hidden_states=hid_states_batch[0]) # match distribution dimension
                actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
                value_batch = self.policy.evaluate(
                    critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
                )
                mu_batch = self.policy.action_mean
                sigma_batch = self.policy.action_std
                entropy_batch = self.policy.entropy
                
                # Calculate the gradient penalty loss
                gradient_penalty_loss = self._calc_grad_penalty(obs_est_batch, actions_log_prob_batch)
                gradient_stage = min(max((self.counter - self.gradient_penalty_coef_schedule[2]), 0) / self.gradient_penalty_coef_schedule[3], 1)
                gradient_penalty_coef = gradient_stage * (self.gradient_penalty_coef_schedule[1] - self.gradient_penalty_coef_schedule[0]) + self.gradient_penalty_coef_schedule[0]
                              
                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) 
                            + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) 
                            / (2.0 * torch.square(sigma_batch))
                            - 0.5,
                            axis=-1
                        )
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + \
                       self.value_loss_coef * value_loss - \
                       self.entropy_coef * entropy_batch.mean() + \
                       gradient_penalty_coef * gradient_penalty_loss

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_grad_penalty_loss += gradient_penalty_loss.item()
                

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_grad_penalty_loss /= num_updates
        
        self.storage.clear()
        self.update_counter()
        return {
            "value_function": mean_value_loss, 
            "surrogate": mean_surrogate_loss, 
            "grad_penalty": mean_grad_penalty_loss, 
            "gradient_penalty_coef": gradient_penalty_coef
        }

    
    def update_counter(self):
        self.counter += 1