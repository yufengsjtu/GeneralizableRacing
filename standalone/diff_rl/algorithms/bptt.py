# *******************************************************************************
# *                                                                             *
# *  Private and Confidential                                                   *
# *                                                                             *
# *  Unauthorized copying of this file, via any medium is strictly prohibited.  *
# *  Proprietary and confidential.                                              *
# *                                                                             *
# *  © 2024 DiffLab. All rights reserved.                                       *
# *                                                                             *
# *  Author: Yu Feng                                                            *
# *  Data: 2025/02/21     	             *
# *  Contact: yu-feng@sjtu.edu.cn                                               *
# *  Description: None                                                          *
# *******************************************************************************
from standalone.diff_rl.algorithms import AlgoBase
import torch
from standalone.diff_rl.algorithms.model import BaseModel, BaseModelRecurrent
class BPTT(AlgoBase):
    def __init__(
        self,
        actor_critic: BaseModel,
        max_iterations=1000,
        learning_rate=1e-3,
        schedule="CosineAnnealingLR",
        device="cpu",
        optimizer="Adam",
        **kwargs,
    ):
        super().__init__(actor_critic, max_iterations, learning_rate, schedule, device, optimizer, self.__class__, **kwargs)
        self.losses = []
        self.losses_detached = []
        self.dones = []
        self.rewards = []
    
    def act(self, obs, critic_obs=None):        
        return self.actor_critic.act(obs)
    
    def update(self):
        losses = torch.stack(self.losses)
        losses_detached = torch.stack(self.losses_detached)
        total_loss_mean = (losses + losses_detached).mean()
        self.optimizer.zero_grad()
        total_loss_mean.backward()
        self.optimizer.step()
        self.schedule.step()
        self.losses = []
        self.losses_detached = []
        self.dones = []
        self.rewards = []

        # [NOTE] cutoff the gradient flow of hidden states for recurrent models
        if isinstance(self.actor_critic, BaseModelRecurrent):
            self.actor_critic.detach_hidden()

        torch.cuda.empty_cache()
        return 0.0, total_loss_mean     # [NOTE] return 0.0 for compatibility with AlgoRunner for logging
    
    def process_env_step(self, losses, losses_detached, dones, rewards=None, infos=None):
        self.losses.append(losses)
        self.losses_detached.append(losses_detached)
        self.dones.append(dones)
        self.rewards.append(rewards)
        self.actor_critic.reset(dones)
    
    



        