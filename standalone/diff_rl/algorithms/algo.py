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
from abc import ABC, abstractmethod
from standalone.diff_rl.algorithms.model import BaseModel
from torch.optim import Adam, AdamW, SGD, Adamax, RMSprop
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau

class AlgoBase(ABC):
    actor_critic: BaseModel

    def __init__(
        self,
        actor_critic,
        max_iterations=1000,
        learning_rate=1e-3,
        schedule="CosineAnnealingLR",
        device="cpu",
        optimizer="Adam",
        child_class=None,
        **kwargs,
    ):
        if kwargs:
            print(
                f"{child_class if child_class is not None else self.__class__}.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        self.device = device
        self.learning_rate = learning_rate

        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.optimizer = eval(optimizer)(self.actor_critic.parameters(), lr=self.learning_rate)
        self.schedule = eval(schedule)(self.optimizer, max_iterations, self.learning_rate * 0.01)

    def test_mode(self,):
        self.actor_critic.test()

    def train_mode(self,):
        self.actor_critic.train()

    def act(self, obs, crtic_obs=None):
        pass
    
    def update(self):
        pass
    
    def process_env_step(self, losses, losses_detached, dones, rewards=None, infos=None):
        pass