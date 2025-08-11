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
# *  Data: 2025/03/28     	             *
# *  Contact: yu-feng@sjtu.edu.cn                                               *
# *  Description: None                                                          *
# *******************************************************************************
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from standalone.rsl_rl.ext.modules.vision_actor_critic import resolve_nn_activation
from standalone.rsl_rl.ext.modules.student_teacher import StudentTeacher

class VisionStudentTeacher(StudentTeacher):
    is_recurrent = False

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,
        num_actions,
        img_res: tuple[int, int],
        dim_hidden_input: int,
        student_hidden_dims=[256, 256, 256],
        teacher_hidden_dims=[256, 256, 256],
        activation="relu",
        init_noise_std=0.1,
        noise_std_type: str="scalar",
        **kwargs
    ):
        if kwargs:
            print(
                "VisionStudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__(
            num_student_obs=dim_hidden_input,
            num_teacher_obs=dim_hidden_input,
            num_actions=num_actions,
            student_hidden_dims=student_hidden_dims,
            teacher_hidden_dims=teacher_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std
        )
        self.noise_std_type=noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Invalid noise_std_type: {self.noise_std_type}")

        self.activation = resolve_nn_activation(activation)

        self.img_res = img_res

        # Depth: height: 72, width: 96
        assert img_res[0] == 72 and img_res[1] == 96, f"Image dimension should be 72x96, got {img_res}"
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, 3, bias=False),  # 24x32x16
            nn.BatchNorm2d(16),
            self.activation,
            nn.Conv2d(16, 32, 3, 3, bias=False),    # 8x10x32
            nn.BatchNorm2d(32),
            self.activation,
            nn.Conv2d(32, 64, 2, 2, bias=False),    # 4x5x64
            nn.BatchNorm2d(64),
            self.activation,
            nn.Flatten(),   # 1280
            nn.Linear(1280, dim_hidden_input),    # 128
        )

        # state: num_actor_state_obs
        self.state_enc = nn.Linear(num_student_obs - img_res[0] * img_res[1], dim_hidden_input)
    
    def update_distribution(self, observations):
        img = observations[:, -self.img_res[0] * self.img_res[1]:].view(-1, 1, self.img_res[0], self.img_res[1])
        state = observations[:, :-self.img_res[0] * self.img_res[1]]
        feat = self.activation(self.stem(img) + self.state_enc(state))
        mean = self.student(feat)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Invalid noise_std_type: {self.noise_std_type}")
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def act_inference(self, observations):
        img = observations[:, -self.img_res[0] * self.img_res[1]:].view(-1, 1, self.img_res[0], self.img_res[1])
        state = observations[:, :-self.img_res[0] * self.img_res[1]]
        feat = self.activation(self.stem(img) + self.state_enc(state))
        mean = self.student(feat)
        return mean
    
    def evaluate(self, privileged_obs, **kwargs):
        img = privileged_obs[:, -self.img_res[0] * self.img_res[1]:].view(-1, 1, self.img_res[0], self.img_res[1])
        state = privileged_obs[:, :-self.img_res[0] * self.img_res[1]]
        feat = self.activation(self.stem(img) + self.state_enc(state))
        return self.teacher(feat)
        