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

class StudentTeacher(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,
        num_actions,
        student_hidden_dims=[256, 256, 256],
        teacher_hidden_dims=[256, 256, 256],
        activation="ReLU",
        init_noise_std=0.1,
        **kwargs
    ):
        if kwargs:
            print(
                "StudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)
        self.loaded_teacher = False  # indicates if teacher has been loaded
        mlp_input_dim_s = num_student_obs
        mlp_input_dim_t = num_teacher_obs

        # student
        student_layers = []
        student_layers.append(nn.Linear(mlp_input_dim_s, student_hidden_dims[0]))
        student_layers.append(activation)
        for layer_index in range(len(student_hidden_dims)):
            if layer_index == len(student_hidden_dims) - 1:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], num_actions))
            else:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], student_hidden_dims[layer_index + 1]))
                student_layers.append(activation)
        self.student = nn.Sequential(*student_layers)

        # teacher
        teacher_layers = []
        teacher_layers.append(nn.Linear(mlp_input_dim_t, teacher_hidden_dims[0]))
        teacher_layers.append(activation)
 
        for layer_index in range(len(teacher_hidden_dims)):
            if layer_index == len(teacher_hidden_dims) - 1:
                teacher_layers.append(nn.Linear(teacher_hidden_dims[layer_index], num_actions))
            else:
                teacher_layers.append(nn.Linear(teacher_hidden_dims[layer_index], teacher_hidden_dims[layer_index + 1]))
                teacher_layers.append(activation)
        self.teacher = nn.Sequential(*teacher_layers)
        self.teacher.eval()

        print(f"Student MLP: {self.student}")
        print(f"Teacher MLP: {self.teacher}")

        # action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
 
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean
 
    @property
    def action_std(self):
        return self.distribution.stddev
 
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
 
    def update_distribution(self, observations):
        mean = self.student(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        actions_mean = self.student(observations)
        return actions_mean

    def evaluate(self, teacher_observations):
        with torch.no_grad():
            actions = self.teacher(teacher_observations)
        return actions

    def load_state_dict(self, state_dict, strict=True):
        # check if state_dict contains teacher and student or just teacher parameters
        if any("actor" in key for key in state_dict.keys()):  # loading parameters from rl training
            # rename keys to match teacher and remove critic parameters
            teacher_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)

            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            return False
 

        elif any("student" in key for key in state_dict.keys()):  # loading parameters from distillation training
            super().load_state_dict(state_dict, strict=strict)
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            return True
 
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")
        
    
    def detach_hidden_states(self):
        pass

    