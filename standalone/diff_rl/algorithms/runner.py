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
import rsl_rl
from rsl_rl.env import VecEnv
from rsl_rl.modules import EmpiricalNormalization
from standalone.diff_rl.algorithms import AlgoBase, BPTT, BaseModel, BaseModelRecurrent, VisionActorCritic, VisionActorCriticRecurrent
import torch
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
from collections import deque
import time
import os
from rsl_rl.utils import store_code_state
import statistics

class AlgoRunner:
    def __init__(self, env: VecEnv, agent_cfg: dict, log_dir:str=None, device="cpu") -> None:
        self.cfg = agent_cfg
        self.alg_cfg = agent_cfg["algorithm"]
        self.policy_cfg = agent_cfg["policy"]
        self.device = device
        self.env = env
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        num_obs = obs.shape[1]
        if "critic" in extras["observations"]:
            num_critic_obs = extras["observations"]["critic"].shape[1]
        else:
            num_critic_obs = num_obs

        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic: BaseModel = actor_critic_class(
            num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)
        alg_class = eval(self.alg_cfg.pop("class_name"))    #BPTT, SHAC,...
        self.alg: AlgoBase = alg_class(actor_critic = actor_critic, max_iterations = agent_cfg["max_iterations"], device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
        
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len:bool=False):
        if self.log_dir is not None and self.writer is None:
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter
                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter
                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")
        
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs)
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lossbuffer = deque(maxlen=100)
        lossbuffer_detached = deque(maxlen=100)
        lossbuffer_tot = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float32, device=self.device)
        cur_loss_sum = torch.zeros(self.env.num_envs, dtype=torch.float32, device=self.device)
        cur_loss_detached_sum = torch.zeros(self.env.num_envs, dtype=torch.float32, device=self.device)
        cur_tot_loss_sum = torch.zeros(self.env.num_envs, dtype=torch.float32, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float32, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # NOTE: detach env (spilt trajectories into windows)
            self.env.unwrapped.detach()

            for t in range(self.num_steps_per_env):
                actions = self.alg.act(obs, critic_obs)
                obs, rewards, dones, extras = self.env.step(actions)
                # perform normalization
                obs = self.obs_normalizer(obs)
                if "critic" in extras["observations"]:
                    critic_obs = self.critic_obs_normalizer(extras["observations"]["critic"])
                else:
                    critic_obs = obs
                # extract losses
                loss = extras["losses"]
                loss_detached = extras["losses_detached"]
                
                # process env step
                self.alg.process_env_step(loss, loss_detached, dones, rewards, extras)

                if self.log_dir is not None:
                    if "episode" in extras:
                        ep_infos.append(extras["episode"])
                    elif "log" in extras:
                        ep_infos.append(extras["log"])
                    cur_reward_sum += rewards
                    cur_loss_sum += loss
                    cur_loss_detached_sum += loss_detached
                    cur_tot_loss_sum += (loss + loss_detached)
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lossbuffer.extend(cur_loss_sum[new_ids][:, 0].detach().cpu().numpy().tolist())
                    lossbuffer_detached.extend(cur_loss_detached_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lossbuffer_tot.extend(cur_tot_loss_sum[new_ids][:, 0].detach().cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_loss_sum[new_ids] = 0
                    cur_loss_detached_sum[new_ids] = 0
                    cur_tot_loss_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0
            
            stop = time.time()
            collection_time = stop - start

            # Learning step
            start = stop
            mean_value_loss, mean_policy_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
            if it == start_iter:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)
        
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))



    def log(self, locs:dict, width:int=80, pad:int=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/policy", locs["mean_policy_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection_time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_loss", statistics.mean(locs["lossbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_loss_detached", statistics.mean(locs["lossbuffer_detached"]), locs["it"])
            self.writer.add_scalar("Train/mean_loss_total", statistics.mean(locs["lossbuffer_tot"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Time/mean_reward", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar("Time/mean_loss", statistics.mean(locs["lossbuffer"]), self.tot_time)
                self.writer.add_scalar("Time/mean_loss_detached", statistics.mean(locs["lossbuffer_detached"]), self.tot_time)
                self.writer.add_scalar("Time/mean_loss_total", statistics.mean(locs["lossbuffer_tot"]), self.tot_time)
                self.writer.add_scalar(
                    "Time/mean_episode_length", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

                self.writer.add_scalar("Timesteps/mean_reward", statistics.mean(locs["rewbuffer"]), self.tot_timesteps)
                self.writer.add_scalar("Timesteps/mean_loss", statistics.mean(locs["lossbuffer"]), self.tot_timesteps)
                self.writer.add_scalar("Timesteps/mean_loss_detached", statistics.mean(locs["lossbuffer_detached"]), self.tot_timesteps)
                self.writer.add_scalar("Timesteps/mean_loss_total", statistics.mean(locs["lossbuffer_tot"]), self.tot_timesteps)
                self.writer.add_scalar(
                    "Timesteps/mean_episode_length", statistics.mean(locs["lenbuffer"]), self.tot_timesteps
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Policy loss:':>{pad}} {locs['mean_policy_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean total loss:':>{pad}} {statistics.mean(locs['lossbuffer_tot']):.4f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Policy loss:':>{pad}} {locs['mean_policy_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)


    def save(self, path:str, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]
    
    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        policy = self.alg.actor_critic.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.actor_critic.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def train_mode(self):
        self.alg.actor_critic.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()
    
    def eval_mode(self,):
        self.alg.actor_critic.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_path: str):
        self.git_status_repos.append(repo_path)
