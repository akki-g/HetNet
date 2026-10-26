"""Reconciled, separately versioned paper-v1 HetNet architecture.

Paper figure 1/section 4.2: independent class status preprocessing/recurrence.
Supplement section 2.1: three rounds, four 16-wide hidden heads, final averaging.
Public-source interpretation: two class-routed SENs and FC plus ReLU/LSTM input
branches. SEN metadata is [N_P,N_A,world_width,zero_based_episode_step].
"""
import math
import operator

import torch
from torch import nn

from hetgat.uavnet import UAVNetA2CEasy
from hetgat.graph.paper import PaperLayer
from hetgat.graph.torch_backend import StaticTopology


class PaperNet(nn.Module):
    # Retain the established typed observation layout and independent copy/cast
    # semantics. These helpers neither construct graphs nor run learned modules.
    remove_excess_action_features_from_all = UAVNetA2CEasy.remove_excess_action_features_from_all
    get_obs_features = UAVNetA2CEasy.get_obs_features

    def __init__(self, in_dim_raw, in_dim, hid_dim, out_dim, num_P, num_A,
                 num_heads, msg_dim=64, use_CNN=True, use_real=True, use_tanh=False,
                 per_class_critic=False, per_agent_critic=False, device=None,
                 with_two_state=False, obs=1, comm_range_P=-1, comm_range_A=-1,
                 lossy_comm=False, min_comm_loss=0, max_comm_loss=0.3,
                 tensor_obs=False, total_state_action_in_batch=500,
                 action_vision=-1, model_spec="paper-v1", message_backend="dgl"):
        super().__init__()
        if model_spec != "paper-v1":
            raise ValueError("PaperNet requires model_spec=paper-v1")
        if num_heads != 4 or any(hid_dim.get(key) != 16 for key in ("P", "A", "state")):
            raise ValueError("paper-v1 requires four 16-wide hidden heads")
        if not use_real and msg_dim != 64:
            raise ValueError("paper-v1 fixes 64 bits per head (256 per sender per round)")
        if not per_class_critic or per_agent_critic or not with_two_state:
            raise ValueError("paper-v1 requires per-class critics and two state nodes")
        if tensor_obs or action_vision != -1 or lossy_comm or use_tanh:
            raise ValueError("Unsupported paper-v1 observation, loss or critic configuration")
        if message_backend not in ("dgl", "torch-v1"):
            raise ValueError("Unknown message backend: " + str(message_backend))
        if message_backend == "torch-v1" and (comm_range_P != -1 or comm_range_A != -1):
            raise ValueError("torch-v1 requires unlimited, non-lossy communication")
        self.device = torch.device("cpu") if device is None else torch.device(device)
        if self.device.type != "cpu":
            raise ValueError("paper-v1 currently supports the pinned CPU runtime only")
        if num_P < 1 or num_A < 0:
            raise ValueError("paper-v1 requires at least one P agent and nonnegative A count")
        self.model_spec, self.message_backend = model_spec, message_backend
        self.num_P, self.num_A = num_P, num_A
        self.in_dim, self.vision = dict(in_dim), in_dim_raw["vision"]
        self.P_s = in_dim["P"] - in_dim["state"]
        self.world_dim = math.isqrt(self.P_s)
        self.obs_squares = 1 if obs is None else obs
        self.per_class_critic, self.per_agent_critic = True, False
        self.with_two_state, self.use_real = True, use_real
        self.use_CNN, self.use_tanh, self.tensor_obs = False, False, False
        self.episode_step = 0
        self._topology = None
        self.prepro_stat = nn.ModuleDict({
            key: nn.Linear(self.P_s * self.obs_squares, self.P_s * self.obs_squares)
            for key in ("P", "A")})
        self.f_module_stat = nn.ModuleDict({
            key: nn.LSTMCell(self.P_s * self.obs_squares, self.P_s)
            for key in ("P", "A")})
        self.prepro_obs = nn.Linear(in_dim["state"] * self.obs_squares,
                                    in_dim["state"] * self.obs_squares)
        self.f_module_obs = nn.LSTMCell(in_dim["state"] * self.obs_squares, in_dim["state"])
        hidden_inputs = {key: 16 * num_heads for key in ("P", "A", "state")}
        layer_args = dict(num_heads=num_heads, msg_dim=msg_dim, use_real=use_real,
                          message_backend=message_backend)
        self.layer1 = PaperLayer(in_dim, hid_dim, **layer_args)
        self.layer2 = PaperLayer(hidden_inputs, hid_dim, **layer_args)
        self.layer3 = PaperLayer(hidden_inputs, out_dim, merge="avg", **layer_args)
        self.P_critic_head = nn.Linear(out_dim["state"], 1)
        self.A_critic_head = nn.Linear(out_dim["state"], 1)
        self.relu = nn.ReLU()

    def set_episode_step(self, step):
        if isinstance(step, bool):
            raise ValueError("Episode step must be a nonnegative integer")
        try:
            step = operator.index(step)
        except TypeError as error:
            raise ValueError("Episode step must be a nonnegative integer") from error
        if step < 0:
            raise ValueError("Episode step must be a nonnegative integer")
        self.episode_step = step

    def init_hidden(self, batch_size):
        if batch_size != 1:
            raise ValueError("paper-v1 processes one team at a time")
        return {key: tuple(torch.zeros(count, width, requires_grad=True, device=self.device)
                           for _ in range(2))
                for key, count, width in (("P_s", self.num_P, self.P_s),
                                           ("P_o", self.num_P, self.in_dim["state"]),
                                           ("A_s", self.num_A, self.in_dim["A"]))}

    def forward(self, x, g=None, episode_step=None):
        if episode_step is not None:
            self.set_episode_step(episode_step)
        observations, previous = x
        p_status, a_status = self.remove_excess_action_features_from_all(observations)
        p_observation = self.get_obs_features(observations)
        h = {}
        p_status = self.relu(self.prepro_stat["P"](p_status.clone().detach()))
        h["P_s"] = self.f_module_stat["P"](p_status, previous["P_s"])
        p_observation = self.relu(self.prepro_obs(p_observation))
        h["P_o"] = self.f_module_obs(p_observation, previous["P_o"])
        if self.num_A:
            a_status = self.relu(self.prepro_stat["A"](torch.Tensor(a_status)))
            h["A_s"] = self.f_module_stat["A"](a_status, previous["A_s"])
        else:
            h["A_s"] = previous["A_s"]
        features = {
            "P": torch.cat([h["P_s"][0], h["P_o"][0]], dim=1),
            "A": h["A_s"][0],
            "state": p_status.new_tensor([
                [self.num_P, self.num_A, self.world_dim, self.episode_step],
                [self.num_P, self.num_A, self.world_dim, self.episode_step]]),
        }
        if self.message_backend == "torch-v1" and self._topology is None:
            self._topology = StaticTopology(self.num_P, self.num_A, with_two_state=True)
        for layer in (self.layer1, self.layer2, self.layer3):
            features = layer(g, features, topology=self._topology)
        p_value = self.P_critic_head(self.relu(features["state"][:1]))
        a_value = self.A_critic_head(self.relu(features["state"][1:]))
        return features, p_value, a_value, h
