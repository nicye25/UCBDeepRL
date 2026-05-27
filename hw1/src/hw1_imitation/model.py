"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dims[0]))
        layers.append(nn.ReLU())

        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dims[-1], chunk_size * action_dim))

        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        flat_output = self.network(state)
        return flat_output.reshape(-1, self.chunk_size, self.action_dim)


    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        loss_fn = nn.MSELoss()
        return loss_fn(self(state), action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        return self(state)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        input_dim = state_dim + (chunk_size * action_dim) + 1

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())

        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dims[-1], chunk_size * action_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self,
                state: torch.Tensor,
                action_chunk_tau: torch.Tensor,
                tau: torch.Tensor
    ) -> torch.Tensor:
        
        if tau.dim() == 1:
            tau = tau.unsqueeze(-1)
        
        flat_action = action_chunk_tau.reshape(state.shape[0], -1)
        x = torch.cat([state, flat_action, tau], dim= -1)
        flat_output = self.network(x)
        return flat_output.reshape(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]

        noise = torch.randn_like(action_chunk)

        tau = torch.rand((batch_size, 1), device=state.device, dtype=state.dtype)
        tau_broadcast = tau.unsqueeze(-1)
        action_chunk_tau = tau_broadcast * action_chunk + (1.0 - tau_broadcast) * noise

        target_velocity = action_chunk - noise
        predicted_velocity = self.forward(state, action_chunk_tau, tau)
        loss_fn = nn.MSELoss()
        return loss_fn(predicted_velocity, target_velocity)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch_size = state.shape[0]

        current_actions = torch.randn(
            (batch_size, self.chunk_size, self.action_dim),
            device=state.device,
            dtype=state.dtype
        )
        d_tau = 1.0 / num_steps
        for step in range(num_steps):
            tau_val = step * d_tau
            tau_tensor = torch.full((batch_size, 1), tau_val, device=state.device, dtype=state.dtype)
            with torch.no_grad():
                velocity = self.forward(state, current_actions, tau_tensor)
            
            current_actions = current_actions + velocity * d_tau

        return current_actions


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
