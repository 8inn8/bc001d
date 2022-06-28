import chex
import jax
import jax.nn as jnn
import jax.numpy as jnp
import pax
import jax.random as jr
import functools


class ResidualBlock(pax.Module):
    def __init__(self, dim):
        super().__init__()
        self.bn1 = pax.BatchNorm2D(dim)
        self.bn2 = pax.BatchNorm2D(dim)
        self.conv1 = pax.Conv2D(dim, dim, 3)
        self.conv2 = pax.Conv2D(dim, dim, 3)

    @jax.jit
    def __call__(self, x):
        t = jnn.swish(self.bn1(x))
        t = self.conv1(t)
        t = jnn.swish(self.bn2(x))
        t = self.conv2(t)
        return x + t


class ResnetPolicyValueNet(pax.Module):
    def __init__(self, input_dims, num_actions: int, dim: int = 128, num_resblock: int = 20):
        super().__init__()


        self.input_dims = input_dims
        self.num_actions = num_actions
        self.backbone = pax.Sequential(pax.Conv2D(1, dim, 1), pax.BatchNorm2D(dim), jnn.swish)

        for _ in range(num_resblock):
            self.backbone >>= ResidualBlock(dim)
        self.action_head = pax.Sequential(
            pax.Conv2D(dim, dim, 1),
            pax.BatchNorm2D(dim),
            jnn.swish,
            pax.Conv2D(dim, self.num_actions, kernel_shape=input_dims, padding="VALID"),
        )
        self.value_head = pax.Sequential(
            pax.Conv2D(dim, dim, 1),
            pax.BatchNorm2D(dim),
            jnn.swish,
            pax.Conv2D(dim, dim, kernel_shape=input_dims, padding="VALID"),
            pax.BatchNorm2D(dim),
            jnn.swish,
            pax.Conv2D(dim, 1, kernel_shape=1, padding="VALID"),
            jnp.tanh
        )

    def __call__(self, x: chex.Array, batched: bool = False):
        x = x.astype(jnp.float32)
        if not batched:
            x = x[None]
        x = x[..., None]
        x = self.backbone(x)
        action_logits = self.action_head(x)
        value = self.value_head(x)

        if batched:
            return action_logits[:, 0, 0, :], value[:, 0, 0, 0]
        return action_logits[0, 0, 0, :], value[0, 0, 0, 0]