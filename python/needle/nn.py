"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:

    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):

    def forward(self, x):
        return x


class Linear(Module):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 device=None,
                 dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.kaiming_uniform(self.in_features, self.out_features))
        self.b = bias
        if self.b:
            self.bias = Parameter(
                init.kaiming_uniform(self.out_features, 1).transpose())
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # (..., N_out)
        wx = X @ self.weight
        output_shape = list(X.shape[:-1]).append(self.out_features)
        if self.b:
            real_bias = ops.broadcast_to(self.bias, wx.shape)
            wx += real_bias
        return wx
        ### END YOUR SOLUTION


class Flatten(Module):

    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        B = X.shape[0]
        s = 1
        for dim in X.shape[1:]:
            s *= dim
        return X.reshape((B, s))
        ### END YOUR SOLUTION


class ReLU(Module):

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):

    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for m in self.modules:
            x = m.forward(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):

    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        nc = logits.shape[-1]
        y_1_hot = init.one_hot(nc, y)
        axes = tuple(list(range(1, len(logits.shape))))
        Zy = ops.summation(logits * y_1_hot, axes=axes)
        return ops.summation(ops.logsumexp(logits, axes=axes) -
                             Zy) / logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):

    def __init__(self,
                 dim,
                 eps=1e-5,
                 momentum=0.1,
                 device=None,
                 dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim))
        self.bias = Parameter(init.zeros(self.dim))
        self.running_mean = init.zeros(self.dim)
        self.running_var = init.ones(self.dim)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]

        if not self.training:
            mean = ops.broadcast_to(self.running_mean, x.shape).detach()
            var = ops.broadcast_to(
                ops.power_scalar(self.running_var.detach() + self.eps, 0.5), x.shape).detach()
            rx = (x - mean.detach()) / var.detach()
        else:
            mean = x.sum(axes=(0, )) / batch_size
            self.running_mean = ((
                1 - self.momentum) * self.running_mean.detach() + self.momentum * mean).detach()
            mean = mean.broadcast_to(x.shape)
            var = ((x - mean)**2).sum(axes=(0, )) / batch_size
            self.running_var = ((1 - self.momentum) * self.running_var.detach(
            ) + self.momentum * var.detach()).detach()
            var = var.broadcast_to(var.shape)
            rx = (x - mean) / ops.power_scalar(var + self.eps, 0.5)
        weight = self.weight.broadcast_to(x.shape)
        return weight * rx + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION

class LayerNorm1d(Module):

    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim))
        self.bias = Parameter(init.zeros(self.dim))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        m = x.shape[0]
        n = x.shape[1]
        ex = x.sum(axes=(1, )) / n
        ex = ex.reshape((m, 1)).broadcast_to((m, n))
        xu = x - ex
        x2 = ops.power_scalar((xu **2).sum(axes=(1, )) / n + self.eps, 0.5)
        x2 = x2.reshape((m, 1)).broadcast_to((m, n))
        return self.weight.broadcast_to(x.shape) * (xu / x2) + self.bias.broadcast_to(x.shape)


class Dropout(Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
            return x
        px = init.randb(*x.shape, p=1 - self.p)
        return x * px / (1 - self.p)
        ### END YOUR SOLUTION


class Residual(Module):

    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
