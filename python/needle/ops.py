"""Operator implementations."""

from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):

    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):

    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):

    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):

    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar, )


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * self.scalar * power_scalar(node.inputs[0], self.scalar - 1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # lhs / rhs, 1/rhs, lhs * (-1) / rhs / rhs
        lhs, rhs = node.inputs
        return out_grad / rhs, out_grad * lhs * (-1) * power_scalar(rhs, -2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):

    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):

    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        dims = len(a.shape)
        (swap1, swap2) = self.axes if self.axes is not None else (dims - 1,
                                                                  dims - 2)
        new_axes = list(range(dims))
        new_axes[swap1], new_axes[swap2] = new_axes[swap2], new_axes[swap1]
        return array_api.transpose(a, axes=new_axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.transpose(axes=self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):

    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):

    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        g = out_grad

        sa = a.shape
        sg = g.shape

        bdim = []
        for i in range(1, len(sa) + 1):
            if sg[-i] != sa[-i]:
                assert sa[-i] == 1, "broadcast size 1 on dim {}".format(i - 1)
                bdim.append(len(sg) - i)

        for i in range(len(sa), len(sg)):
            bdim.append(len(sg) - 1 - i)
        return g.sum(tuple(bdim)).reshape(a.shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):

    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        a = list(input_shape)
        if self.axes is not None:
            for ax in self.axes:
                a[ax] = 1
        else:
            a = array_api.ones(len(input_shape), dtype=int)
        og = reshape(out_grad, tuple(a)).broadcast_to

        return out_grad.reshape(tuple(a)).broadcast_to(input_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # (, N, K)
        a = node.inputs[0]
        # (, K, M)
        b = node.inputs[1]
        # (, N, M)
        c = out_grad

        va = c @ transpose(b)
        vb = transpose(a) @ c

        if va.shape != a.shape:
            va = va.sum(tuple(range(len(va.shape) - len(a.shape))))
        if vb.shape != b.shape:
            vb = vb.sum(tuple(range(len(vb.shape) - len(b.shape))))
        return va, vb
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


# TODO
class ReLU(TensorOp):

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(0, a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return Tensor(out_grad * (node.inputs[0].realize_cached_data() > 0))
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

def padding_shape(shape, insert_index) -> tuple[int]:
    shape_list = list(shape)
    if insert_index is None:
        insert_index = range(len(shape_list))
    ret = []
    ni = []
    for x in insert_index:
        if x < 0:
            x += len(shape_list)
        ni.append(x)

    ni.sort()
    for i in range(len(shape_list)):
        if len(insert_index) > 0 and len(ret) == insert_index[0]:
            ret.append(1)
            insert_index = insert_index[1:]
        ret.append(shape_list[i])
    ret += [1 for i in range(len(insert_index))]
    
    return tuple(ret)

def recover_shape(t, axes, origin_shape):
    reshapped = t.reshape(padding_shape(t.shape, axes))
    return broadcast_to(reshapped, origin_shape)

def recover_shape_array_api(t, axes, origin_shape):
    reshapped = t.reshape(padding_shape(t.shape, axes))
    return array_api.broadcast_to(reshapped, origin_shape)

class LogSumExp(TensorOp):

    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # Z: m x n
        Zmax = array_api.max(Z, axis=self.axes)
        b = recover_shape_array_api(Zmax, self.axes, Z.shape)
        ret = array_api.log(array_api.exp(Z - b).sum(axis=self.axes)) + Zmax
        return ret

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        Zmax = array_api.max(Z.realize_cached_data(), axis=self.axes)
        b = recover_shape_array_api(Zmax, self.axes, Z.shape)
        Zsub = Z - Tensor(b)
        Zexp = exp(Zsub)
        Znorm = recover_shape(summation(Zexp, axes=self.axes), self.axes, Z.shape)
        return recover_shape(out_grad, self.axes, Z.shape) * (Zexp / Znorm)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
