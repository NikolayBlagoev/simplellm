from torch.nn import Module, Parameter, Linear
from torch.nn import functional as F, init
from torch import tensor, matmul, Tensor
import torch
import math
from torch.cuda.amp import custom_bwd, custom_fwd
class CustomLinear(Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
       
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        
        return LinearWithGradAccumulation.apply(input,self.weight,self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
class WeightStore:
    cache = []

    def put(func, *args):
        WeightStore.cache.append((func,args))

    def poll():
        if len(WeightStore.cache) == 0:
            return
        f, args = WeightStore.cache.pop()
        f(*args)

    def flush():
        while len(WeightStore.cache) != 0:
            f, args = WeightStore.cache.pop()
            f(*args)


class LinearWithGradAccumulation(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Parameter,
        bias: Parameter
    ):
        ctx.save_for_backward(input, weight, bias)
        ctx.use_bias = bias is not None

        output = matmul(input.data, weight.t())
        if bias is not None:
            output = output + bias.data


        return output
    @staticmethod
    def delayed_weight_update(weight: Parameter, bias: Parameter, grad_output: Tensor, input: Tensor):
        grad_output = grad_output.to(weight.device)
        input = input.to(weight.device)
        w_grad = grad_output.mT @ input

        if weight.grad == None:
            weight.grad = torch.mean(w_grad,axis = 0)
        else:
            weight.grad += torch.mean(w_grad,axis = 0)


        if bias is not None:
            b_grad = grad_output.sum(axis=0)
            if bias.grad == None:
                bias.grad = torch.mean(w_grad,axis = 0)
            else:
                bias.grad += torch.mean(w_grad,axis = 0)
            del b_grad
        del w_grad
        torch.cuda.empty_cache()
        
        
        
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        use_bias = ctx.use_bias
        grad_input = grad_output @ weight.data
        WeightStore.put(LinearWithGradAccumulation.delayed_weight_update,weight,bias,grad_output.to("cpu"),input.to("cpu"))
        return grad_input, None, None
