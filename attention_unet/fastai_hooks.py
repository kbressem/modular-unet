# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/fastai-hooks.ipynb (unless otherwise specified).

__all__ = ['Hook', 'hook_output', 'Hooks', 'hook_outputs', 'maybe_gather', 'to_detach', 'apply']

# Cell
# default_exp fastai_hooks

import torch
from torch import nn, Tensor
from fastcore.dispatch import retain_type


# Cell

class Hook():
    "Create a hook on `m` with `hook_func`."
    def __init__(self, m, hook_func, is_forward=True, detach=True, cpu=False, gather=False):
        self.hook_func, self.detach, self.cpu, self.gather = hook_func, detach, cpu, gather
        f = m.register_forward_hook if is_forward else m.register_backward_hook
        self.hook = f(self.hook_fn)
        self.stored,self.removed = None,False

    def hook_fn(self, module, input, output):
        "Applies `hook_func` to `module`, `input`, `output`."
        if self.detach:
            input,output = to_detach(input, cpu=self.cpu, gather=self.gather),to_detach(output, cpu=self.cpu, gather=self.gather)
        self.stored = self.hook_func(module, input, output)

    def remove(self):
        "Remove the hook from the model."
        if not self.removed:
            self.hook.remove()
            self.removed=True

    def __enter__(self, *args): return self
    def __exit__(self, *args): self.remove()

    _docs = dict(__enter__="Register the hook",
                 __exit__="Remove the hook")

# Cell
def _hook_inner(m,i,o): return o if isinstance(o,(Tensor, tuple, list)) else list(o)

# Cell
def hook_output(module, detach=True, cpu=False, grad=False):
    "Return a `Hook` that stores activations of `module` in `self.stored`"
    return Hook(module, _hook_inner, detach=detach, cpu=cpu, is_forward=not grad)

# Cell
class Hooks():
    "Create several hooks on the modules in `ms` with `hook_func`."
    def __init__(self, ms, hook_func, is_forward=True, detach=True, cpu=False):
        self.hooks = [Hook(m, hook_func, is_forward, detach, cpu) for m in ms]

    def __getitem__(self,i): return self.hooks[i]
    def __len__(self):       return len(self.hooks)
    def __iter__(self):      return iter(self.hooks)
    @property
    def stored(self):        return L(o.stored for o in self)

    def remove(self):
        "Remove the hooks from the model."
        for h in self.hooks: h.remove()

    def __enter__(self, *args): return self
    def __exit__ (self, *args): self.remove()

    _docs = dict(stored = "The states saved in each hook.",
                 __enter__="Register the hooks",
                 __exit__="Remove the hooks")

# Cell
def hook_outputs(modules, detach=True, cpu=False, grad=False):
    "Return `Hooks` that store activations of all `modules` in `self.stored`"
    return Hooks(modules, _hook_inner, detach=detach, cpu=cpu, is_forward=not grad)

# Cell
def maybe_gather(x, axis=0):
    "Gather copies of `x` on `axis` (if training is distributed)"
    if num_distrib()<=1: return x
    ndim = x.ndim
    res = [x.new_zeros(*x.shape if ndim > 0 else (1,)) for _ in range(num_distrib())]
    torch.distributed.all_gather(res, x.contiguous() if ndim > 0 else x[None])
    return torch.cat(res, dim=axis) if ndim > 0 else torch.cat(res, dim=axis).mean()

# Cell
def to_detach(b, cpu=True, gather=True):
    "Recursively detach lists of tensors in `b `; put them on the CPU if `cpu=True`."
    def _inner(x, cpu=True, gather=True):
        if not isinstance(x,Tensor): return x
        x = x.detach()
        if gather: x = maybe_gather(x)
        return x.cpu() if cpu else x
    return apply(_inner, b, cpu=cpu, gather=gather)

# Cell
def apply(func, x, *args, **kwargs):
    "Apply `func` recursively to `x`, passing on args"
    if isinstance(x, (list, tuple)): return type(x)([apply(func, o, *args, **kwargs) for o in x])
    if isinstance(x,dict):  return {k: apply(func, v, *args, **kwargs) for k,v in x.items()}
    res = func(x, *args, **kwargs)
    return res if x is None else retain_type(res, x)