# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/utils.ipynb (unless otherwise specified).

__all__ = ['all_equal', 'first_layer', 'hasattrs', 'test_forward']

# Cell
# default_exp utils
import torch

# Cell
def all_equal(iterator):
    " Check if all elements of a list are equal (https://stackoverflow.com/a/3844832/12995344)"
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)

# Cell
def first_layer(module):
    " Extract the first layer of a (nested) nn.Sequential"
    while True:
        try: module = next(module.children())
        except: return module

# Cell
def hasattrs(x, attrs, do_raise=False):
    " Check if `x` has all atributes. Optionally raise `ValueError` if some are missing"
    present = [hasattr(x, attr) for attr in attrs]
    all_present = all(present)
    if do_raise and not all_present:
        attrs = [attr for attr, p in zip(attrs, present) if not p]
        raise ValueError(f'{x.__class__.__name__} has no attributes {attrs}')
    return all_present

# Cell
def test_forward(model, inp_sz = (10, 25, 25), check_size=True):
    try: in_c = first_layer(model).in_channels
    except: in_c = 3 # make an educated guess :)
    x = torch.randn(2, in_c, *inp_sz)
    out = model(x)
    if check_size:
        assert out.shape[2:] == x.shape[2:], 'Size of input and output are not equal.'