"""Restore a complete projection path, including removal of scale transport."""


def restore_projection(model, name, original, device):
    import torch
    if not isinstance(original, torch.nn.Linear) or hasattr(original, 'beta_inverse'):
        raise ValueError('original BF16 nn.Linear required, not a quantized adapter')
    current = model.get_submodule(name)
    if (current.in_features, current.out_features) != (original.in_features, original.out_features):
        raise ValueError('restoration shape mismatch')
    parent, leaf = name.rsplit('.', 1)
    linear = original.to(device)
    linear.requires_grad_(False)
    setattr(model.get_submodule(parent), leaf, linear)
    return linear
