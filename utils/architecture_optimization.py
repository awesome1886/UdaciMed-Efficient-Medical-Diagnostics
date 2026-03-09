"""
Architecture optimization utilities for hardware-aware model optimization in medical imaging.

This module provides comprehensive implementations of modern neural network optimization
techniques specifically designed for clinical deployment scenarios. Focuses on reducing
computational overhead, memory usage, and inference latency while maintaining diagnostic
accuracy for the PneumoniaMNIST binary classification task.
"""

import torch
import torch.nn as nn
import copy

def _set_module(model, name, new_module):
    """
    Helper function to safely replace a layer within a nested PyTorch model.
    """
    parts = name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def apply_interpolation_removal_optimization(model):
    """
    Removes the 64x64 -> 224x224 interpolation step at the beginning of the forward pass.
    This drastically reduces activation memory and compute overhead.
    """
    if hasattr(model, 'target_size'):
        # Set to 64 (native size) instead of None to safely bypass the interpolation check
        model.target_size = 64 
        print(" -> Interpolation removed: Native resolution (64x64) will be used.")
    return model


def apply_depthwise_separable_optimization(model):
    """
    Replaces standard 3x3 Convolutions with Depthwise Separable Convolutions
    (a depthwise conv followed by a 1x1 pointwise conv) to save parameters and FLOPs.
    """
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1 and module.groups == 1:
            in_c = module.in_channels
            out_c = module.out_channels
            
            # 1. Depthwise layer (groups = in_channels)
            depthwise = nn.Conv2d(in_c, in_c, kernel_size=module.kernel_size, 
                                  stride=module.stride, padding=module.padding, 
                                  groups=in_c, bias=False)
            
            # 2. Pointwise layer (1x1 conv)
            pointwise = nn.Conv2d(in_c, out_c, kernel_size=1, bias=module.bias is not None)
            
            if module.bias is not None:
                pointwise.bias.data = module.bias.data.clone()
                
            new_module = nn.Sequential(depthwise, pointwise)
            _set_module(model, name, new_module)
            count += 1
            
    print(f" -> Replaced {count} layers with Depthwise Separable Convolutions.")
    return model


def apply_grouped_convolution_optimization(model, groups=2):
    """Replaces standard 3x3 convolutions with grouped convolutions."""
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1 and module.groups == 1:
            in_c = module.in_channels
            out_c = module.out_channels
            
            if in_c % groups == 0 and out_c % groups == 0:
                new_conv = nn.Conv2d(in_c, out_c, kernel_size=module.kernel_size, 
                                     stride=module.stride, padding=module.padding, 
                                     groups=groups, bias=module.bias is not None)
                _set_module(model, name, new_conv)
                count += 1
                
    print(f" -> Replaced {count} layers with Grouped Convolutions (groups={groups}).")
    return model


def apply_lowrank_factorization(model, rank_ratio=0.25):
    """Applies SVD (Singular Value Decomposition) to large Linear layers."""
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            out_feat, in_feat = module.weight.shape
            
            if out_feat * in_feat > 10000:
                rank = max(1, int(min(in_feat, out_feat) * rank_ratio))
                W = module.weight.data.clone()
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                
                U_r = U[:, :rank]
                S_r = S[:rank]
                Vh_r = Vh[:rank, :]
                
                fc1 = nn.Linear(in_feat, rank, bias=False)
                fc2 = nn.Linear(rank, out_feat, bias=module.bias is not None)
                
                fc1.weight.data = Vh_r
                fc2.weight.data = U_r * S_r.unsqueeze(0)
                if module.bias is not None:
                    fc2.bias.data = module.bias.data.clone()
                
                new_module = nn.Sequential(fc1, fc2)
                _set_module(model, name, new_module)
                count += 1
                
    print(f" -> Factorized {count} Linear layers using Low-Rank SVD.")
    return model


def apply_channel_optimization(model):
    """
    Applies hardware-aware channel optimizations:
    1. Converts memory format to channels_last (NHWC) which Tensor Cores prefer.
    2. Converts ReLU activations to inplace=True to save activation memory.
    """
    count = 0
    # 1. Convert to channels_last memory format
    model = model.to(memory_format=torch.channels_last)
    
    # 2. Convert ReLUs to inplace
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.ReLU) and not module.inplace:
            module.inplace = True
            count += 1
            
    print(f" -> Converted {count} ReLU layers to inplace=True and applied channels_last memory format.")
    return model


def apply_inverted_residual_optimization(model):
    """Placeholder for Inverted Residual implementation."""
    print(" -> Inverted Residuals selected but relies on manual block replacement (skipping auto-apply).")
    return model


def apply_parameter_sharing(model):
    """Placeholder for Parameter sharing."""
    print(" -> Parameter Sharing selected (skipping auto-apply to preserve standard ResNet flow).")
    return model


def create_optimized_model(baseline_model, config):
    """Applies the selected optimizations to the model"""
    model = copy.deepcopy(baseline_model)
    
    print("Starting clinical model optimization pipeline...")
    
    if config.get('interpolation_removal', False):
        print("- Applying Interpolation Removal...")
        model = apply_interpolation_removal_optimization(model)
        
    if config.get('depthwise_separable', False):
        print("- Applying Depthwise Separable Convolutions...")
        model = apply_depthwise_separable_optimization(model)
        
    if config.get('channel_optimization', False):
        print("- Applying Channel Optimization...")
        model = apply_channel_optimization(model)
        
    print("--- Optimization Pipeline Complete ---")
    return model