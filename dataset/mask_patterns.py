import torch
from torch import Tensor
import numpy as np
import math


def fully_random(x, keep_percentage=0.5):
    """
    Create a mask with keep_percentage of the tokens set to 0.
    """
    mask = torch.rand(x.size())
    mask = (mask > keep_percentage).type(torch.int)
    return mask.to(x.device)


def pattern(x, shift=0, keep_percentage=0.5):
    """
    Create a mask with keep_percentage of the tokens set to 0.
    The pattern will be similar to a checkerboard, but depend on the drop probability.
    Shift moves the pattern by shift pixels to the right.
    """
    mask = torch.zeros_like(x, dtype=torch.int)
    if keep_percentage < 1.0:
        mask = torch.ones_like(x, dtype=torch.int)
        # Flatten the tensor and keep as many evenly spaced elements as the keep probability.
        mask = mask.flatten()
        # The mask is evenly spaced, but because we want consecutive images to save different parts
        # of the image, we enable shifting the mask.
        size = mask.numel()
        numElems = round(size * keep_percentage.item())
        idx = torch.round(
            torch.linspace((0 + shift), (size - 1 + shift), numElems)).int()
        # Take modulo of the indices to ensure that the indices are within the bounds of the tensor.
        idx = idx % size
        mask[idx] = torch.tensor(0, dtype=torch.int, device=x.device)
        mask = mask.reshape(x.size())
    return mask


def attention_based(x, attn, keep_percentage=0.5):
    """
    Create a mask based on the attention weights.
    :param x: The input tensor of one VAE encoded image - shape (H, W)
    :param keep_percentage: What percentage of tokens to keep
    :param attn: The self-attention weights - shape (seq_len, seq_len)
    """
    H, W = x.shape
    mask = torch.zeros_like(x, dtype=torch.int)
    if keep_percentage < 1.0:
        mask = torch.ones_like(x, dtype=torch.int)
        mask = mask.flatten()
        # Remove the class token as we can not keep it and therefore should
        # not include it in our analysis of token importance
        attn = attn[1:]
        attn = attn[:, 1:]
        assert attn.shape[0] == H * W
        num_tokens = round(H * W * (keep_percentage))
        # We assume columns in the attention matrix with a higher sum
        # to be more informative and thus more important to keep (each
        # column j tells us how important key j is for all queries).
        sum_attn = torch.sum(attn, dim=0)
        sort, idx = torch.sort(sum_attn, descending=True)
        # Keep the top num_tokens tokens.
        idx = idx[:num_tokens]
        if idx.dim() == 0:
            # This is required if the function is called with vmap. In this case
            # the idx tensor will be a scalar and .item() would be called, which
            # would raise an error.
            idx = idx.unsqueeze(0)
        mask[idx] = torch.tensor(0, dtype=torch.int, device=x.device)
        mask = mask.reshape(x.shape)
    return mask


def patched_key_attention(x, attn, patch_size=(4, 4), keep_percentage=0.5):
    """
    In contrast to the attention_based mask, this mask picks the most important keys in each patch.
    :param
        x (Tensor): The input tensor of one VAE encoded image - shape (H, W).
        attn (Tensor): The self-attention weights - shape (seq_len, seq_len).
        bin_size (Tuple[int, int]): The size of the bins in which the tokens in the VAE encoded image should be grouped.
        keep_percentage (float): What percentage of tokens to keep.
    :return (Tensor): The mask based on the attention weights.
    """
    mask = torch.zeros_like(x, dtype=torch.int)
    if keep_percentage < 1.0:
        H, W = x.shape
        mask = torch.ones_like(x, dtype=torch.int)
        mask = mask.flatten()
        # Remove the class token.
        attn = attn[1:]
        attn = attn[:, 1:]
        # Take sum over key attention.
        sum_attn = torch.sum(attn, dim=0)
        sort, idx = torch.sort(sum_attn, descending=True)
        # Create rank from sort index by writing at each position the rank of the element.
        ranks = idx.new_empty((mask.size(0), 2))
        ranks[idx, 0] = torch.arange(idx.size(0)).to(ranks.device)
        ranks[:, 1] = torch.arange(idx.size(0)).to(ranks.device)
        ranks = ranks.reshape(H, W, 2)
        num_tokens = round(H * W * (keep_percentage))
        # Create patches.
        patch_H, patch_W = patch_size
        num_patches_H = math.ceil(H / patch_H)
        num_patches_W = math.ceil(W / patch_W)
        patched_ranks = idx.new_empty(
            (num_patches_H * num_patches_W, patch_H * patch_W))
        for i in range(num_patches_H):
            for j in range(num_patches_W):
                patch = ranks[i * patch_H:(i + 1) * patch_H,
                              j * patch_W:(j + 1) * patch_W]
                patch = patch.reshape(-1, 2)
                key = patch[:, 0]
                value = patch[:, 1]
                _, idx = torch.sort(key)
                value = value[idx]
                # Add the second element of each tuple to the patched ranks.
                patched_ranks[i * num_patches_W + j] = value
        for i in range(num_tokens):
            # Iteratively sample from each patch, starting from the left most entry in each patch.
            div, mod = divmod(i, patched_ranks.size(0))
            idx = patched_ranks[mod, div]
            if idx.dim() == 0:
                # This is required if the function is called with vmap. In this case
                # the idx tensor will be a scalar and .item() would be called, which
                # would raise an error.
                idx = idx.unsqueeze(0)
            mask[idx] = torch.tensor(0, dtype=torch.int, device=x.device)
        mask = mask.reshape(x.shape)
    return mask


def k_medoids(x,
              values,
              keep_percentage=0.5,
              normalized=False,
              max_steps=100,
              choice=None):
    """
    Implements a k-medoids-based subsampling of the tokens.
    Tokens are sorted into k clusters in the space of the last transformer 
    layer (or any layer that is passed as transformer_values).
    After convergence, we keep the k cluster centroids.
    k is chosen as the number of elements in x times the keep_percentage.
    """
    mask = torch.zeros_like(x, dtype=torch.int)
    if keep_percentage < 1.0:
        H, W = x.shape
        k = round(H * W * (keep_percentage))
        mask = torch.ones_like(x, dtype=torch.int)
        if k < 1:
            return mask
        mask = mask.flatten()
        # Remove the class token.
        values = values[1:]
        n = values.size(0)
        # Initialize k random medoids.
        choice = torch.randperm(n)[:k] if choice is None else choice
        choice = torch.Tensor(choice).to(x.device).long()
        medoids = values[choice]
        # -1 indicates that the value is not assigned to a cluster.
        value_to_cluster = x.new_zeros(n, dtype=torch.long) - 1
        # Initialize value_to_cluster with so that each medoid is in its own cluster.
        value_to_cluster[choice] = torch.arange(k).to(x.device)
        for step in range(max_steps):
            # Assign each element to the closest medoid
            dist = torch.cdist(values, medoids)
            new_value_to_cluster = torch.argmin(dist, dim=1)

            # If the assignments do not change, we've converged
            if torch.equal(new_value_to_cluster, value_to_cluster):
                break

            value_to_cluster = new_value_to_cluster

            # Update medoids
            for j in range(k):
                cluster_indices = torch.where(value_to_cluster == j)[0]
                cluster_values = values[cluster_indices]
                intra_cluster_distances = torch.cdist(cluster_values,
                                                      cluster_values)
                sum_distances = intra_cluster_distances.sum(dim=1)
                new_medoid_index = cluster_indices[torch.argmin(sum_distances)]
                choice[j] = new_medoid_index
                medoids[j] = values[new_medoid_index]
        mask[choice] = torch.tensor(0, dtype=torch.int, device=x.device)
        mask = mask.reshape(x.shape)
    return mask
