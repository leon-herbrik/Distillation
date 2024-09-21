import math
import warnings

from torch.utils.data.dataloader import default_collate
from torch import Tensor
import torch
import itertools

from .mask_patterns import fully_random, pattern


def test_collate_fn(batch,
                    mask_token=3072,
                    subsample_types=["random"],
                    keep_percentage=None,
                    context_before=0,
                    context_after=0,
                    codebook_size=1024,
                    past_shift=1024,
                    future_shift=2048,
                    feature_token=3073,
                    sep_token=3074,
                    num_frames=1):
    """Collate function that combines tokens from context during loading."""
    batch = default_collate(batch)
    x, y = batch
    # Select frame indices from the input tensor.
    positions = get_selected_frame_indices(x.size(1), num_frames)
    # Subsample the codes.
    masked_code, mask = get_mask_code(
        x,
        mask_value=mask_token,
        codebook_size=codebook_size,
        subsample_types=subsample_types,
        mode="linear",
        keep_percentage=keep_percentage,
    )
    if context_before > 0 or context_after > 0:
        combined_masks = []
        masked_codes = []
        for position in positions:
            # Goes from B, T, H, W to B, H, W
            curr_masked_code, combined_mask = combine_tokens_from_temporal_context(
                tokens=masked_code,
                masks=mask,
                frames_before=context_before,
                frames_after=context_after,
                position=position,
                mask_value=mask_token,
            )
            masked_codes.append(curr_masked_code)
            combined_masks.append(combined_mask)
        # Concat along the T dimension -> B, num_frames, H, W
        masked_codes = torch.stack(masked_codes, dim=1)
        masked_code = masked_codes
    else:
        # Select num_frames frames from the input tensor.
        masked_code = select_frames(masked_code, num_frames)
    masked_code = prepend_separator(masked_code, sep_token)
    masked_code = prepend_feature_token(masked_code, feature_token)
    x = masked_code
    x, y = masked_code.to(torch.int64), y.to(torch.float32)

    batch = (x, y)
    return batch


def get_mask_code(
    code,
    mode="arccos",
    subsample_types=["pattern"],
    keep_percentage=None,
    mask_value=None,
    codebook_size=256,
):
    """Replace the code token by *value* according the the *mode* scheduler
        :param
         code  -> torch.LongTensor(): bsize * 16 * 16, the unmasked code | bsize * context_size * 16 * 16
         mode  -> str:                the rate of value to mask
         mask_value -> int:                mask the code by the value
        :return
         masked_code -> torch.LongTensor(): bsize * 16 * 16, the masked version of the code | bsize * context_size * 16 * 16
         mask        -> torch.LongTensor(): bsize * 16 * 16, the binary mask of the mask | bsize * context_size * 16 * 16
        """
    # Keep percentage should be passed either as one number for each batch element or as one number for the entire batch.
    match keep_percentage:
        case Tensor():
            match keep_percentage.numel():
                case code.size(0):
                    r = keep_percentage
                case 1:
                    # Repeat the keep_percentage for each batch.
                    r = keep_percentage.repeat(code.size(0))
        case float():
            r = keep_percentage * torch.ones(code.size(0))
        case _:
            r = torch.rand(code.size(0))
    match mode:
        case "square":
            val_to_mask = r**2
        case "cosine":
            val_to_mask = torch.cos(r * math.pi * 0.5)
        case "arccos":
            val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        case "linear" | _:
            val_to_mask = r

    # After determining a random percentage of tokens to keep, we can use our subsampling strategies.
    mask_code = code.detach().clone()

    # Sample the amount of tokens + localization to mask
    mask_codes, masks = [], []
    # Give each batch a subsampling type.
    batches_per_type = math.ceil(code.size(0) / len(subsample_types))
    for i, t in enumerate(subsample_types):
        mc, m = subsample_tokens(
            tokens=mask_code[i * batches_per_type:(i + 1) * batches_per_type],
            mask_token=mask_value,
            type=t,
            keep_percentage=val_to_mask[i * batches_per_type:(i + 1) *
                                        batches_per_type],
            random_shift=True,
        )
        mask_codes.append(mc)
        masks.append(m)
    mask_code = torch.cat(mask_codes, dim=0)
    mask = torch.cat(masks, dim=0)

    return mask_code, mask


def combine_tokens_from_temporal_context(
    tokens: torch.Tensor,
    masks: torch.Tensor,
    frames_before: int,
    frames_after: int,
    position: int,
    past_shift: int = 1024,
    future_shift: int = 2048,
    mask_value=1024,
):
    """
        We take tokens from adjacent frames to fill up missing space in the current frame (identified by the position).
        The closer a frame is, temporally, to the current frame, the higher priority are its tokens.
        This function expects a shape of B, T, H, W or T, H, W.
        """
    if tokens.shape != masks.shape:
        raise ValueError(
            f"Tokens and masks must have the same shape. Tokens have shape {tokens.shape} and masks have shape {masks.shape}."
        )
    if tokens.dim() == 3:
        tokens = tokens.unsqueeze(0)
        masks = masks.unsqueeze(0)
    B, T, H, W = tokens.shape
    shape = (B, H, W)
    # Use -1 as a placeholder for missing tokens.
    combined_tokens = (
        torch.zeros(shape, dtype=tokens.dtype).to(tokens.device) + mask_value)
    # Choose randomly whether to prioritize earlier or later frames' tokens.
    combined_masks = torch.ones(shape, dtype=masks.dtype).to(masks.device)
    # Go from both sides over the masks and create a combined tensor.
    before_range = range(max(0, position - frames_before), position)
    after_range = range(min(T - 1, position + frames_after), position, -1)
    frame_idx = itertools.zip_longest(before_range, after_range)
    past_shift = past_shift
    future_shift = future_shift
    for before, after in frame_idx:
        if choice := (torch.rand(1) < 0.5):
            if before is not None:
                combined_tokens = torch.where(
                    masks[:, before] == 0,
                    tokens[:, before] + past_shift,
                    combined_tokens,
                )
            if after is not None:
                combined_tokens = torch.where(
                    masks[:, after] == 0,
                    tokens[:, after] + future_shift,
                    combined_tokens,
                )
        else:
            if after is not None:
                combined_tokens = torch.where(
                    masks[:, after] == 0,
                    tokens[:, after] + future_shift,
                    combined_tokens,
                )
            if before is not None:
                combined_tokens = torch.where(
                    masks[:, before] == 0,
                    tokens[:, before] + past_shift,
                    combined_tokens,
                )
        if before is not None:
            combined_masks = torch.where(masks[:, before] == 0,
                                         torch.tensor(0), combined_masks)
        if after is not None:
            combined_masks = torch.where(masks[:, after] == 0, torch.tensor(0),
                                         combined_masks)
    # Add the tokens from the current frame with highest priority.
    combined_tokens = torch.where(masks[:, position] == 0, tokens[:, position],
                                  combined_tokens)
    return combined_tokens, combined_masks


def subsample_tokens(
    tokens,
    mask_token=1024,
    type="random",
    keep_percentage=None,
    shift=None,
    labels=None,
    drop_label=None,
    random_shift=False,
):
    """
        Subsample tokens in a given tensor, using vmap to parallelize the process, wherever possible.
        :param: tokens: torch.Tensor: The tensor to subsample - patch_size x patch_size or bsize x patch_size x patch_size or bsize x context_size x patch_size x patch_size.
        """
    tokens = tokens.detach().clone()
    if not (tokens.dim() == 2 or tokens.dim() == 3 or tokens.dim() == 4):
        raise ValueError(
            f"Code should be 2D (psize x psize), 3D (bsize x psize x psize), or 4D (bsize x csize x psize x psize), not {tokens.dim()}D"
        )
    match type:
        case "random":
            match tokens.dim():
                case 4:
                    # Replicate keep_percentage along the context dimension.
                    keep_percentage = keep_percentage.unsqueeze(1).expand(
                        -1, tokens.size(1))
                    mask = torch.vmap(
                        torch.vmap(fully_random, randomness="different"),
                        randomness="different",
                    )(tokens, keep_percentage)
                case 3:
                    mask = torch.vmap(fully_random,
                                      randomness="different")(tokens,
                                                              keep_percentage)
                case 2:
                    mask = fully_random(tokens, keep_percentage)
        case "pattern":
            token_amount = tokens.size(-2) * tokens.size(-1)
            match tokens.dim():
                case 4:
                    if shift is None:
                        if random_shift:
                            # Create tensor of shape (bsize, csize) where in each batch, the shift is a range from 0 to csize-1 which is
                            # shifted to the right by a random amount and then modulo the number of tokens in the last two dimensions.
                            # Create random shift tensor.
                            offset = (torch.randint(
                                0, token_amount,
                                (tokens.size(0), )).unsqueeze(1).expand(
                                    -1, tokens.size(1)))
                            shift = (torch.arange(tokens.size(1)).unsqueeze(
                                0).expand(tokens.size(0), -1) +
                                     offset) % token_amount
                        else:
                            # Create tensor of shape (bsize, csize) where in each batch, the shift is the same tensor, which is a range from 0 to csize-1.
                            shift = (torch.arange(
                                tokens.size(1)).unsqueeze(0).expand(
                                    tokens.size(0), -1))
                    batch = []
                    for i in range(tokens.size(0)):
                        context = []
                        keep = keep_percentage[i]
                        for j in range(tokens.size(1)):
                            mask = pattern(
                                tokens[i, j],
                                keep_percentage=keep,
                                shift=shift[i, j],
                            )
                            context.append(mask)
                        batch.append(torch.stack(context, dim=0))
                    mask = torch.stack(batch, dim=0)
                case 3:
                    if shift is None:
                        if random_shift:
                            shift = torch.randint(low=0,
                                                  high=token_amount,
                                                  size=(tokens.size(0), ))
                        else:
                            shift = 0
                    batch = []
                    for i in range(tokens.size(0)):
                        mask = pattern(
                            tokens[i],
                            shift=shift[i],
                            keep_percentage=keep_percentage[i],
                        )
                        batch.append(mask)
                    mask = torch.stack(batch, dim=0)
                case 2:
                    if shift is None:
                        if random_shift:
                            shift = torch.randint(low=0,
                                                  high=token_amount,
                                                  size=(1, ))
                        else:
                            shift = 0
                    mask = pattern(tokens, shift, keep_percentage)
        case None:
            mask = torch.zeros_like(tokens, dtype=torch.int)
        case _:
            warnings.warn(
                f"Subsampling type {type} not recognized. No subsampling applied."
            )
            mask = torch.zeros_like(tokens, dtype=torch.int)
    masked_tokens = torch.where(
        mask.bool(),
        torch.tensor(mask_token, dtype=torch.int).to(tokens.device),
        tokens,
    )
    return masked_tokens, mask


def select_frames(tensor, num_frames):
    B, T, H, W = tensor.shape
    assert num_frames > 0 and num_frames <= T, "num_frames must be between 1 and T"

    # If num_frames is 1, select the middle frame
    if num_frames == 1:
        mid_frame_idx = (T - 1) // 2  # select the middle frame (8 if T=16)
        selected_frames = tensor[:, mid_frame_idx:mid_frame_idx + 1]
    else:
        # Generate evenly spaced indices
        indices = torch.linspace(0, T - 1, steps=num_frames).long()
        selected_frames = tensor[:, indices]

    return selected_frames


def get_selected_frame_indices(T, num_frames):
    assert num_frames > 0 and num_frames <= T, "num_frames must be between 1 and T"

    # If num_frames is 1, return the middle frame index
    if num_frames == 1:
        mid_frame_idx = (T - 1) // 2  # select the middle frame (8 if T=16)
        indices = torch.tensor([mid_frame_idx])
    else:
        # Generate evenly spaced indices
        indices = torch.linspace(0, T - 1, steps=num_frames).long()

    return indices


def create_tensor(T, H, W):
    # Create a tensor of shape (T, 1, 1) where each element contains its index
    index_tensor = torch.arange(T).view(T, 1, 1)

    # Expand the tensor to shape (T, H, W) by repeating the values across the H and W dimensions
    tensor = index_tensor.expand(T, H, W)

    return tensor


def prepend_separator(tensors, sep_token):
    """
    Add separator tokens before each frame. Flatten the sequence during the process.
    """
    B, T, H, W = tensors.shape
    tensors = tensors.flatten(start_dim=2)
    tensors = torch.cat([
        sep_token * torch.ones(
            (B, T, 1), dtype=tensors.dtype).to(tensors.device), tensors
    ],
                        dim=2)
    return tensors


def prepend_feature_token(tensors, feature_token):
    """
    Add feature tokens before each frame. Flatten the sequence during the process.
    """
    B, T, S = tensors.shape
    tensors = tensors.flatten(start_dim=1)
    tensors = torch.cat([
        feature_token * torch.ones(
            (B, 1), dtype=tensors.dtype).to(tensors.device), tensors
    ],
                        dim=1)
    return tensors


def test():
    # Test collate function.
    # Create 1 elements list of input tensors of shape 16, 16, 16 for x.
    # And 1 elements list of output tensors of shape 2304 for y.
    # Make each element of the T dimension have the same number (0 to 15).
    x = create_tensor(16, 16, 16)
    y = torch.arange(2304).reshape(2304)
    batch = [(x, y)]
    batch = test_collate_fn(batch,
                            mask_token=3072,
                            subsample_types=["random"],
                            context_before=0,
                            context_after=0,
                            num_frames=4,
                            keep_percentage=0.001)
    x, y = batch
    pass


if __name__ == "__main__":
    test()
