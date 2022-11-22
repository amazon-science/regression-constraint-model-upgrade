import torch
from torch.autograd import Function
import torch.distributed as dist


def all_gather(tensor, group=dist.group.WORLD):
    """
    Gathers tensors from the whole group in a list.
    Arguments:
        tensor (Tensor): Tensor to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on.
    Returns:
        tuple[Tensor]): Output of the collective.
    """
    return _AllGather.apply(group, tensor)


class _AllGather(Function):
    @staticmethod
    def forward(ctx, group, tensor):
        ctx.group = group
        out_tensor_list = [
            torch.empty_like(tensor) for i in range(dist.get_world_size(group=group))
        ]
        dist.all_gather(out_tensor_list, tensor, group=group)
        return tuple(out_tensor_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        # gxs = _AlltoAll.apply(ctx.group, *grad_outputs)
        # gx = torch.sum(torch.stack(gxs), dim=0)

        # WARNING: this only works when all processes run the same loss function on the same data.
        # Use the canonical operation commented above for all other cases.
        scale_factor = dist.get_world_size(group=ctx.group)
        gx = grad_outputs[dist.get_rank()] * scale_factor
        return (None, gx)


class _AlltoAll(Function):
    @staticmethod
    def forward(ctx, group, *tensors):
        ctx.group = group
        out_tensor_list = [
            torch.empty_like(tensors[i]) for i in range(dist.get_world_size(group=group))
        ]
        reqs = [None] * dist.get_world_size(group=group)
        my_rank = dist.get_rank(group=group)
        # Implement it on means of scatter/gather, send/recv async operations have issues
        if dist.get_backend(group=group) is dist.Backend.GLOO:
            for i in range(dist.get_world_size(group=group)):
                to_send = None
                if i == my_rank:
                    to_send = list(tensors)
                dist.scatter(out_tensor_list[i], to_send, i, group=group)
        else:
            dist.all_to_all(out_tensor_list, list(tensors), group=group)
        return tuple(out_tensor_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None,) + _AlltoAll.apply(ctx.group, *grad_outputs)
