import numpy as np
import torch

from . import nms_cpu, nms_cuda
from .cas_nms_cpu import cas_nms_cpu


def nms(dets, iou_thr, device_id=None):
    # convert dets (tensor or numpy array) to tensor
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else 'cuda:{}'.format(device_id)
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    # execute cpu or cuda nms
    if dets_th.shape[0] == 0:
        inds = dets_th.new_zeros(0, dtype=torch.long)
    else:
        if dets_th.is_cuda:
            inds = nms_cuda.nms(dets_th, iou_thr)
        else:
            inds = nms_cpu.nms(dets_th, iou_thr)

    if is_numpy:
        inds = inds.cpu().numpy()
    return dets[inds, :], inds


def nms_cpus(dets, iou_thr, device_id=None):
    # convert dets (tensor or numpy array) to tensor
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else 'cuda:{}'.format(device_id)
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    inds = nms_cpu.nms(dets_th.cpu(), iou_thr)

    if is_numpy:
        inds = inds.cpu().numpy()
    return dets[inds, :], inds


def cas_nms(dets, count_pred, similarity_matrix, thresh=0.5, t_2=1.5, t_1=1.0, N_st=0.5):
    dets_np = dets.detach().cpu().float().numpy()
    count_np = count_pred.detach().cpu().float().numpy()
    similarity_matrix_np = similarity_matrix.detach().cpu().float().numpy()
    inds = cas_nms_cpu(dets_np, count_np, similarity_matrix_np, thresh, t_2, t_1, N_st)
    return dets[inds], np.array(inds, dtype=np.int)
