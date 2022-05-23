import torch
import torch.nn as nn

from models.common import *


# Code location is: root/utils/loss.py -> 140 lines
if ota:

    temps_s = b * 3 * 80 * 81 + a * 80 * 81 + gj * 81 + gi + 1  # 1228800 = 64 * 3 * 80 * 80
    temps_lst, temps_idx = temps_s.sort(dim=-1)

    temps_lst2 = torch.zeros_like(temps_lst, device=device)
    temps_lst2[1:] = temps_lst[:n-1]
    temps_lst = (temps_lst - temps_lst2) > 0
    head_lst = torch.ones_like(temps_lst, device=device)
    head_lst[:n-1] = temps_lst[1:]
    head_lst = ((temps_lst.int() - head_lst.int()) > 0).int()
    
    r_lst = torch.zeros_like(temps_lst, device=device)
    r_lst[:n-1] = (~temps_lst)[1:]
    r_lst = (~((temps_lst.int() - r_lst.int()) > 0)).int()

    al_lst = head_lst + r_lst

    al_lst_single = al_lst == 0
    al_lst_single = torch.nonzero(al_lst_single, as_tuple=True)

    al_lst_idx = torch.nonzero(al_lst, as_tuple=True)

    new_idx = temps_idx[al_lst_idx]

    bps = pi[b[new_idx], a[new_idx], gj[new_idx], gi[new_idx]]
    bpxy = bps[:, :2].sigmoid() * 2 - 0.5
    bpwh = (bps[:, 2:4].sigmoid() * 2) ** 2 * anc[new_idx]
    bpbox = torch.cat((bpxy, bpwh), 1)  # predicted box
    biou = bbox_iou(bpbox.T, tbx[new_idx], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
    cost_reg = (1.0 - biou)  # iou loss
    bt = torch.full_like(bps[:, 5:], self.cn, device=device)  # targets
    bt[range(len(new_idx)), tc[new_idx]] = self.cp
    cost_cls = self.BCEclsCost(bps[:, 5:], bt).sum(dim=-1)  # BCE

    # bo = torch.ones_like(bps[:, 4], device=device)
    # cost_obj = self.BCEobjCost(bps[:, 4], bo)  # BCE

    cost = cost_cls + cost_reg * 3. # + cost_obj * 3.

    al_s = 0
    final_idx = []
    al_total = al_lst[al_lst_idx].shape[0]
    for ali, al in enumerate(al_lst[al_lst_idx]):
        if al == 2:
            if ali != 0 :#and ali - al_s > 1:
                #print(1,biou[al_s:ali].shape, cost[al_s:ali].shape, ali - al_s)
                # if ali - al_s>1:
                _,bl_idx = biou[al_s:ali].topk(k=2)
                _,al_idx = cost[al_s:ali][bl_idx].min(dim = -1)
                final_idx.append(new_idx[al_s + bl_idx[al_idx]].item())
                # else:
                #     final_idx.append(new_idx[al_s : ali].item())
                al_s = ali
        elif ali == al_total-1 :#and al_total - al_s > 1:
            #print(2,biou[al_s:].shape,al_total - al_s)
            # if al_total - al_s>1:
            _,bl_idx = biou[al_s:].topk(k=2)
            _,al_idx = cost[al_s:][bl_idx].min(dim = -1)
            final_idx.append(new_idx[al_s + bl_idx[al_idx]].item())
            # else:
            #     final_idx.append(new_idx[al_s: ].item())
        # elif ali - al_s == 1 or al_total - al_s == 1:
        #     print(3,biou[al_s:ali].shape)
        #     final_idx.append(new_idx[ali].item())
        #     al_s = ali


    
    if len(final_idx)>0:
        try:
            final_idx = torch.tensor(final_idx, device=device)
            final_idx = torch.cat([final_idx,temps_idx[al_lst_single]],dim=-1)
            b, a, gj, gi = b[final_idx], a[final_idx], gj[final_idx], gi[final_idx]
            tbx, anc, tc, n = tbx[final_idx], anc[final_idx], tc[final_idx], final_idx.shape[0]
        except:
            print(final_idx)