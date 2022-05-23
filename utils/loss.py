# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

from ast import excepthandler
import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel

import time


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        self.BCEclsCost = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device), reduction="none")
        self.BCEobjCost = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device), reduction="none")

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        ota = True

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            tbx = tbox[i]
            anc = anchors[i]
            tc = tcls[i]

            n = b.shape[0]  # number of targets
            if n:
                if ota:
                    # for _, bv in temp_s.items():
                    #     if len(bv) >  1:
                    #         bps = pi[b[bv], a[bv], gj[bv], gi[bv]]
                    #         bpxy = bps[:, :2].sigmoid() * 2 - 0.5
                    #         bpwh = (bps[:, 2:4].sigmoid() * 2) ** 2 * anc[bv]
                    #         bpbox = torch.cat((bpxy, bpwh), 1)  # predicted box
                    #         biou = bbox_iou(bpbox.T, tbx[bv], x1y1x2y2=False)  # iou(prediction, target)
                    #         cost_reg = (1.0 - biou)  # iou loss
                    #         bt = torch.full_like(bps[:, 5:], self.cn, device=device)  # targets
                    #         bt[range(len(bv)), tc[bv]] = self.cp
                    #         cost_cls = self.BCEcls(bps[:, 5:], bt)  # BCE
                    #         cost = cost_cls + cost_reg * 3.
                    #         _, c_idx = cost.min(dim=-1)
                    #         new_idx.append(bv[c_idx])
                    #     else:
                    #         new_idx.append(bv[0])
                    # temp_s = {}
                    # new_odr = {}
                    # final_idx = []
                    
                    # for idx, _ in enumerate(b):
                    #     b_s = str(b[idx].item()) + "_" + str(a[idx].item()) + "_" + str(gj[idx].item()) + "_" + str(gi[idx].item())
                    #     if idx !=0:
                    #         if b_s in temp_s:
                    #             temp_s[b_s].append(idx)
                    #             temp_blst = temp_s[b_s]
                    #             new_odr[b_s] = (len(temp_blst), temp_blst)
                    #         else:
                    #             temp_s[b_s] = [idx]
                    #             final_idx.append(idx)
                    
                    # del temp_s
                    
                    # # new_odr = []
                    # new_idx = []
                    # for _, bv in new_odr.items():
                    #     new_idx += bv[1]
                    
                    # bps = pi[b[new_idx], a[new_idx], gj[new_idx], gi[new_idx]]
                    # bpxy = bps[:, :2].sigmoid() * 2 - 0.5
                    # bpwh = (bps[:, 2:4].sigmoid() * 2) ** 2 * anc[new_idx]
                    # bpbox = torch.cat((bpxy, bpwh), 1)  # predicted box
                    # biou = bbox_iou(bpbox.T, tbx[new_idx], x1y1x2y2=False)  # iou(prediction, target)
                    # cost_reg = (1.0 - biou)  # iou loss
                    # bt = torch.full_like(bps[:, 5:], self.cn, device=device)  # targets
                    # bt[range(len(new_idx)), tc[new_idx]] = self.cp
                    # cost_cls = self.BCEcls(bps[:, 5:], bt)  # BCE
                    # cost = cost_cls + cost_reg * 3.
                    # temp_last = 0

                    # for _, ov in new_odr.items():
                    #     _, cidx = cost[temp_last: temp_last + ov[0]].min(dim=-1)
                    #     final_idx.append(ov[1][cidx])
                    #     temp_last = ov[0]

                    # del new_odr
                    # print(temp_s)
                    # final_idx = [x[0] for x in temp_s.values()]

                    # t1 = time.time()

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
            
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anc
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbx, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tc] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

            
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        # gain shape 7,
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        # ai shape (3, 509) , 509 is number of labels
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
  
        # targets shape (3, 509, 7) dim -1 is 6 labels and 1 anchor index
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        dpse = False

        g = 0.5  # bias
        off = None
        if dpse:
            off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            [0.7, 0.7], [-0.7, 0.7], [-0.7, -0.7], [0.7, -0.7],  # jk,kl,lm, mj
                            ], device=targets.device).float() * g  # offsets
        else:
            # off 5, 2
            off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            # anchors shape (3, 2)
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors , box size is based on grid
            t = targets * gain
            wh_thr = 1.0
            if nt:
                # Matches 
                # t[:, :, 4:6] shape (3, 509, 2)  anchors[:, None] shpae (3, 1, 2)
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio 
                # j shape (3, 509) , the mask of anchor matched with box , 这里 j 是选择哪个anchor来预测的蒙版
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                # filter the gt by the mask of j, t shape (n, 7), n is the number of matched gt
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gwh = t[:, 4:6]

                gxi = gain[[2, 3]] - gxy  # inverse

                if dpse:
                    # wh_thr = 1
                    j, k = ((gxy % 1. < 0.5) & (gxy > 1.) ).T  # & (gwh > wh_thr) | ((gxy > 1.) & (gwh >= wh_thr * 3)) 
                    l, m = ((gxi % 1. < 0.5) & (gxi > 1.) ).T  # & (gwh >= wh_thr ) #| ((gxi > 1.) & (gwh >= wh_thr * 3))
                    j_s, k_s = ( (gxy % 1. < 0.35) & (gxy > 1.) ).T  # & (gwh >= wh_thr * 2) #| ((gxy > 1.) & (gwh >= wh_thr * 4))
                    l_s, m_s = ( (gxi % 1. < 0.35) & (gxi > 1.) ).T  #  & (gwh >= wh_thr * 2) # | ((gxi > 1.) & (gwh >= wh_thr * 4))

                    o1, p1 = j_s & k_s, k_s & l_s
                    q1, s1 = l_s & m_s, m_s & j_s

                    # 把one和j, k, l, m连接 shape(5,?)
                    j = torch.stack((torch.ones_like(j), j, k, l, m, o1, p1, q1, s1))
                    
                    # 选出所有合适的正样本 shape(?, 7)
                    t = t.repeat((9, 1, 1))[j]
                else:
                    j, k = ((gxy % 1 < g) & (gxy > 1)).T
                    l, m = ((gxi % 1 < g) & (gxi > 1)).T
                    j = torch.stack((torch.ones_like(j), j, k, l, m))
                    # t shape (n, 7) n is the number of matched gt, 2524
                    t = t.repeat((5, 1, 1))[j]
                
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j] 
            else:
                t = targets[0]
                offsets = 0

            # Define t shape (n, 7)
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
