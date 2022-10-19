# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from numpy.core.numeric import full
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from pathlib import Path
from torchvision.utils import save_image
import time
import copy
import torchvision.transforms as T

epsilon = 1e-8
def do_nothing(x): return x

class Trainer():
    def __init__(self, eval_freq):
        self.eval_freq = eval_freq

    def update(self, model, batch, step, eval=False):
        t0 = time.time()
        metrics = dict()
        if eval:
            model.eval()
        else:
            model.train()

        t1 = time.time()
        ## Batch
        b_im, b_reward = batch
        t2 = time.time()

        ## Encode Start and End Frames
        bs = b_im.shape[0]
        img_stack_size = b_im.shape[1]
        H = b_im.shape[-2]
        W = b_im.shape[-1]
        b_im_r = b_im.reshape(bs*img_stack_size, 3, H, W)
        alles = model(b_im_r)
        alle = alles.reshape(bs, img_stack_size, -1)
        e0 = alle[:, 0] # initial, o_0
        eg = alle[:, 1] # final, o_g
        es0_vip = alle[:, 2] # o_t
        es1_vip = alle[:, 3] # o_t+1

        full_loss = 0

        # Compute alignment as a metric
        if alle.shape[1] > 4:
            with torch.no_grad():
                es0 = alle[:, -3] 
                es1 = alle[:, -2]
                es2 = alle[:, -1]
                sim_0_2 = model.module.sim(es2, es0) 
                sim_1_2 = model.module.sim(es2, es1)
                sim_0_1 = model.module.sim(es1, es0)
                a_state = ((1.0 * (sim_0_2 < sim_1_2)) * (1.0 * (sim_0_1 > sim_0_2))).mean()
                metrics['aligned'] = a_state.item()

        ## LP Loss
        l2loss = torch.linalg.norm(alles, ord=2, dim=-1).mean()
        l1loss = torch.linalg.norm(alles, ord=1, dim=-1).mean()
        metrics['l2loss'] = l2loss.item()
        metrics['l1loss'] = l1loss.item()
        full_loss += model.module.lweight * l2loss
        full_loss += model.module.lweight * l1loss
        t3 = time.time()

        ## VIP Loss 
        V_0 = model.module.sim(e0, eg) # -||phi(s) - phi(g)||_2
        r =  b_reward.to(V_0.device) # R(s;g) = (s==g) - 1 
        V_s = model.module.sim(es0_vip, eg)
        V_s_next = model.module.sim(es1_vip, eg)
        V_loss = (1-model.module.gamma) * -V_0.mean() + torch.log(epsilon + torch.mean(torch.exp(-(r + model.module.gamma * V_s_next - V_s))))

        # Optionally, add additional "negative" observations
        V_s_neg = []
        V_s_next_neg = []
        for _ in range(model.module.num_negatives):
            perm = torch.randperm(es0_vip.size()[0])
            es0_vip_shuf = es0_vip[perm]
            es1_vip_shuf = es1_vip[perm]

            V_s_neg.append(model.module.sim(es0_vip_shuf, eg))
            V_s_next_neg.append(model.module.sim(es1_vip_shuf, eg))

        if model.module.num_negatives > 0:
            V_s_neg = torch.cat(V_s_neg)
            V_s_next_neg = torch.cat(V_s_next_neg)
            r_neg = -torch.ones(V_s_neg.shape).to(V_0.device)
            V_loss = V_loss + torch.log(epsilon + torch.mean(torch.exp(-(r_neg + model.module.gamma * V_s_next_neg - V_s_neg))))
        
        metrics['vip_loss'] = V_loss.item()
        full_loss += V_loss
        metrics['full_loss'] = full_loss.item()
        t4 = time.time()

        if not eval:
            model.module.encoder_opt.zero_grad()
            full_loss.backward()
            model.module.encoder_opt.step()
        t5 = time.time()    

        st = f"Load time {t1-t0}, Batch time {t2-t1}, Encode and LP time {t3-t2}, VIP time {t4-t3}, Backprop time {t5-t4}"
        return metrics,st
