# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.optim.optimizer import Optimizer, required


class SGD_Win(Optimizer):
    r"""Implements Win- and Win2-accelerated SGD algorithm.

    SGD Optimizer Implementation refered from https://github.com/clovaai/AdamP/blob/master/adamp/sgdp.py

    Paper: `Slowing Down the Weight Norm Increase in Momentum-based Optimizers` - https://arxiv.org/abs/2006.08217

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        momentum (float, optional): the coefficient used for computing
            running averages of gradient (default: 0.9)
        dampening (float, optional): the  coefficient used for computing
            running averages of gradient (default: 0.0)
        reckless_steps (Tuple[float, float], optional): two coefficients used as the multiples
            of the reckless stepsizes over the conservative stepsize in Win and Win2 (default: (2.0, 8.0))
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        max_grad_norm (float, optional): value used to clip global grad norm (default: 0.0, no gradient clip)
        nesterov (bool, optional): hether SGD optimizer uses nesterov to update its gradient, (default: True)
        acceleration_mode (string, optional): win or win2 or none (vanilla AdamW)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
    """
    def __init__(self, params, lr=required, momentum=0.9, dampening=0, reckless_steps=(2.0, 8.0),
                 weight_decay=0.0, max_grad_norm=0.0, nesterov=True, acceleration_mode='win', eps=1e-8):
        defaults = dict(
            lr=lr, momentum=momentum, dampening=dampening, reckless_steps=reckless_steps, 
            weight_decay=weight_decay, max_grad_norm=max_grad_norm,
            acceleration_mode=acceleration_mode, nesterov=nesterov, eps=eps)
        if reckless_steps[0] < 0.0:
            raise ValueError("Invalid reckless_steps parameter at index 0: {}".format(reckless_steps[0]))
        if reckless_steps[1] < 0.0:
            raise ValueError("Invalid reckless_steps parameter at index 1: {}".format(reckless_steps[1]))
        
        super(SGD_Win, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        ## whether perform gradient clip 
        if self.defaults['max_grad_norm'] > 1e-8:
            device = self.param_groups[0]['params'][0].device
            global_grad_norm = torch.zeros(1, device=device)
            
            max_grad_norm = torch.tensor(self.defaults['max_grad_norm'], device=device)
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        grad = p.grad 
                        global_grad_norm.add_(grad.pow(2).sum())

            global_grad_norm = torch.sqrt(global_grad_norm)

            clip_global_grad_norm = torch.clamp(max_grad_norm /(global_grad_norm + group['eps']) , max=1.0)
        
        # parameter update
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum  = group['momentum']
            dampening = group['dampening']
            nesterov_update_mode = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform optimization step
                if self.defaults['max_grad_norm'] > 1e-8:
                    grad = p.grad.mul_(clip_global_grad_norm)
                else:
                    grad = p.grad

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    if group['acceleration_mode'] == 'win':
                        state['x'] = torch.zeros_like(p) 
                        state['x'].add_(p.data.clone(), alpha=1)  
                    elif group['acceleration_mode'] == 'win2':
                        state['x'] = torch.zeros_like(p) 
                        state['x'].add_(p.data.clone(), alpha=1)  
                        state['y'] = torch.zeros_like(p) 
                        state['y'].add_(p.data.clone(), alpha=1)  

                    state['momentum'] = torch.zeros_like(p)
                    state['step'] = 0

                ## Win and Win2 acceleration for parameter update 
                if 'win' in group['acceleration_mode']:
                    beta1, beta2 = group['reckless_steps']

                    buf = state['momentum']
                    if state['step'] == 0:
                        buf.add_(grad, alpha=1.)
                    else:
                        buf.mul_(momentum).add_(grad, alpha=1. - dampening)

                    if nesterov_update_mode:
                        update = grad + momentum * buf 
                    else:
                        update = buf 

                    ## update x
                    lr_x = group['lr']
                    state['x'].add_(update, alpha= -lr_x) 
                    state['x'].data.mul_(1.0 / (1 + lr_x * group['weight_decay']))
  

                    lr_y = beta1 * group['lr']
                    gamma = 1.0 / (1.0 + lr_y / lr_x  + lr_y * group['weight_decay'])
                    if group['acceleration_mode'] == 'win':
                        ## update y 
                        p.mul_(gamma).add_(state['x'], alpha = (lr_y / lr_x) * gamma).add_(update, alpha= - lr_y * gamma) 

                    elif group['acceleration_mode'] == 'win2':
                        ## update y 
                        state['y'].data.mul_(gamma).add_(state['x'], alpha = (lr_y / lr_x) * gamma).add_(update, alpha= - lr_y * gamma) 

                        ## update z
                        lr_z = beta2 * group['lr']
                        gamma = 1.0 / (1.0 + lr_z / lr_x + lr_z / lr_y  + lr_z * group['weight_decay'])
                        p.mul_(gamma).add_(update, alpha= - lr_z * gamma) 
                        p.add_(state['x'], alpha = (lr_z / lr_x) * gamma).add_(state['y'], alpha = (lr_z / lr_y) * gamma)

                else: ## vanilla SGD optimizer
                    # Weight decay
                    if weight_decay != 0: 
                        grad.add_(weight_decay, p)
                    
                    buf = state['momentum']
                    if state['step'] == 0:
                        buf.add_(grad, alpha=1.)
                    else:
                        buf.mul_(momentum).add_(grad, alpha=1. - dampening)
                    # Step
                    if nesterov_update_mode:
                        p.add_(grad + momentum * buf, alpha=-group['lr'])
                    else:
                        p.add_(buf, alpha=-group['lr'])
                
                state['step'] += 1

        return loss
