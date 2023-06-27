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

import math
import torch
from torch.optim.optimizer import Optimizer

class Adam_Win(Optimizer):
    r"""Implements Win- and Win2-accelerated Adam algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_. 

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): two coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        reckless_steps (Tuple[float, float], optional): two coefficients used as the multiples
            of the reckless stepsizes over the conservative stepsize in Win and Win2 (default: (2.0, 8.0))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        max_grad_norm (float, optional): value used to clip global grad norm (default: 0.0, no gradient clip)
        acceleration_mode (string, optional): win or win2 or none (vanilla AdamW)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), reckless_steps=(2.0, 8.0), eps=1e-8,
                 weight_decay=0, amsgrad=False, max_grad_norm=0.0, acceleration_mode=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if reckless_steps[0] < 0.0:
            raise ValueError("Invalid reckless_steps parameter at index 0: {}".format(reckless_steps[0]))
        if reckless_steps[1] < 0.0:
            raise ValueError("Invalid reckless_steps parameter at index 1: {}".format(reckless_steps[1]))

        defaults = dict(lr=lr, betas=betas, reckless_steps=reckless_steps, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, 
                        acceleration_mode=acceleration_mode, max_grad_norm=max_grad_norm)
        super(Adam_Win, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam_Win, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
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
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform optimization step
                if self.defaults['max_grad_norm'] > 1e-8:
                    grad = p.grad.mul_(clip_global_grad_norm)
                else:
                    grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                    # extra variables for acceleration 
                    if group['acceleration_mode'] == 'win':
                        state['x'] = torch.zeros_like(p) 
                        state['x'].add_(p.data.clone(), alpha=1)  
                    elif group['acceleration_mode'] == 'win2':
                        state['x'] = torch.zeros_like(p) 
                        state['x'].add_(p.data.clone(), alpha=1)  
                        state['y'] = torch.zeros_like(p) 
                        state['y'].add_(p.data.clone(), alpha=1) 

                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']


                ## Win and Win2 acceleration for parameter update 
                if 'win' in group['acceleration_mode']:
                    beta3, beta4 = group['reckless_steps']

                    ## compute parameter update 
                    update = (exp_avg / denom) * (math.sqrt(bias_correction2) / bias_correction1)

                    ## update x
                    lr_x = group['lr']
                    state['x'].add_(update, alpha= -lr_x)
                    

                    lr_y = beta3 * group['lr']
                    gamma = 1.0 / (1.0 + lr_y / lr_x)
                    if group['acceleration_mode'] == 'win':
                        ## update y 
                        p.data.mul_(gamma).add_(state['x'], alpha = (lr_y / lr_x) * gamma).add_(update, alpha= - lr_y * gamma) 

                    elif group['acceleration_mode'] == 'win2':
                        ## update y 
                        state['y'].data.mul_(gamma).add_(state['x'], alpha = (lr_y / lr_x) * gamma).add_(update, alpha= - lr_y * gamma) 

                        ## update z
                        lr_z = beta4 * group['lr']
                        gamma = 1.0 / (1.0 + lr_z / lr_x + lr_z / lr_y)
                        p.data.mul_(gamma).add_(update, alpha= - lr_z * gamma) 
                        p.data.add_(state['x'], alpha = (lr_z / lr_x) * gamma).add_(state['y'], alpha = (lr_z / lr_y) * gamma)
                
                else: ## vanilla Adam optimizer
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                    p.data.addcdiv_(-step_size, exp_avg, denom) 

        return loss