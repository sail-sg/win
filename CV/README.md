# Win: Weight-Decay-Integrated Nesterov Acceleration for Adaptive Gradient Algorithms


For all vision tasks, we run them by using the official [`timm ` implementation](https://github.com/rwightman/pytorch-image-models). To reproduce our results, please first refer to [`timm`](https://github.com/rwightman/pytorch-image-models) and install it. Then you can follow the following two steps to reproduce our experiments in paper. 



## Environment

Our experiments for this task are based on the following pkg version.

```python
torch.__version__  = '1.10.0+cu113'
torchvision.__version__ = '0.11.1+cu113'
timm.__version__ = '0.6.1'
torchaudio.__version__ = '0.10.0+cu113'
```

Note that our timm is a developer version. If you want to strictly follow our environment, please refer to our released docker image [xyxie/adan-image:timm](https://hub.docker.com/repository/docker/xyxie/adan-image).



## Usage of Adan in timm

### Two steps to use Adan

**Step 1.** add Adan-dependent hyper-parameters by adding the following hyper-parameters to the `main_train.py`:

```python
parser.add_argument('--acceleration-mode', type=str, default="win",
                    help='whether use win or win2 or vanilla (no accleration) to accelerate optimizer, choices: win, win2, none')
parser.add_argument('--reckless-steps', default=(2.0, 8.0), type=float, nargs='+', 
                    help='two coefficients used as the multiples of the reckless stepsizes over the conservative stepsize in Win and Win2.  (default: (2.0, 8.0)')
parser.add_argument('--max-grad-norm', type=float, default=0.0,
                    help='Max grad norm (same as clip gradient norm, default: 0.0, no clipping)')
```
`reckless-steps`: two coefficients used as the multiples of the reckless stepsizes over the conservative stepsize in Win and Win2, namely $\gamma_1$ and $\gamma_2$ in [the extension paper](https://openreview.net/forum?id=CPdc77SQfQ5):
$$\eta_{k}^{\mathbf{y}} = \gamma_1 \eta_{k}^{\mathbf{x}}  \quad \text{and} \quad \eta_{k}^{\mathbf{z}} = \gamma_2 \eta_{k}^{\mathbf{x}}$$
In all experiments in our paper, we set $(\gamma_1, \gamma_2)=(2.0, 8.0)$.


**Step 2.** creat Win- or Win2-accelerated optimizer. In this step, we can add the new optimizer below the vanilla optimizer creator by using the following three substeps. 

i) Add Win- and Win2- accelerated AdamW, Adam, SGD and LAMB into `optim_factory.py` in `Timm`:

  ```python
from sgd_win import SGD_Win
from lamb_win import Lamb_Win
from adam_win import Adam_Win
from adamw_win import AdamW_Win 
  ```
  
 ii) Add the optimizers into `create_optimizer_v2` function in `optim_factory.py`:
  
  ```python
elif opt_lower == 'sgd-win':
    opt_args.pop('betas', None) 
    optimizer = SGD_Win(parameters, momentum=momentum, nesterov=True, **opt_args)
elif opt_lower == 'adam-win':
    optimizer = Adam_Win(parameters, **opt_args)
elif opt_lower == 'adamw-win':
    optimizer = AdamW_Win(parameters, **opt_args) 
elif opt_lower == 'lamb-win':
    optimizer = Lamb_Win(parameters, **opt_args)
        
  ```

iii) Import the optimizer creator into your training file `main_train.py` from `optim_factory` :

  ```python
  from optim_factory import create_optimizer
  ```

iv) Replace the vanilla creator (`optimizer = create_optimizer(args, model)`) in the training file `main_train.py` with Win- and Win2- accelerated AdamW or Adam or SGD or LAMB:

  ```python
opt_lower = args.opt.lower()
if opt_lower == 'sgd-win':
    args.opt_args = {'acceleration_mode': args.acceleration_mode,\
                        'reckless_steps': args.reckless_steps,\
                        'max_grad_norm': args.max_grad_norm}  

elif opt_lower == 'adam-win':
    args.opt_args = {'acceleration_mode': args.acceleration_mode,\
                        'reckless_steps': args.reckless_steps,\
                        'max_grad_norm': args.max_grad_norm}    

elif opt_lower == 'adamw-win':
    args.opt_args = {'acceleration_mode': args.acceleration_mode,\
                        'reckless_steps': args.reckless_steps,\
                        'max_grad_norm': args.max_grad_norm, }   
                
elif opt_lower == 'lamb-win':
    args.opt_args = {'acceleration_mode': args.acceleration_mode, \
                        'reckless_steps': args.reckless_steps,\
                        'max_grad_norm': args.max_grad_norm} 


optimizer = create_optimizer(args, model, filter_bias_and_bn = not args.bias_decay)
  ```



## Results, Configs and Logs on ImageNet-1K Dataset

For your convenience to use Win and Win2, we provide the configs and log files for the experiments on ImageNet-1k.
<div align="center">
<img width="80%" alt="Overall framework of Mugs. " src="../results/ResNet.png">
</div>

<div align="center">
<img width="80%" alt="Overall framework of Mugs. " src="../results/ViT.png">
</div>

Here we provide the training logs and configs under 300 training epochs 
<div align="center">

|      Model    |     ResNet50      |     ResNet101     |      ViT-small    |        ViT-base   |
| ------------- | :---------------: | :---------------: | :---------------: | :---------------: | 
| SGD-Win   |     ([config](./results/win_log/sgd/ResNet50_config.yaml), [log](./results/win_log/sgd/ResNet50_summary.csv))   |  ([config](./results/win_log/sgd/ResNet101_config.yaml), [log](./results/win_log/sgd/ResNet101_summary.csv)) | ([config](./results/win_log/sgd/ViT_S_config.yaml), [log](./results/win_log/sgd/ViT_S_summary.csv))   | ([config](./results/win_log/sgd/ViT_B_config.yaml), [log](./results/win_log/sgd/ViT_B_summary.csv)) |
| SGD-Win2   |     ([config](./results/win2_log/sgd/ResNet50_config.yaml), [log](./results/win2_log/sgd/ResNet50_summary.csv))   |  ([config](./results/win2_log/sgd/ResNet101_config.yaml), [log](./results/win2_log/sgd/ResNet101_summary.csv)) | ([config](./results/win2_log/sgd/ViT_S_config.yaml), [log](./results/win2_log/sgd/ViT_S_summary.csv))   | ([config](./results/win2_log/sgd/ViT_B_config.yaml), [log](./results/win2_log/sgd/ViT_B_summary.csv)) |
| Adam-Win   |     ([config](./results/win_log/adam/ResNet50_config.yaml), [log](./results/win_log/adam/ResNet50_summary.csv))   |  ([config](./results/win_log/adam/ResNet101_config.yaml), [log](./results/win_log/adam/ResNet101_summary.csv)) | ([config](./results/win_log/adam/ViT_S_config.yaml), [log](./results/win_log/adam/ViT_S_summary.csv))   | ([config](./results/win_log/adam/ViT_B_config.yaml), [log](./results/win_log/adam/ViT_B_summary.csv)) |
| Adam-Win2   |     ([config](./results/win2_log/adam/ResNet50_config.yaml), [log](./results/win2_log/adam/ResNet50_summary.csv))   |  ([config](./results/win2_log/adam/ResNet101_config.yaml), [log](./results/win2_log/adam/ResNet101_summary.csv)) | ([config](./results/win2_log/adam/ViT_S_config.yaml), [log](./results/win2_log/adam/ViT_S_summary.csv))   | ([config](./results/win2_log/adam/ViT_B_config.yaml), [log](./results/win2_log/adam/ViT_B_summary.csv)) |
| AdamW-Win   |     ([config](./results/win_log/adamw/ResNet50_config.yaml), [log](./results/win_log/adam/ResNet50_summary.csv))   |  ([config](./results/win_log/adamw/ResNet101_config.yaml), [log](./results/win_log/adamw/ResNet101_summary.csv)) | ([config](./results/win_log/adamw/ViT_S_config.yaml), [log](./results/win_log/adamw/ViT_S_summary.csv))   | ([config](./results/win_log/adamw/ViT_B_config.yaml), [log](./results/win_log/adamw/ViT_B_summary.csv)) |
| AdamW-Win2   |     ([config](./results/win2_log/adamw/ResNet50_config.yaml), [log](./results/win2_log/adam/ResNet50_summary.csv))   |  ([config](./results/win2_log/adamw/ResNet101_config.yaml), [log](./results/win2_log/adamw/ResNet101_summary.csv)) | ([config](./results/win2_log/adamw/ViT_S_config.yaml), [log](./results/win2_log/adamw/ViT_S_summary.csv))   | ([config](./results/win2_log/adamw/ViT_B_config.yaml), [log](./results/win2_log/adamw/ViT_B_summary.csv)) |
| LAMB-Win   |     ([config](./results/win_log/lamb/ResNet50_config.yaml), [log](./results/win_log/lamb/ResNet50_summary.csv))   |  ([config](./results/win_log/lamb/ResNet101_config.yaml), [log](./results/win_log/lamb/ResNet101_summary.csv)) | ([config](./results/win_log/lamb/ViT_S_config.yaml), [log](./results/win_log/lamb/ViT_S_summary.csv))   | ([config](./results/win_log/lamb/ViT_B_config.yaml), [log](./results/win_log/lamb/ViT_B_summary.csv)) | 
| LAMB-Win2   |     ([config](./results/win2_log/lamb/ResNet50_config.yaml), [log](./results/win2_log/lamb/ResNet50_summary.csv))   |  ([config](./results/win2_log/lamb/ResNet101_config.yaml), [log](./results/win2_log/lamb/ResNet101_summary.csv)) | ([config](./results/win2_log/lamb/ViT_S_config.yaml), [log](./results/win2_log/lamb/ViT_S_summary.csv))   | ([config](./results/win2_log/lamb/ViT_B_config.yaml), [log](./results/win2_log/lamb/ViT_B_summary.csv)) | 
 </div>
