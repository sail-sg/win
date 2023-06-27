# Win: Weight-Decay-Integrated Nesterov Acceleration for Adaptive Gradient Algorithms

This is an official PyTorch implementation of **Win**. See the [ICLR paper](https://openreview.net/forum?id=CPdc77SQfQ5) and its [extension](https://openreview.net/forum?id=CPdc77SQfQ5). If you find our Win helpful or heuristic to your projects, please cite this paper and also star this repository. Thanks!

```tex
@inproceedings{zhou2022win,
  title={Win: Weight-Decay-Integrated Nesterov Acceleration for Adaptive Gradient Algorithms},
  author={Zhou, Pan and Xie, Xingyu and Yan, Shuicheng},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```
 
## Usage

For your convenience to use Win, we briefly provide some intuitive instructions below, then provide some general experimental tips, and finally provide more details (e.g., specific commands and hyper-parameters) for the main experiment in the paper.

#### 1) Two steps to use Win and Win2
Win can be integrated with most optimizers, including SoTA Adan, Adam, AdamW, LAMB and SGD by using the following steps.

**Step 1.** add Win- and Win2-dependent hyper-parameters by adding the following hyper-parameters to the config:

```python
parser.add_argument('--acceleration-mode', type=str, default="win",
                    help='whether use win or win2 or vanilla (no acceleration) to accelerate optimizer, choices: win, win2, none')
parser.add_argument('--reckless-steps', default=(2.0, 8.0), type=float, nargs='+', 
                    help='two coefficients used as the multiples of the reckless stepsizes over the conservative stepsize in Win and Win2.  (default: (2.0, 8.0)')
parser.add_argument('--max-grad-norm', type=float, default=0.0,
                    help='Max grad norm (same as clip gradient norm, default: 0.0, no clipping)')
```
`reckless-steps`: two coefficients used as the multiples of the reckless stepsizes over the conservative stepsize in Win and Win2, namely $\gamma_1$ and $\gamma_2$ in [the extension paper](https://openreview.net/forum?id=CPdc77SQfQ5):
$$\eta_{k}^{\mathbf{y}} = \gamma_1 \eta_{k}^{\mathbf{x}}  \quad \text{and} \quad \eta_{k}^{\mathbf{z}} = \gamma_2 \eta_{k}^{\mathbf{x}}$$
In all experiments in our paper, we set $(\gamma_1, \gamma_2)=(2.0, 8.0)$.

**Step 2.** create corresponding vanilla or Win- or Win2-accelerated optimizer as follows. In this step, we can directly replace the vanilla optimizer by using the following command:

```python
from adamw_win import AdamW_Win
optimizer = AdamW_Win(params, lr=args.lr, betas=(0.9, 0.999), reckless_steps=args.reckless_steps, weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm, acceleration_mode=args.acceleration_mode)

from adam_win import Adam_Win
optimizer = Adam_Win(params, lr=args.lr, betas=(0.9, 0.999), reckless_steps=args.reckless_steps, weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm, acceleration_mode=args.acceleration_mode) 

from lamb_win import LAMB_Win
optimizer = LAMB_Win(params, lr=args.lr, betas=(0.9, 0.999), reckless_steps=args.reckless_steps, weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm, acceleration_mode=args.acceleration_mode) 

from sgd_win import SGD_Win
optimizer = SGD_Win(params, lr=args.lr, momentum=0.9, reckless_steps=args.reckless_steps, weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm, nesterov=True, acceleration_mode=args.acceleration_mode) 
```

#### 2) Tips for Experiments

-  For **hyper-parameters**, to make Win and Win2 simple, in all experiments, we tune the vanilla hyper-parameters, e.g. stepsize, weight decay and warmup, around the vanilla values used in the vanilla optimizer. 
- For **robustness**, in most cases, Win and Win2-accelerated optimizers are much more robust to hyper-parameters, e.g. stepsize, weight decay and warmup, than vanilla optimizers, especially on large models. For example, on the ViT-base model, official AdamW would fail and collapse when the default stepsize becomes slightly larger, e.g. 2$\times$-larger, while Win and Win2-accelerated optimizers can tolerate this big stepsize and still achieve very good performance.
- For **GPU memory and computational ahead**, Win and Win2-accelerated optimizers do not bring big extra GPU memory cost and also computational ahead as shown in the paper. This is because, for most models, the BP process indeed dominates the GPU memory cost, e.g. storing features of all layers to compute the gradient, which is much higher than the memory cost caused by the model parameter itself. But for huge models, they may bring extra memory costs, since they need to maintain one or two extra sequences which are at the same size as the vanilla model parameter. However, this problem can be solved using the [ZeroRedundancyOptimizer](https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html), which shares optimizer states across distributed data-parallel processes to reduce per-process memory footprint.  


## Model Zoo

### Results on vision tasks

For your convenience to use Win and Win2, we provide the configs and log files for the experiments on ImageNet-1k.
<div align="center">
<img width="80%" alt="Overall framework of Mugs. " src="./results/ResNet.png">
</div>

<div align="center">
<img width="80%" alt="Overall framework of Mugs. " src="./results/ViT.png">
</div>

Here we provide the training logs and configs under 300 training epochs 
<div align="center">
  
|      Model    |     ResNet50      |     ResNet101     |      ViT-small    |        ViT-base   |
| ------------- | :---------------: | :---------------: | :---------------: | :---------------: | 
| SGD-Win   |     ([config](./CV/results/win_log/sgd/ResNet50_config.yaml), [log](./CV/results/win_log/sgd/ResNet50_summary.csv))   |  ([config](./CV/results/win_log/sgd/ResNet101_config.yaml), [log](./CV/results/win_log/sgd/ResNet101_summary.csv)) | ([config](./CV/results/win_log/sgd/ViT_S_config.yaml), [log](./CV/results/win_log/sgd/ViT_S_summary.csv))   | ([config](./CV/results/win_log/sgd/ViT_B_config.yaml), [log](./CV/results/win_log/sgd/ViT_B_summary.csv)) |
| SGD-Win2   |     ([config](./CV/results/win2_log/sgd/ResNet50_config.yaml), [log](./CV/results/win2_log/sgd/ResNet50_summary.csv))   |  ([config](./CV/results/win2_log/sgd/ResNet101_config.yaml), [log](./CV/results/win2_log/sgd/ResNet101_summary.csv)) | ([config](./CV/results/win2_log/sgd/ViT_S_config.yaml), [log](./CV/results/win2_log/sgd/ViT_S_summary.csv))   | ([config](./CV/results/win2_log/sgd/ViT_B_config.yaml), [log](./CV/results/win2_log/sgd/ViT_B_summary.csv)) |
| Adam-Win   |     ([config](./CV/results/win_log/adam/ResNet50_config.yaml), [log](./CV/results/win_log/adam/ResNet50_summary.csv))   |  ([config](./CV/results/win_log/adam/ResNet101_config.yaml), [log](./CV/results/win_log/adam/ResNet101_summary.csv)) | ([config](./CV/results/win_log/adam/ViT_S_config.yaml), [log](./CV/results/win_log/adam/ViT_S_summary.csv))   | ([config](./CV/results/win_log/adam/ViT_B_config.yaml), [log](./CV/results/win_log/adam/ViT_B_summary.csv)) |
| Adam-Win2   |     ([config](./CV/results/win2_log/adam/ResNet50_config.yaml), [log](./CV/results/win2_log/adam/ResNet50_summary.csv))   |  ([config](./CV/results/win2_log/adam/ResNet101_config.yaml), [log](./CV/results/win2_log/adam/ResNet101_summary.csv)) | ([config](./CV/results/win2_log/adam/ViT_S_config.yaml), [log](./CV/results/win2_log/adam/ViT_S_summary.csv))   | ([config](./CV/results/win2_log/adam/ViT_B_config.yaml), [log](./CV/results/win2_log/adam/ViT_B_summary.csv)) |
| AdamW-Win   |     ([config](./CV/results/win_log/adamw/ResNet50_config.yaml), [log](./CV/results/win_log/adam/ResNet50_summary.csv))   |  ([config](./CV/results/win_log/adamw/ResNet101_config.yaml), [log](./CV/results/win_log/adamw/ResNet101_summary.csv)) | ([config](./CV/results/win_log/adamw/ViT_S_config.yaml), [log](./CV/results/win_log/adamw/ViT_S_summary.csv))   | ([config](./CV/results/win_log/adamw/ViT_B_config.yaml), [log](./CV/results/win_log/adamw/ViT_B_summary.csv)) |
| AdamW-Win2   |     ([config](./CV/results/win2_log/adamw/ResNet50_config.yaml), [log](./CV/results/win2_log/adam/ResNet50_summary.csv))   |  ([config](./CV/results/win2_log/adamw/ResNet101_config.yaml), [log](./CV/results/win2_log/adamw/ResNet101_summary.csv)) | ([config](./CV/results/win2_log/adamw/ViT_S_config.yaml), [log](./CV/results/win2_log/adamw/ViT_S_summary.csv))   | ([config](./CV/results/win2_log/adamw/ViT_B_config.yaml), [log](./CV/results/win2_log/adamw/ViT_B_summary.csv)) |
| LAMB-Win   |     ([config](./CV/results/win_log/lamb/ResNet50_config.yaml), [log](./CV/results/win_log/lamb/ResNet50_summary.csv))   |  ([config](./CV/results/win_log/lamb/ResNet101_config.yaml), [log](./CV/results/win_log/lamb/ResNet101_summary.csv)) | ([config](./CV/results/win_log/lamb/ViT_S_config.yaml), [log](./CV/results/win_log/lamb/ViT_S_summary.csv))   | ([config](./CV/results/win_log/lamb/ViT_B_config.yaml), [log](./CV/results/win_log/lamb/ViT_B_summary.csv)) | 
| LAMB-Win2   |     ([config](./CV/results/win2_log/lamb/ResNet50_config.yaml), [log](./CV/results/win2_log/lamb/ResNet50_summary.csv))   |  ([config](./CV/results/win2_log/lamb/ResNet101_config.yaml), [log](./CV/results/win2_log/lamb/ResNet101_summary.csv)) | ([config](./CV/results/win2_log/lamb/ViT_S_config.yaml), [log](./CV/results/win2_log/lamb/ViT_S_summary.csv))   | ([config](./CV/results/win2_log/lamb/ViT_B_config.yaml), [log](./CV/results/win2_log/lamb/ViT_B_summary.csv)) | 
 </div>
   
### Results on NLP tasks
We will release them soon.
<!-- 
#### BERT-base

We give the configs and log files of the BERT-base model pre-trained on the Bookcorpus and Wikipedia datasets and fine-tuned on GLUE tasks. Note that we provide the config, log file, and detailed [instructions](./NLP/BERT/README.md) for BERT-base in the folder `./NLP/BERT`.
<div align="center">

| Pretraining |                         Config                         | Batch Size |                             Log                             | Model |
| ----------- | :----------------------------------------------------: | :--------: | :---------------------------------------------------------: | :---: |
| Adan        | [config](./NLP/BERT/config/pretraining/bert-adan.yaml) |    256     | [log](./NLP/BERT/exp_results/pretrain/hydra_train-adan.log) | model |

| Fine-tuning on GLUE-Task | Metric                       |  Result   |                         Config                         |
| ------------------------ | :--------------------------- | :-------: | :----------------------------------------------------: |
| CoLA                     | Matthew's corr.              |   64.6    | [config](./NLP/BERT/config/finetuning/cola-adan.yaml)  |
| SST-2                    | Accuracy                     |   93.2    | [config](./NLP/BERT/config/finetuning/sst_2-adan.yaml) |
| STS-B                    | Person corr.                 |   89.3    | [config](./NLP/BERT/config/finetuning/sts_b-adan.yaml) |
| QQP                      | Accuracy                     |   91.2    |  [config](./NLP/BERT/config/finetuning/qqp-adan.yaml)  |
| MNLI                     | Matched acc./Mismatched acc. | 85.7/85.6 | [config](./NLP/BERT/config/finetuning/mnli-adan.yaml)  |
| QNLI                     | Accuracy                     |   91.3    | [config](./NLP/BERT/config/finetuning/qnli-adan.yaml)  |
| RTE                      | Accuracy                     |   73.3    |  [config](./NLP/BERT/config/finetuning/rte-adan.yaml)  |
 </div>
 
For fine-tuning on GLUE-Task, see the total batch size in their corresponding configure files.

#### Transformer-XL-base

We provide the config and log for Transformer-XL-base trained on the WikiText-103 dataset. The total batch size for this experiment is `60*4`.
<div align="center">

|                     | Steps | Test PPL |                          Download                           |
| ------------------- | :---: | :------: | :---------------------------------------------------------: |
| Baseline (Adam)     | 200k  |   24.2   | [log&config](./NLP/Transformer-XL/exp_results/log-adam.txt) |
| Transformer-XL-base |  50k  |   26.2   | [log&config](./NLP/Transformer-XL/exp_results/log-50k.txt)  |
| Transformer-XL-base | 100k  |   24.2   | [log&config](./NLP/Transformer-XL/exp_results/log-100k.txt) |
| Transformer-XL-base | 200k  |   23.5   | [log&config](./NLP/Transformer-XL/exp_results/log-200k.txt) |
 </div> -->
