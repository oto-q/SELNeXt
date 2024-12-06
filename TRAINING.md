# Training

We provide ImageNet-1K training, ImageNet-22K pre-training, and ImageNet-1K fine-tuning commands here.
Please check [INSTALL.md](INSTALL.md) for installation instructions first.

## Multi-node Training
We use multi-node training on a SLURM cluster with [submitit](https://github.com/facebookincubator/submitit) for producing the results and models in the paper. Please install:
```
pip install submitit
```
We will give example commands for both multi-node and single-machine training below.

## ImageNet-1K Training 
ConvNeXt-T training on ImageNet-1K with 4 8-GPU nodes:
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model selnext_tiny --drop_path 0.1 \
--batch_size 128 --lr 4e-3 --update_freq 1 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
```

- You may need to change cluster-specific arguments in `run_with_submitit.py`.
- You can add `--use_amp true` to train in PyTorch's Automatic Mixed Precision (AMP).
- Use `--resume /path_or_url/to/checkpoint.pth` to resume training from a previous checkpoint; use `--auto_resume true` to auto-resume from latest checkpoint in the specified output folder.
- `--batch_size`: batch size per GPU; `--update_freq`: gradient accumulation steps.
- The effective batch size = `--nodes` * `--ngpus` * `--batch_size` * `--update_freq`. In the example above, the effective batch size is `4*8*128*1 = 4096`. You can adjust these four arguments together to keep the effective batch size at 4096 and avoid OOM issues, based on the model size, number of nodes and GPU memory.

You can use the following command to run this experiment on a single machine: 
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model selnext_tiny --drop_path 0.1 \
--batch_size 128 --lr 4e-3 --update_freq 4 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k 
--output_dir /path/to/save_results
```

- Here, the effective batch size = `--nproc_per_node` * `--batch_size` * `--update_freq`. In the example above, the effective batch size is `8*128*4 = 4096`. Running on one machine, we increased `update_freq` so that the total batch size is unchanged.

To train other ConvNeXt variants, `--model` and `--drop_path` need to be changed. Examples are given below, each with both multi-node and single-machine commands:

<details>
<summary>
ConvNeXt-S
</summary>

Multi-node
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model selnext_small --drop_path 0.4 \
--batch_size 128 --lr 4e-3 --update_freq 1 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
```

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model selnext_small --drop_path 0.4 \
--batch_size 128 --lr 4e-3 --update_freq 4 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```
</details>
<details>
<summary>
ConvNeXt-B
</summary>

Multi-node
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model selnext_base --drop_path 0.5 \
--batch_size 128 --lr 4e-3 --update_freq 1 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
``` 

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model selnext_base --drop_path 0.5 \
--batch_size 128 --lr 4e-3 --update_freq 4 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
``` 

</details>


## CIFAR-100 / Oxford 102 flowers / Stanford cars / FGVC Aircraft dataset Fine-tuning
### Finetune from ImageNet-1K pre-training 
### Follow [these instructions](https://github.com/pytorch/vision/issues/7545#issuecomment-1631441616) for Stanford cars dataset.
The training commands given above for ImageNet-1K use the default resolution (224). We also fine-tune these trained models

Single-machine
```
python main.py --model [model name] --finetune <PATH_TO_PRETRAINED_WEIGHTS> \
--data_set [CIFAR/CARS/FLOWER/AIR] --data_path <DATASET FOLDER> --batch_size 32 --update_freq 8 --input_size 224 --lr 1e-4 --weight_decay 1e-4 --epochs 40 --reprob 0.0 --min_lr 1e-5 --model_ema true --warmup_epochs 10
```



