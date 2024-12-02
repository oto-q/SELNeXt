

## Catalog
- [x] ImageNet-1K Training Code  
- [x] Fine-tune on [CIFAR-100, Oxford 102 flowers, Stanford cars, FGVC-Aircraft] with Weights & Biases logging 



<!-- ✅ ⬜️  -->

## Results and Pre-trained Models
### ImageNet-1K trained models

| name | resolution |acc@1 | #params | FLOPs | model |
|:---:|:---:|:---:|:---:| :---:|:---:|
| [ConvNeXt-T](https://github.com/facebookresearch/ConvNeXt.git) | 224x224 | 82.1 | 28M | 4.5G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth) |
| [ConvNeXt-S](https://github.com/facebookresearch/ConvNeXt.git) | 224x224 | 83.1 | 50M | 8.7G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth) |
| [ConvNeXt-B](https://github.com/facebookresearch/ConvNeXt.git) | 224x224 | 83.8 | 89M | 15.4G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth) |


## Installation
Please check [INSTALL.md](INSTALL.md) for installation instructions. 

## Evaluation

Single-GPU
```
python main.py --model selnext_base --eval true \
--resume <PATH_TO_PRETRAINED_WEIGHTS> \
--input_size 224 \
--data_path <PATH_IMAGENET-1K> \
--model_ema true \
--model_ema_eval true
```
Multi-GPU
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model selnext_base --eval true \
--resume <PATH_TO_PRETRAINED_WEIGHTS> \
--input_size 224 \
--data_path <PATH_IMAGENET-1K> \
--model_ema true \
--model_ema_eval true
```

This should give 
```
* Acc@1 85.820 Acc@5 97.868 loss 0.563
```

- For evaluating other model variants, change `--model`, `--resume` accordingly.

## Training
See [TRAINING.md](TRAINING.md) for training and fine-tuning instructions.

## Acknowledgement
This repository is forked from the [ConvNeXt](https://github.com/facebookresearch/ConvNeXt.git) repository.
[ConvNeXt](https://github.com/facebookresearch/ConvNeXt.git) is built using the [timm](https://github.com/rwightman/pytorch-image-models) library, [DeiT](https://github.com/facebookresearch/deit) and [BEiT](https://github.com/microsoft/unilm/tree/master/beit) repositories.

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

