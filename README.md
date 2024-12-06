

## Catalog
- [x] ImageNet-1K Training Code  
- [x] Fine-tune on [CIFAR-100, Oxford 102 flowers, Stanford cars, FGVC-Aircraft]



<!-- ✅ ⬜️  -->

## Results and Pre-trained Models
### ImageNet-1K trained models

| name | resolution |acc@1 | #params | FLOPs | model |
|:---:|:---:|:---:|:---:| :---:|:---:|
| [ConvNeXt-T](https://github.com/facebookresearch/ConvNeXt.git) | 224x224 | 82.1 | 28M | 4.5G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth) |
| [ConvNeXt-S](https://github.com/facebookresearch/ConvNeXt.git) | 224x224 | 83.1 | 50M | 8.7G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth) |
| [ConvNeXt-B](https://github.com/facebookresearch/ConvNeXt.git) | 224x224 | 83.8 | 89M | 15.4G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth) |
| SELNeXt-T | 224x224 | 82.6 | 35M | 4.5G | [model](https://drive.google.com/file/d/1fXYWVJTXOmM4UUJeaCk1OcszcRjN1-1N/view?usp=sharing) |
| SELNeXt-S | 224x224 | 83.8 | 62M | 8.7G | [model](https://drive.google.com/file/d/14G3em_WrH968DM5y_GGeIrDqafjiv6ZR/view?usp=sharing) |
| SELNeXt-B | 224x224 | 84.2 | 110M | 15.4G | [model](https://drive.google.com/file/d/1YO4uvF0iFeubffHN6LO_D0n3BzvhpMnY/view?usp=sharing) |


## Installation
Please check [INSTALL.md](INSTALL.md) for installation instructions. 

## Evaluation

Single-GPU
```
python main.py --model [model name] --eval true \
--resume <PATH_TO_PRETRAINED_WEIGHTS> \
--input_size 224 \
--data_path <PATH_IMAGENET-1K> 
```
Multi-GPU
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model [modelname] --eval true \
--resume <PATH_TO_PRETRAINED_WEIGHTS> \
--input_size 224 \
--data_path <PATH_IMAGENET-1K> 

```

This should give 
```
python main.py --model selnext_base --eval true --resume ./selnext_base_1k_224_ema.pth --input_size 224 --data_path /imagenet
* Acc@1 84.230 Acc@5 96.830 loss 0.670
```

- For evaluating other model variants, change `--model`, `--resume` accordingly.

## Training
See [TRAINING.md](TRAINING.md) for training and fine-tuning instructions.

## Acknowledgement
This repository is modified from the [ConvNeXt](https://github.com/facebookresearch/ConvNeXt.git) repository.

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

