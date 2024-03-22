# XB-MAML: Learning Expandable Basis Parameters for Effective Meta-Learning with Wide Task Coverage




> **XB-MAML: Learning Expandable Basis Parameters for Effective Meta-Learning with Wide Task Coverage**  [[Paper]](https://arxiv.org/abs/2403.06768)
>
> Jae-Jun Lee, Sung Whan Yoon
>
> AISTATS 2024




## Installation

1. Clone its repository

   ```bash
   git clone https://github.com/johnjaejunlee95/XB-MAML.git
   ```

2. Install torchmeta

   ```bash
   conda create env -y -n xbmaml python=3.9
   conda activate xbmaml
   pip install torchmeta
   ```



## Datasets

**Download all datasets via [this link](https://drive.google.com/file/d/1-HGJ0C1QHGs6RFzgVrvlYHDXZ2eFu8sR/view?usp=sharing).** 

Alternatively, if you prefer to download them individually, please use the following links:

1. CIFAR-FS: [link](https://drive.google.com/file/d/1--SLwRqQzIRu_RcK91L4UjrGR7y267FN/view?usp=drive_link)
2. mini-ImageNet: [link](https://drive.google.com/file/d/1-8XtrPWViumNpgT4u53TYu3qWqJ0r-Aa/view?usp=sharing)
3. tiered-ImageNet: [link](https://drive.google.com/file/d/16H2Hlv3HE0P3cVHGr_es36RnWV_zrCjH/view?usp=drive_link)
4. Omniglot: [link](https://drive.google.com/file/d/1-NgAuCphzvmLao_1vDkn7v6-gd3aC1qt/view?usp=sharing)
5. Aircraft, Birds, Fungi, Texture (ABF, BTAF): [link](https://drive.google.com/file/d/1-I8QRuYeY1pWBpp2CxNZtQmI9POvDlpZ/view?usp=sharing)



## Run

### Training

#### Single Datasets (mini, tiered, CIFAR-FS)

```bash
python train_meta.py --multi --temp_scaling 5 --batch_size 2 --update_step 3 --update_step_test 7 --update_lr 0.03 --regularizer 5e-4 --datasets miniimagenet --epoch 60000 --max_test_task 1000 --gpu_id 0
```

#### Multiple Datasets (ABF, BTAF, CIO)

```bash
python train_meta.py --multi --temp_scaling 8 --batch_size 2 --update_step 3 --update_step_test 7 --update_lr 0.05 --regularizer 1e-3 --datasets_path /your/own/path --datasets MetaABF --epoch 80000 --max_test_task 600 --gpu_id 0
```



### Evaluation/Test

```bash
python test_meta.py --datasets_path /your/own/path --checkpoint_path ./save/ckpt/ --datasets MetaABF --num_test 1
```



### Arguments

```
option arguments:  
--epochs:             epoch number (default: 60000)  
--num_ways:           N-way (default: 5)  
--num_shots:          k shots for support set (default: 5)  
--num_shots_test:     number of query set (default: 15) 
--imgc:               RGB(image channel) (default: 3)  
--filter_size:        size of convolution filters (default: 64)  
--batch_size:         meta-batch size (default: 2)  
--max_test_task:      number of tasks for evaluation (default: 600)  
--meta_lr:            outer-loop learning rate (default: 1e-3)  
--update_lr:          inner-loop learning rate (default: 1e-2)  
--update_step:        number of inner-loop update steps while training (default: 5)  
--update_test_step:   number of inner-loop update steps while evaluating (default: 10) 
--dropout:            dropout probability (default: 0.)
--gpu_id:             gpu device number (default: 0)
--model:              model architecture: Conv-4, ResNet12 (default: conv4)
--datasets:           datasets: miniimagenet, tieredimagenet, cifar-fs, MetaABF, MetaBTAF, MetaCIO (default: MetaABF)
--multi:	      Apply XB-MAML (store true)
--datasets_path:      Datasets directory path
--checkpoint_path:    checkpoint directory path (default:./save/ckpt/)
--version:            file version (default: 0)
--num_test:           how many times to try test
```

**you can check the details on ./utils/args.py**
