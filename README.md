# MCMoE
[AAAI'26] Code for ''MCMoE: Completing Missing Modalities with Mixture of Experts for Incomplete Multimodal Action Quality Assessment''

Paper Link: [Arxiv](https://arxiv.org/abs/2511.17397v2)

## Environments

- RTX 3090
- CUDA: 12.4
- Python: 3.8.19
- PyTorch: 2.4.1+cu124

## Dataset Preparation
### Features

The features (RGB, Audio, Flow) and label files of Rhythmic Gymnastics and Fis-V dataset can be downloaded from the [PAMFN](https://github.com/qinghuannn/PAMFN) repository.

The features (RGB, Audio) and label files of FS1000 dataset can be downloaded from the [Skating-Mixer](https://github.com/AndyFrancesco29/Audio-Visual-Figure-Skating) repository. We adopt the same frame sampling method to extract Optical Flow features from the FS1000 dataset, which can be downloaded via this [link](https://1drv.ms/f/c/056e0e22eb875f5c/IgAq1rrVu9n_Rqd2CaYr00ADAcpMRjh4Tf_gmX3yUsYIFEc?e=g9rFJf).

### Datasets Structure
You can place the corresponding datasets according to the following structure:

```
$DATASET_ROOT
├── FS1000
    ├── output_feature_fs1000_new
        ├── 2018_Final_MF_Junhwan.npy
        ...
        └── 2021_R_PF_7.npy
    ├── ast_feature_fs1000_new
        ├── 2018_Final_MF_Junhwan.npy
        ...
        └── 2021_R_PF_7.npy
    ├── i3d_avg_clip8_5s_fs1000
        ├── 2018_Final_MF_Junhwan.npy
        ...
        └── 2021_R_PF_7.npy
    ├── train_fs1000_new.txt
    └── val_fs1000_new.txt
├── Fis-V
    ├── Fis-feature
        ├── FISV_audio_AST.npy
        ├── FISV_flow_I3D.npy
        └── FISV_rgb_VST.npy
    ├── train.txt
    └── test.txt
└── RG
    ├──RG-feature
        ├── Ball_audio_AST.npy
        ├── Ball_flow_I3D.npy
        ...
        └── Ribbon_rgb_VST.npy
    ├── train.txt
    └── test.txt
```

## Model Weights
You can download the model weights for all tasks across three datasets from this [link](https://1drv.ms/f/c/056e0e22eb875f5c/IgAq1rrVu9n_Rqd2CaYr00ADAcpMRjh4Tf_gmX3yUsYIFEc?e=g9rFJf).

## Running
### Please fill in or select the args enclosed by {} first.
On the **FS1000** dataset:

- Training

```
CUDA_VISIBLE_DEVICES={device ID} python main.py --video-path {path of video features} --audio-path {path of audio features} --flow-path {path of flow features} --train-label-path {path of label file of training set} --test-label-path {path of label file of test set} --model-name {the name used to save model and log} --action-type {TES/PCS/SS/TR/PE/CO/IN} --dataset FS1000 --clip-num 95 --lr 1e-4 --epoch {360/460/360/210/520/520/390} --in_dim 768 --n_head 2 --n_encoder 3 --n_decoder 3 --n_query 4 --alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3
```

- Testing

```
CUDA_VISIBLE_DEVICES={device ID} python main.py --video-path {path of video features} --audio-path {path of audio features} --flow-path {path of flow features} --train-label-path {path of label file of training set} --test-label-path {path of label file of test set} --action-type {TES/PCS/SS/TR/PE/CO/IN} --dataset FS1000 --clip-num 95 --in_dim 768 --n_head 2 --n_encoder 3 --n_decoder 3 --n_query 4 --test --ckpt {the name of the used checkpoint}
```

On the **FisV** dataset:

- Training

```
CUDA_VISIBLE_DEVICES={device ID} python main.py --video-path {path of video features} --audio-path {path of audio features} --flow-path {path of flow features} --train-label-path {path of label file of training set} --test-label-path {path of label file of test set} --model-name {the name used to save model and log} --action-type {TES/PCS} --dataset FisV --clip-num 124 --lr 2e-4 --epoch {460/510} --in_dim 1024 --n_head 2 --n_encoder 3 --n_decoder 3 --n_query 4 --use_pe True --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3
```

- Testing

```
CUDA_VISIBLE_DEVICES={device ID} python main.py --video-path {path of video features} --audio-path {path of audio features} --flow-path {path of flow features} --train-label-path {path of label file of training set} --test-label-path {path of label file of test set} --action-type {TES/PCS} --dataset FisV --clip-num 124 --in_dim 1024 --n_head 2 --n_encoder 3 --n_decoder 3 --n_query 4 --use_pe True --test --ckpt {the name of the used checkpoint}
```

On the **RG** dataset:

- Training

```
CUDA_VISIBLE_DEVICES={device ID} python main.py --video-path {path of video features} --audio-path {path of audio features} --flow-path {path of flow features} --train-label-path {path of label file of training set} --test-label-path {path of label file of test set} --model-name {the name used to save model and log} --action-type {Ball/Clubs/Hoop/Ribbon} --dataset RG --clip-num 68 --lr 2e-4 --epoch {410/560/270/300} --in_dim 1024 --n_head 2 --n_encoder 3 --n_decoder 3 --n_query 4 --use_pe True --alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3
```

- Testing

```
CUDA_VISIBLE_DEVICES={device ID} python main.py --video-path {path of video features} --audio-path {path of audio features} --flow-path {path of flow features} --train-label-path {path of label file of training set} --test-label-path {path of label file of test set} --action-type {Ball/Clubs/Hoop/Ribbon} --dataset RG --clip-num 68 --in_dim 1024 --n_head 2 --n_encoder 3 --n_decoder 3 --n_query 4 --use_pe True --test --ckpt {the name of the used checkpoint}
```

**Please note! During training, we save the model that performs best in complete multimodal scenarios. Then, during testing, we evaluate this model across all incomplete multimodal scenarios. Additionally, we save both the model with the best SP. Corr. metric and the model with the best MSE metric, then select the model that achieves better overall balance across all settings. You can modify the code based on your specific application and select the optimal model for your needs.**

Be patient and persistent in tuning the code to achieve new state-of-the-art results.

## Citation
If our project is helpful for your research, please consider citing:
```
@article{xu2025mcmoe,
  title={MCMoE: Completing Missing Modalities with Mixture of Experts for Incomplete Multimodal Action Quality Assessment},
  author={Xu, Huangbiao and Wu, Huanqi and Ke, Xiao and Wu, Junyi and Xu, Rui and Xu, Jinglin},
  journal={arXiv preprint arXiv:2511.17397},
  year={2025}
}
```

## Acknowledgement
This repository builds upon [GDLT (CVPR 2022)](https://github.com/xuangch/CVPR22_GDLT) and [MoMKE (ACMMM 2024)](https://github.com/wxxv/MoMKE).

We thank the authors for their contributions to the research community.