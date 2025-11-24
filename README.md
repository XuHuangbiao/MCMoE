# MCMoE: Completing Missing Modalities with Mixture of Experts for Incomplete Multimodal Action Quality Assessment
[AAAI'26] Code for ''MCMoE: Completing Missing Modalities with Mixture of Experts for Incomplete Multimodal Action Quality Assessment''

## Environments

- RTX 3090
- CUDA: 12.4
- Python: 3.8.19
- PyTorch: 2.4.1+cu124

## Features

The features (RGB, Audio, Flow) and label files of Rhythmic Gymnastics and Fis-V dataset can be downloaded from the [PAMFN](https://github.com/qinghuannn/PAMFN) repository.

The features (RGB, Audio) and label files of FS1000 dataset can be downloaded from the [Skating-Mixer](https://github.com/AndyFrancesco29/Audio-Visual-Figure-Skating) repository. We adopt the same frame sampling method to extract Optical Flow features from the FS1000 dataset, which can be downloaded via this [link](https://1drv.ms/u/c/056e0e22eb875f5c/ERsOcMRUcahPpaHPgVZN-SQBE1zb1E9xqGDs3Car8J9qlA?e=lM6ODb).

## Running
### The following are examples only, more details coming soon!

Please fill in or select the args enclosed by {} first. For example, on the **FS1000** dataset:

- Training

```
CUDA_VISIBLE_DEVICES={device ID} python main.py --video-path {path of video features} --audio-path {path of audio features} --flow-path {path of flow features} --train-label-path {path of label file of training set} --test-label-path {path of label file of test set} --model-name {the name used to save model and log} --action-type {TES/PCS/SS/TR/PE/CO/IN} --lr 1e-4 --epoch {360/460/360/210/520/520/390} --in_dim 768 --n_head 2 --n_encoder 3 --n_decoder 3 --n_query 4 --alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3
```

- Testing

```
CUDA_VISIBLE_DEVICES={device ID} python main.py --video-path {path of video features} --audio-path {path of audio features} --flow-path {path of flow features} --train-label-path {path of label file of training set} --test-label-path {path of label file of test set} --action-type {TES/PCS/SS/TR/PE/CO/IN} --in_dim 768 --n_head 2 --n_encoder 3 --n_decoder 3 --n_query 4 --test --ckpt {the name of the used checkpoint}
```

Be patient and persistent in tuning the code to achieve new state-of-the-art results.


## Acknowledgement
This repository builds upon [GDLT (CVPR 2022)](https://github.com/xuangch/CVPR22_GDLT) and [MoMKE (ACMMM 2024)](https://github.com/wxxv/MoMKE).

We thank the authors for their contributions to the research community.