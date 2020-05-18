Unconditional Audio Generation with GAN and Cycle Regularization
================================================================

This repository contains the code and samples for our paper "Unconditional Audio Generation with GAN and Cycle Regularization". The goal is to unconditionally generate singing voices, speech, and instrument sounds with GAN.

The model is implemented with PyTorch. 

## Paper
TBA

## Install dependencies
```
pip install -r requirements.txt
```

## Download pretrained parameters

The pretrained parameters can be downloaded here: 
[Pretrained parameters](https://www.dropbox.com/s/sz8flqb9v4d7edz/models.zip)

Unzip it so that the `models` folder is in the current folder.


Or use the following script
```
bash download_and_unzip_models.sh
```

## Usage
```
python generate.py
```

## Vocoder

We use [MelGAN](https://github.com/descriptinc/melgan-neurips) as the vocoder. The trained vocoders are included in the `models.zip`

For singing, piano, and violin, we have modify the MelGAN to include GRU in the vocoder architecture. We have found that this modification yields improved audio quality. For speech, we directly use the trained LJ vocoder from [MelGAN](https://github.com/descriptinc/melgan-neurips/blob/master/models).

## Audio samples

Some generated audio samples can be found in:
```
samples/
```
