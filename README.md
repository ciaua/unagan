Unconditional Audio Generation with GAN and Cycle Regularization
================================================================

This repository contains the code and samples for our paper "Unconditional Audio Generation with GAN and Cycle Regularization". The goal is to unconditionally generate singing voices, speech, and instrument sounds with GAN.

The model is implemented with PyTorch. 

## Paper
[Unconditional Audio Generation with GAN and Cycle Regularization](https://arxiv.org/abs/2005.08526)

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
Display the options
```
python generate.py -h
```

### Generate singing voices
The following commands are equivalent.
```
python generate.py
python generate.py -data_type singing -arch_type hc --duration 10 --num_samples 5
python generate.py -d singing -a hc --duration 10 -ns 5
```

### Generate speech
```
python generate.py -d speech
```

### Generate piano sounds
```
python generate.py -d piano
```

### Generate violin sounds
```
python generate.py -d violin
```

## Vocoder

We use [MelGAN](https://github.com/descriptinc/melgan-neurips) as the vocoder. The trained vocoders are included in the `models.zip`

For singing, piano, and violin, we have modify the MelGAN to include GRU in the vocoder architecture. We have found that this modification yields improved audio quality. For speech, we directly use the trained LJ vocoder from [MelGAN](https://github.com/descriptinc/melgan-neurips/blob/master/models).


# Train your own model

One may use the following steps to train their own models.

0. (Singing only) Separate singing voices from the audios you collect.
We use a separation model we developed. You can use open-sourced ones such as [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch) or [Spleeter](https://github.com/deezer/spleeter).

1. `scripts/collect_audio_clips.py`
2. `scripts/extract_mel.py`
3. `scripts/make_dataset.py`
4. `scripts/compute_mean_std.mel.py`

5. `scripts/train.*.py`

# Audio samples

Some generated audio samples can be found in:
```
samples/
```
