Unconditional Audio Generation with GAN and Cycle Regularization
================================================================

## Install requirements
```
pip install -r requirements.txt
```

## Download pretrained parameters

The pretrained parameters can be downloaded here: 
[Pretrained parameters](https://www.dropbox.com/s/sz8flqb9v4d7edz/models.zip)


Or use the following script
```
bash download_and_unzip_models.sh
```

## Usage
```
python generate.py
```

## Vocoder

https://github.com/descriptinc/melgan-neurips


For singing, piano, and violin, we have modify the MelGAN to include GRU in the vocoder architecture. We have found that this modification yields improved audio quality.

## Audio samples

```
samples/
```
