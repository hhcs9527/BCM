# BCM Framework for Image Deblurring

Please download GoPro dataset into Desktop and modify the path in modifier.py and run
https://drive.google.com/file/d/1H0PIXvJH4c40pk7ou6nAwoxuR4Qh_Sa2/view
```
cd datas
python modifier.py
```

__Requires.__
```
pytorch (>= 0.4.1)
numpy
scipy
scikit-image
opencv
```

__For setting config, check the tempplate inside expConfig, you can set the custom deblur net here.__
```python
# experiment Name / Method
experiment_name: Patch_Deblur_FPN_VAE_PAC_content_condiser_c16

# Some setting about training
isTrain : True
train_epoch : 100
start_train_epoch : 0
batch_size : 6
image_size : 256
gpu : 0
learning_rate : 0.0001

# deblur part
# deblur net (including size) setting
G : 'Patch_Deblur_FPN_VAE_PAC'
channel : 16

# reblur part
# per-pixel conv size & recurrent time
per_pix_kernel: 3
Recurrent_times: 3
```

__For model training with specific config, run following commands.__

```
python main.py
```


__For model testing, change the config with isTrain to False, then run following commands.__

```
python main.py
```

