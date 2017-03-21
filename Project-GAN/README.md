## GAN

|Project|Algorithm|venv|dataset|result|  
|:--:|:--:|:--:|:--:|:--:|  
|AC-GAN-burness-tf / [repo](https://github.com/burness/tensorflow-101/tree/master/GAN/AC-GAN)|ACGAN|||:sleeping:|  
|[DCGAN-carpedm20-tf](#dcgan-carpedm20-tensorflow) / [repo](https://github.com/carpedm20/DCGAN-tensorflow)|DCGAN|envDL|mnist+celebA|:ok_hand:|  
|DCGAN-soumith-torch-lsun+ImageNet / [repo](https://github.com/soumith/dcgan.torch)|DCGAN|||:sleeping:|  
|[Info-GAN-burness-tf](#info-gan-burness-tensorflow) / [repo](https://github.com/burness/tensorflow-101/tree/master/GAN/Info-GAN)|Info-GAN|envDL|mnist|:ok_hand:|  
|[WassersteinGAN-torch](#wassersteingan) / [repo](https://github.com/martinarjovsky/WassersteinGAN)|WGAN|envDL/envTorch|LSUN+mnist|:ok_hand:|  
|[StyleSynthesis-machrisaa-tf+VGG19](#stylesynthesis-machrisaa-tensorflowvgg19) / [repo](https://github.com/machrisaa/stylenet)|-|envDL|random img|:ok_hand:|  


[Video for DCGAN mnist](https://youtu.be/Mbb6TD_8p98)
[Video for DCGAN mixed flower](https://youtu.be/298K3OalzKM)
[Video for DCGAN sunflower](https://youtu.be/1uC40kYTr8I)

[Video for WGAN flowers](https://youtu.be/e50WBRManWU 'WGAN flowers')  
[Video for WGAN-LSUN-bedroom](https://youtu.be/wQKdqHHEvg0 'WGAN-bedroom-DC')  

[Video for Style CraterLake to starry night](https://youtu.be/Au0RY8onKMk 'Style CraterLake to starry night')  
[Video for Style OSU Valley to starry night](https://youtu.be/TsWGWEtyIPg 'Style OSU Valley to starry night')  
[Video for Style OSU Valley to OSU buil sty3](https://youtu.be/xIMjj269z7w 'Style OSU Valley to OSU buil sty3')  
[Video for Style OSU Valley to Les Demoiselles dAvignon](https://youtu.be/Tg4G3nEljf8 'Style OSU Valley to Les Demoiselles dAvignon')  
[Video for Style husky to the scream](https://youtu.be/MmTYH6zKO9g 'Style husky to the scream')  


### Start-GAN
#### 02/14/2017 Tue midnight  
Intend to implement a GAN project [Generative Adversarial Text-to-Image Synthesis](https://github.com/reedscot/icml2016), needs Torch, CuDNN, and the display package as prerequisites. Then I started to configure them. The issues I has is listed in [torch installation](#torch-installation)。  
GANs have been primarily applied to modelling natural images. They are now producing [excellent results in image generation tasks](https://arxiv.org/abs/1511.06434), generating images that are significantly sharper than those trained using other leading generative methods based on maximum likelihood training objectives.

A [project](http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/) of Approximating a 1D Gaussian distribution demo+[code](https://github.com/AYLIEN/gan-intro), with the video
[Generative Adversarial Network visualization](https://youtu.be/mObnwR-u8pc).

[***Back*** to subcontents ***GAN***](#gan)  

### Info-GAN-burness-tensorflow
#### 02/15/2017 Wed
implement Info-GAN-burness-tensorflow with mnist dataset, from [repo](https://github.com/burness/tensorflow-101/tree/master/GAN/Info-GAN)  
run `train.py`, then `generate.py`. The results are only 2 figures.  
dir is `liukaib@lifgroup:/home/liukaib/tenso/GAN-burness/tensorflow-101/GAN/Info-GAN`,   
then updated to `liukaib@lifgroup:/home/liukaib/GAN/Info-GAN-burness-tensorflow`

[***Back*** to subcontents ***GAN***](#gan)  

### DCGAN-carpedm20-tensorflow
#### 03/01/2017 Wed
implement DCGAN with celebA face dataset, from [repo](https://github.com/carpedm20/DCGAN-tensorflow). The type of dataset is a folder with 202599 `.jpg` images.  
dir is `liukaib@lifgroup:/home/liukaib/GAN/DCGAN-tensorflow`, then updated to `liukaib@lifgroup:/home/liukaib/GAN/DCGAN-carpedm20-tensorflow` on 03/11/2017 Sat.
total train is 24 epoch/7.7h.

After group meeting, I tried mnist dataset and got some results which can be used as gif to illustrate.

#### 03/16/2017 Thu  
I put Eugene's flower folder into `DCGAN-carpedm20-tensorflow/data` and run with 100 epochs:   
```zsh
CUDA_VISIBLE_DEVICES=1 python main.py --dataset flower  --epoch 100 --is_train --is_crop True
```
But, the result is not good and we cannot generate any flower images.

#### 03/17/2017 Fri  
I change the epoch to 3000 to see if the result will get better.  
```zsh
python main.py --dataset flower  --epoch 3000 --is_train --is_crop True
```

#### 03/18/2017 Sat
The generated images is not consistent to the corresponding position. I believe it's because the type of flower in the same folder are too different with unique features. Then I tried another time with:  
```zsh
python main.py --dataset flower  --epoch 1000 --is_train --is_crop True
```


[***Back*** to subcontents ***GAN***](#gan)  

### WassersteinGAN
#### 03/08/2017 Wed
workon WGAN on `lifgroup` at night  
install [PyTorch](http://pytorch.org/)  
with python-2.7+CUDA-8.0
```zsh
pip install https://s3.amazonaws.com/pytorch/whl/cu80/torch-0.1.10.post2-cp27-none-linux_x86_64.whl
pip install torchvision
```
clone [WGAN repo](https://github.com/martinarjovsky/WassersteinGAN) into `lifgroup:~/GAN` (moved to `/media/HDD/LargeDisk/liukaib/WassersteinGAN` on 03/10/2017)  
venv is `envDL`  
execute command is not friendly given (fixed on [03/10/2017 Fri](#03102017-fri)'s log):
```zsh
# With DCGAN
python main.py --dataset lsun --dataroot [lsun-train-folder] --cuda


# With MLP
python main.py --mlp_G --ngf 512
```

After run `python main.py.....`, there is an error `ImportError: No module named lmdb`. So I install lmdb with `pip install lmdb`.  
Then, run the main, turns out that LSUN dataset(bedroom) not found. After research, pytorch offers downloading command for mnist, not for lsun, which make users have do it manually. So I find a [repo](https://github.com/fyu/lsun) for downloading LSUN-bedroom data. **remember**, do not use `wget` on git files which are in encrypted url, I just need to open 'raw' of a `.py` code then use `wget` to download a file separately.  
So I run `python download.py -c bedroom` to start downloading the two 45GB zip files, which takes 1.5 hours. But it failed and left about a half un-downloaded. With that uncompleted zip I cannot unzip it.


#### 03/10/2017 Fri

After several trial of downloading with ssh and tmux, I turned to the desktop and login my accout directly to run the download code. Finally it works after about 2 hours. And I can unzip them successfully.

#### 03/11/2017 Sat
In the morning, I put both unzipped LSUN data folder (`bedroom_train_lmdb`/45GB->54GB,`bedroom_val_lmdb`/4MB->6MB) into `/WassersteinGAN`.   
Since the `lsun-train-folder` is in `./`, the same path as `main.py`, so the run command `python main.py --dataset lsun --dataroot [lsun-train-folder] --cuda` should be
```zsh
python main.py --dataset lsun --dataroot './' --cuda
```
As is mentioned in the repo's note,
> The first time running on the LSUN dataset it can take a long time (up to an hour) to create the dataloader. After the first run a small cache file will be created and the process should take a matter of seconds. The cache is a list of indices in the lmdb database (of LSUN)

It really took a while to create the dataloader.  

After cache created, both TitanX GPUs in group server got fully occupied again. So my plan has to stop for the moment.

After waiting from noon to dusk, I can execute my code after the termination of Jialin's project at about 18:00.


#### 03/13/2017 Mon  
After 51 hours' running, the program for DCGAN got end. Runtime is 2h/epoch.
In total, N = 3M input_images, E = 25 epochs.  
Training parameters are:   
- batch_size = 64 input_images
- output_image_step = 500 iterations  
- iteration_step = 5 batches

47392 batches = N / batch_size = 3M / 64  
227862 iterations = N / batch_size / iteration_step \* epochs = batches / iteration_step \* epochs = batches / 5 \* epochs  
455 output_images = iterations / output_image_step = 227862 / 500  
1 iteration = iteration_step \* batch_size = (5\*64) = 320 input_images  
1 output_image = 500 iterations = (500\*64\*5) = 0.16M input_images  
the format of output is:  
[epoch_i][batch_i][gen_iterations]
```zsh
[24/25][47380/47392][227859] Loss_D: -0.178149 Loss_G: 0.003281 Loss_D_real: 0.020692 Loss_D_fake 0.198841
[24/25][47385/47392][227860] Loss_D: -0.160666 Loss_G: 0.049911 Loss_D_real: -0.213569 Loss_D_fake -0.052902
[24/25][47390/47392][227861] Loss_D: -0.204119 Loss_G: 0.022151 Loss_D_real: -0.128741 Loss_D_fake 0.075378
[24/25][47392/47392][227862] Loss_D: -0.118183 Loss_G: 0.026365 Loss_D_real: -0.016479 Loss_D_fake 0.101703
```
If I want to use MLP:
```zsh
CUDA_VISIBLE_DEVICES=1 python main.py --dataset lsun --dataroot './' --cuda --mlp_G --ngf 512
```
If I want to to continue training:
```zsh
# for DCGAN
CUDA_VISIBLE_DEVICES=1 python main.py --dataset lsun --dataroot './' --cuda --netG './samples/netG_epoch_24.pth' --netD './samples/netD_epoch_24.pth'

# for MLP
CUDA_VISIBLE_DEVICES=1 python main.py --dataset lsun --dataroot './' --cuda --netG './samples-MLP/netG_epoch_24.pth' --netD './samples-MLP/netD_epoch_24.pth' --mlp_G --ngf 512
```
**Remember**, checkpoint of DCGAN cannot be loaded to a continued MLP execution, or the other way round, i.e. DCGAN mode can only load DCGAN checkpoint `.pth`, and MLP mode can only load MLP checkpoint `.pth`,

At night, I add a function to save 4 types of loss into `.csv` so that I can plot some curves later.  
Then I start 3 rounds of training simultaneously on DCGAN-cont.+loss, DCGAN+loss, and MLP+loss.

#### 03/15/2017 Wed  
I found running WGAN much slower on the server(2.5h/epoch). Maybe because there are multiple processes simultaneously including Zheng's project on GPU1.

And, the 54GB data is located in HDD(2TB) due to the limited space in SSD (512GB), so the transfer between disk and CPU is another bottleneck for speed.

My teammates Eugene and Bill have some problems running the projects. I found that tensorflow version has a great update from 0 to 1, which are listed in the [official website](https://www.tensorflow.org/install/migration). So I recommend them to install older version `0.12.1` in their virtual environment.
At afternoon, I copied the generated images from servers with filter command `find`. Then I recorded the fast review of generated bedroom images into `.mov` file and converted it to `gif` for further presentation.  

![](./images/WGAN/WGAN-DC.gif 'WGAN-DC-animation')

|1000|50000|227500|
|:--:|:--:|:--:|
|![](./images/WGAN/WGAN_DC_p1000.png '1000')|![](./images/WGAN/WGAN_DC_p50000.png '50000')|![](./images/WGAN/WGAN_DC_p227500.png '227500')|


 [***Back*** to subcontents ***GAN***](#gan)  

#### 03/16/2017 Thu  
After 62.5 hours' running, all executions DCGAN-cont.+loss, DCGAN+loss, and MLP+loss end.  
Results are in folders named samples-DCGAN-cont, samples-DCGAN+loss, and samples-MLP, respectively.  
Now there are 3 types of files: `.png`, `loss_data.csv`, `.pth`.


Then I start 2nd round of MLP at 17:21. 1.87 second/step *   
And 1st round of Eugene's flowers(1,360 images) after putting flower folder into `./17flowers/`, making epochs = 5000 and image_output each 50 steps.
```zsh
python main.py --dataset folder --dataroot './17flowers/' --cuda --niter 5000
```

At night, I use Eugene's handwriting data(3,529 images), putting folder `EnglishHnd` to the server `./` to run WGAN.
```
EnglishHnd/Sample001/img001-001.png
EnglishHnd/Sample001/img001-002.png
EnglishHnd/Sample001/img001-003.png
...
EnglishHnd/Sample062/img001-001.png
EnglishHnd/Sample062/img001-002.png
EnglishHnd/Sample062/img001-003.png
```
```zsh
python main.py --dataset folder --dataroot './EnglishHnd/' --cuda --niter 5000
```
Almost 1min/epoch

#### 03/17/2017 Fri  
I check the result of training, flowers ends first while EnglishHnd is still on the stage of less than one third...  
Results of `python main.py --dataset folder --dataroot './17flowers/' --cuda --niter 5000`:  
```zsh
 [4999/5000][15/22][24894] Loss_D: -0.595433 Loss_G: 0.157938 Loss_D_real: -0.189094 Loss_D_fake 0.406339
 [4999/5000][20/22][24895] Loss_D: -0.495998 Loss_G: 0.419604 Loss_D_real: -0.469681 Loss_D_fake 0.026317
 [4999/5000][22/22][24896] Loss_D: -0.504061 Loss_G: 0.375015 Loss_D_real: -0.432782 Loss_D_fake 0.071279
```
And the generated images are good.


#### 03/18/2017 Fri  
EnglishHnd and bedroom-MLP-cont is still running.  
I tried to extract the number images from EnglishHnd. First option is making 0~9 numbers into `NumHnd` with separate folders, the other option is putting them into one folder `NumHnd-one` for WGAN.
```zsh
python main.py --dataset folder --dataroot './NumHnd/' --cuda --niter 1000 --batchSize 16
python main.py --dataset folder --dataroot './NumHnd-one/' --cuda --niter 1000 --batchSize 16
```
each of them takes 7 second/epoch.

### StyleSynthesis-machrisaa-tensorflow+VGG19

#### 03/10/2017 Fri

clone the [repo](https://github.com/machrisaa/stylenet), run and fail.   
Then I modified the import function/files, which is not maitained for one year. Moreover, the stylesyn project relies on vgg19 project. So I download the [vgg19.py](https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg19.py) file into the folder to be imported.

in `stylenet_patch.py`:
```python
from neural_style import custom_vgg19
-->
import custom_vgg19
```
in `custom_vgg19.py`
```python
from tensoflow_vgg import vgg19
-->
import vgg19
```
Then I find I need a `vgg19.npy` downloaded into the same path according to the same author's another [repo](https://github.com/machrisaa/tensorflow-vgg). So a 574.7 MB file is downloaded from [cloud disk](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs) into my local laptop and copied to **TitanX** server.

Since **TitanX** is occupied by Jialin and Lawrance fully, I turned to **pelican** server.  

However, `.meta` files are generated frequently, each of which has the size **500MB~2GB**, making the server stop me from continuing.    
To disable the creation of the `.meta` file, I need to add `write_meta_graph=False`, into line 237 in `stylenet_patch.py`, like this:  
```python
saved_path = saver.save(sess, "./train/saves-" + get_filename(content_file), global_step=global_step, write_meta_graph=False)
```
So everything is OK for running.  
Basic usage is, in main of `stylenet_patch.py`, set  
```
stylenet_patch.render_gen( <content image path> , <style image path>, height=<output height>)
```
and run `python stylenet_patch.py`.

Alternatively, you can go as I did. I wrote another `run_main.py` to import `stylenet_patch` and set image paths to run the `render_gen()` function from `stylenet_patch.py`.

Husky (without region) after running,
```zsh
Step 199: cost:0.0030512088     (27.3 sec)       content:0.00000, style_3:0.00089, style_4:0.00216, gram:0.00000, diff_cost_out:0.00000
Step 200: cost:0.0030384166     (27.4 sec)       content:0.00000, style_3:0.00089, style_4:0.00215, gram:0.00000, diff_cost_out:0.00000
net saved:  ./train/saves-output-g1-80-200
img saved:  ./train/output-g2-200.jpg
----------- 2 generation finished in 52 sec -----------
```

cat after running:
```zsh
Step 199: cost:0.0070971507     (15.9 sec)       content:0.00000, style_3:0.00220, style_4:0.00490, gram:0.00000, diff_cost_out:0.00000
Step 200: cost:0.0070948927     (15.9 sec)       content:0.00000, style_3:0.00219, style_4:0.00490, gram:0.00000, diff_cost_out:0.00000
net saved:  ./train/saves-output-g1-80-200
img saved:  ./train/output-g2-200.jpg
----------- 2 generation finished in 28 sec -----------
```
starry night after running:
```zsh
Step 199: cost:0.0043881140     (27.2 sec)       content:0.00000, style_3:0.00126, style_4:0.00312, gram:0.00000, diff_cost_out:0.00000
Step 200: cost:0.0043800147     (27.3 sec)       content:0.00000, style_3:0.00126, style_4:0.00312, gram:0.00000, diff_cost_out:0.00000
net saved:  ./train/saves-output-g1-80-200
img saved:  ./train/output-g2-200.jpg
----------- 2 generation finished in 51 sec -----------
```
Then I generate some oil painting style pics of natural OSU landscapes with the app `prisma` and upload them to the server.  
But the **pelican** is also fully occupied by a guy named alizades.  
Then I have to switch to **steed** instead.

After some environment configuration, I upgrade the `numpy` then I can import tensorflow successfully on **steed**.

Some results from OSU landscapes are generated with famous painting styles and saved to my own computer. Here are part of them.

The Starry Night, 1889 by Vincent van Gogh  
Les Demoiselles d'Avignon, 1907, Pablo Picasso  
The Scream, 1893 by Edvard Munch  
Mona Lisa, 1503 by Leonardo da Vinci  


|Content|Style|Result|
|:--:|:--:|:--:|
|![](./images/OSU-buil-real.jpg 'OSU-buil-real')|![](./images/OSU-pole-sty2.jpg 'OSU-pole-sty2')|![](./images/output-OSU_built_to_pole-sty2-g1-50.jpg 'output-OSU_built_to_pole-sty2-g1-50')|
|Content|+[Style]|=Result|
|![](./images/OSU-buil-real.jpg 'OSU-buil-real')|![](./images/Les_Demoiselles_dAvignon.jpg 'Les Demoiselles dAvignon, 1907, Pablo Picasso')|![](./images/output-OSU-buil-real_to_Les_Demoiselles_dAvignon-g1-55.jpg 'output-OSU-buil-real_to_Les_Demoiselles_dAvignon-g1-55')|
|Content|+[Style]|=Result|
|![](./images/OSU-Valley-real.jpg 'OSU-Valley-real')|![](./images/Les_Demoiselles_dAvignon.jpg 'Les Demoiselles dAvignon, 1907, Pablo Picasso')|![](./images/output-OSU-Valley_to_Les_Demoiselles_dAvignon-g1-50.jpg 'output-OSU-Valley_to_Les_Demoiselles_dAvignon-g1-50')|
|Content|+[Style]|=Result|
|![](./images/OSU-Valley-real.jpg 'OSU-Valley-real')|![](./images/OSU-buil-sty3.jpg 'OSU-buil-sty3')|![](./images/output-OSU-Valley_to_OSU-buil-sty3-g1-15.jpg 'output-OSU-Valley_to_OSU-buil-sty3-g1-15')|
|Content|+[Style]|=Result|
|![](./images/OSU-Valley-real.jpg 'OSU-Valley-real')|![](./images/starry_night_paint.jpg 'starry_night_paint')|![](./images/output-OSU-Valley_to_starry_night_paint-g1-25.jpg 'output-OSU-Valley_to_starry_night_paint-g1-25')|
|Content|+[Style]|=Result|
|![](./images/CraterLake-real.jpg 'CraterLake-real')|![](./images/starry_night_paint.jpg 'starry_night_paint')|![](./images/output-CraterLake_to_starry_night_paint-g1-50.jpg 'output-CraterLake_to_starry_night_paint-g1-50')|
|Content|+[Style]|=Result|
|![](./images/cat_h.jpg 'cat')|![](./images/monnalisa.jpg 'Mona Lisa, 1503 by Leonardo da Vinci')|![](./images/output-cat_to_monnalisa-g1-55.jpg 'output-cat_to_monnalisa-g1-55')|
|Content|+[Style]|=Result|
|![](./images/husky_real.jpg 'husky_real')|![](./images/the-scream.jpg 'The Scream, 1893 by Edvard Munch')|![](./images/output-husky_real_to_the-scream-g1-50.jpg 'output-husky_real_to_the-scream-g1-50')|
|Content|+[Style]|=Result|
|![](./images/beaverlogo.jpg 'beaverlogo')|![](./images/OSU-kec-sty.jpg 'OSU-kec-sty')|![](./images/output-beaverlogo_to_OSU-kec-sty-g2-50.jpg 'output-beaverlogo_to_OSU-kec-sty-g2-50')|
|Content|+[Style]|=Result|
|![](./images/OSU-kec-sty.jpg 'OSU-kec-sty')|![](./images/OSU-kec-real.jpg 'OSU-kec-real')|![](./images/OSU-kec-output-g0-80.jpg 'OSU-kec-output-g0-80')|
|Content|+[Style]|=Result|
|![](./images/OSU-pole-sty1.jpg 'OSU-pole-sty1')|![](./images/OSU-pole-real.jpg 'OSU-pole-real')|![](./images/OSU-pole-sty1-output-g0-80.jpg 'OSU-pole-sty1-output-g0-80')|
|Content|+[Style]|=Result|
|![](./images/starry_night_paint.jpg 'starry_night_paint')|![](./images/starry_night_real.jpg 'starry_night_real')|![](./images/starry_night_output-g1-80.jpg 'starry_night_output-g1-80')|
|Content|+[Style]|=Result|  


This project can be executed on **TitanX** server under my `envDL` virtualenv other than `keras_theano_py2`.

[***Back*** to subcontents ***GAN***](#gan)
