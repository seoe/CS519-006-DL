## GAN

|Project|Algorithm|venv|dataset|result|
|:--:|:--:|:--:|:--:|:--:|
|AC-GAN-burness-tf / [repo](https://github.com/burness/tensorflow-101/tree/master/GAN/AC-GAN)|ACGAN|||:sleepy:|
|[DCGAN-carpedm20-tf](#dcgan-carpedm20-tensorflow) / [repo](https://github.com/carpedm20/DCGAN-tensorflow)|DCGAN|envDL|mnist+celebA|:ok_hand:|
|DCGAN-soumith-torch-lsun+ImageNet / [repo](https://github.com/soumith/dcgan.torch)|DCGAN|||:sleepy:|
|[Info-GAN-burness-tf](#info-gan-burness-tensorflow) / [repo](https://github.com/burness/tensorflow-101/tree/master/GAN/Info-GAN0)|Info-GAN|envDL|mnist|:ok_hand:|
|[WassersteinGAN](#wassersteingan) / [repo](https://github.com/martinarjovsky/WassersteinGAN)|WGAN|envDL/envTorch|LSUN+mnist|:ok_hand:|
|[StyleSynthesis-machrisaa-tf+VGG19](#stylesynthesis-machrisaa-tensorflowvgg19) / [repo](https://github.com/machrisaa/stylenet)|-|envDL|random img|:ok_hand:|

### Start-GAN
#### 2017/02/14 Tue midnight  
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
implement DCGAN with celebA face dataset, from [repo](https://github.com/carpedm20/DCGAN-tensorflow).
dir is `liukaib@lifgroup:/home/liukaib/GAN/DCGAN-tensorflow`, then updated to `liukaib@lifgroup:/home/liukaib/GAN/DCGAN-carpedm20-tensorflow` on 03/11/2017 Sat.
total train is 24 epoch/7.7h.

After group meeting, I tried mnist dataset and got some results which can be used as gif to illustrate.

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
```bash
python main.py --dataset lsun --dataroot './' --cuda
```
As is mentioned in the repo's note,
> The first time running on the LSUN dataset it can take a long time (up to an hour) to create the dataloader. After the first run a small cache file will be created and the process should take a matter of seconds. The cache is a list of indices in the lmdb database (of LSUN)

It really took a while to create the dataloader.  

After cache created, both TitanX GPUs in group server got fully occupied again. So my plan has to stop for the moment.

After waiting from noon to dusk, I can execute my code after the termination of Jialin's project at about 18:00.


#### 03/13/2017 Mon  
After 51 hours' running, the program for DCGAN got end. Runtime is 2h/epoch.
```bash
[24/25][47380/47392][227859] Loss_D: -0.178149 Loss_G: 0.003281 Loss_D_real: 0.020692 Loss_D_fake 0.198841
[24/25][47385/47392][227860] Loss_D: -0.160666 Loss_G: 0.049911 Loss_D_real: -0.213569 Loss_D_fake -0.052902
[24/25][47390/47392][227861] Loss_D: -0.204119 Loss_G: 0.022151 Loss_D_real: -0.128741 Loss_D_fake 0.075378
[24/25][47392/47392][227862] Loss_D: -0.118183 Loss_G: 0.026365 Loss_D_real: -0.016479 Loss_D_fake 0.101703
```
If I want to use MLP:
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --dataset lsun --dataroot './' --cuda --mlp_G --ngf 512
```
If I want to to continue training:
```bash
# for DCGAN
CUDA_VISIBLE_DEVICES=1 python main.py --dataset lsun --dataroot './' --cuda --netG './samples/netG_epoch_24.pth' --netD './samples/netD_epoch_24.pth'

# for MLP
CUDA_VISIBLE_DEVICES=1 python main.py --dataset lsun --dataroot './' --cuda --netG './samples/netG_epoch_24.pth' --netD './samples/netD_epoch_24.pth' --mlp_G --ngf 512
```
**Remember**, checkpoint of DCGAN cannot be loaded to a continued MLP execution, or the other way round, i.e. DCGAN mode can only load DCGAN checkpoint `.pth`, and MLP mode can only load MLP checkpoint `.pth`,

At night, I add a function to save 4 types of loss into `.csv` so that I can plot some curves later.  
Then I start 3 round of training simultaneously on DCGAN-cont.+loss, DCGAN+loss, and MLP+loss.

#### 03/15/2017 Wed  
I found running WGAN much slower on the server(2.5h/epoch). Maybe because there are multiple processes simultaneously including Zheng's project on GPU1.

[***Back*** to subcontents ***GAN***](#gan)  

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
|![](./images/OSU-buil-real.jpg 'OSU-buil-real')|![](./images/Les_Demoiselles_dAvignon.jpg 'Les Demoiselles d'Avignon, 1907, Pablo Picasso')|![](./images/output-OSU-buil-real_to_Les_Demoiselles_dAvignon-g1-55.jpg 'output-OSU-buil-real_to_Les_Demoiselles_dAvignon-g1-55')|
|Content|+[Style]|=Result|
|![](./images/OSU-Valley-real.jpg 'OSU-Valley-real')|![](./images/Les_Demoiselles_dAvignon.jpg 'Les Demoiselles d'Avignon, 1907, Pablo Picasso')|![](./images/output-OSU-Valley_to_Les_Demoiselles_dAvignon-g1-50.jpg 'output-OSU-Valley_to_Les_Demoiselles_dAvignon-g1-50')|
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
