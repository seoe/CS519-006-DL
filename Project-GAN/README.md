## GAN

|Project|Algorithm|venv|dataset|result|
|:--:|:--:|:--:|:--:|:--:|
|AC-GAN-burness-tf / [repo](https://github.com/burness/tensorflow-101/tree/master/GAN/AC-GAN)|ACGAN|||:sleepy:|
|[DCGAN-carpedm20-tf](#dcgan-carpedm20-tensorflow) / [repo](https://github.com/carpedm20/DCGAN-tensorflow)|DCGAN|envDL|mnist+celebA|:ok_hand:|
|DCGAN-soumith-torch-lsun+ImageNet / [repo](https://github.com/soumith/dcgan.torch)|DCGAN|||:sleepy:|
|[Info-GAN-burness-tf](#info-gan-burness-tensorflow) / [repo](https://github.com/burness/tensorflow-101/tree/master/GAN/Info-GAN0)|Info-GAN|envDL|mnist|:ok_hand:|
|[WassersteinGAN](#wassersteingan) / [repo](https://github.com/martinarjovsky/WassersteinGAN)|WGAN|envDL/envTorch|LSUN+mnist|:sleepy:|
|[StyleSynthesis-machrisaa-tf+VGG19](#stylesynthesis-machrisaa-tensorflowvgg19) / [repo](https://github.com/machrisaa/stylenet)|-|keras_theano_py2|random img|:ok_hand:|

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

After group meeting, I tried mnist dataset and got some result with can be used as gif to illustrate.

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
With DCGAN:
```zsh
python main.py --dataset lsun --dataroot [lsun-train-folder] --cuda
```
With MLP:
```zsh
python main.py --mlp_G --ngf 512
```
After run `python main.py.....`, there is an error `ImportError: No module named lmdb`. So I install lmdb with `pip install lmdb`.  
Then, run the main, turns out that LSUN dataset(bedroom) not found. After research, pytorch offer download of mnist, not for lsun, which make users do it manually. So I find a [repo](https://github.com/fyu/lsun) for downloading LSUN-bedroom data. **rember**, do not use `wget` on encrypted url, I just need to open 'raw' of a py code then use `wget` to download a file seperately.  
So I run `python download.py -c bedroom` to start downloading the two 45GB zip files, which takes 1.5 hours. But it failed and left about a hafl undownloaded. With that uncompleted zip I cannot unzip it.


#### 03/10/2017 Fri

After several trial of downloading with ssh and tmux, I turned to the desktop and login my accout directly to run the download code. Finally it works. And I can unzip them successfully.

#### 03/11/2017 Sat
In the morning, I put both unzipped LSUN data folder (`bedroom_train_lmdb`/45GB->54GB,`bedroom_val_lmdb`/4MB->6MB) into `/WassersteinGAN`.   
Since the `lsun-train-folder` is in `./`, the same path as `main.py`, so the run command `python main.py --dataset lsun --dataroot [lsun-train-folder] --cuda` should be
```bash
python main.py --dataset lsun --dataroot './' --cuda
```
As is mentioned in the repo's note, 
> The first time running on the LSUN dataset it can take a long time (up to an hour) to create the dataloader. After the first run a small cache file will be created and the process should take a matter of seconds. The cache is a list of indices in the lmdb database (of LSUN)

It really took a while to create the dataloader.
[***Back*** to subcontents ***GAN***](#gan)  

### StyleSynthesis-machrisaa-tensorflow+VGG19

#### 03/10/2017 Fri

clone the [repo](https://github.com/machrisaa/stylenet), run and fail.   
Then I modified the import function/files, which is not maitained for one year. Moreover, the stylesyn project relies on vgg19 project. So I download the vgg19.py file into the folder as import.

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
Then I find I need `vgg19.npy` downloaded into the same path. So a 574.7 MB file is downloaded from cloud disk and copied to TitanX server.

Since TitanX is occupied by Jialin and Lawrance fully. I turned to pelican server. However, `.meta` files are generated frequently, each of which has the size 500MB~2GB, making the server stopped me from continuing.    
To disable the creation of the `.meta` file, I need to add `write_meta_graph=False`, in line 237 in `stylenet_patch.py`,Like this:  
```python
saved_path = saver.save(sess, "./train/saves-" + get_filename(content_file), global_step=global_step, write_meta_graph=False)
```
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
But the pelican is also fully occupied by a guy alizades.  
Then I have to switch to steed instead.

After some environment configuration, I upgrade the numpy then I can import tensorflow successfully in steed.

Some results from OSU landscapes are generated and saved to my own computer.

[***Back*** to subcontents ***GAN***](#gan)  
[***Back to CONTENTS***](#contents)  
