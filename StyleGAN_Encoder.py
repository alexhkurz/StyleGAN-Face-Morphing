#!/usr/bin/env python
# coding: utf-8

# # Colab Notebook for my video "Face editing with Generative Adversarial Networks": 
# https://www.youtube.com/watch?v=dCKbRCUyop8
# 

# # Part I: Encoding images into StyleGAN's latent space

# ![alt text](https://miro.medium.com/max/1280/0*eeFaGLx96mlbQcrK.gif)

# # Start: To run this demo StyleGAN code, make a local copy of this notebook in your drive!

# ## Before you move on, make sure you have GPU acceleration enabled:
# > ### Click 'Runtime' in the menu tab at the top
# > ### Click 'Change runtime type'
# > ### Make sure the hardware accelerator is set to 'GPU'

# # This is a hosted IPython notebook
# ### For those not familiar, all you really need to know is:
# * There are text cells (like this one) 
# * And Python code cells (like the one below)
# * If you want the full tutorial, you can find it here: https://colab.research.google.com/notebooks/welcome.ipynb
# 
# ### To run a code cell, simply click inside it and hit "shift+enter"
# (hitting shift + enter repeatedly will run through all the cells sequentially)
# ### Don't worry if you're new to this: 
# > If things don't work, you can always click "Runtime-->Reset all runtimes" and restart the whole notebook if you mess up!

# In[1]:


a = 20
b = 30
c = a+b
print("The sum of %d and %d is %d." %(a,b,c))


# ## --EDIT-- Colab tqdm version fix:

# 1. Upgrade tqdm:

# In[2]:


import os
os.system('pip install --upgrade tqdm')


# 2. Restart the Python kernel to load the updated version:

# In[ ]:


import os 
os.kill(os.getpid(), 9)


# ## Now we can start for real:
# ### Let's first clone the Github repo we'll use: https://github.com/pbaylies/stylegan-encoder

# In[1]:


os.system('rm -rf sample_data')
os.system('git clone https://github.com/pbaylies/stylegan-encoder')


# ### cd into the repo folder: (only run this cell once or things might get buggy)

# In[2]:


os.chdir('stylegan-encoder')


# ### Let's see the files inside the repo we just cloned:

# In[3]:


ls


# ### Some housekeeping: setting up folder structure for our images:

# In[ ]:


rm -rf aligned_images raw_images


# In[ ]:


mkdir aligned_images raw_images


# # I. Get Images:

# ## Some tips for the images:
# 
# 
# *   Use HD images (preferably > 1000x1000 pixels)
# *   Make sure your face is not too small
# *   Neutral expressions & front facing faces will give better results
# *   Clear, uniform lighting conditions are also recommened
# 
# 

# ## Option 1: Upload Images manually (usually gives the best results)

# 
# 
# *   Click the '>' icon in the panel on the top left 
# *   Go to the 'Files' tab
# *   Unfold the stylegan-encoder folder (left-click)
# *   Right click the 'stylegan-encoder/raw_images' folder and click "upload"
# *   I'd recommend starting with 3 - 6 different images containing faces
# 
# 

# ## Option 2: Take images using your webcam

# In[ ]:


from IPython.display import HTML, Audio
from google.colab.output import eval_js
from base64 import b64decode
import numpy as np
import io
from PIL import Image
from datetime import datetime

VIDEO_HTML = """
<video autoplay
 width=%d height=%d style='cursor: pointer;'></video>
<script>

var video = document.querySelector('video')

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream=> video.srcObject = stream)
  
var data = new Promise(resolve=>{
  video.onclick = ()=>{
    var canvas = document.createElement('canvas')
    var [w,h] = [video.offsetWidth, video.offsetHeight]
    canvas.width = w
    canvas.height = h
    canvas.getContext('2d')
          .drawImage(video, 0, 0, w, h)
    video.srcObject.getVideoTracks()[0].stop()
    video.replaceWith(canvas)
    resolve(canvas.toDataURL('image/jpeg', %f))
  }
})
</script>
"""

def take_photo(quality=1.0, size=(800,600)):
  display(HTML(VIDEO_HTML % (size[0],size[1],quality)))
  data = eval_js("data")
  binary = b64decode(data.split(',')[1])
  f = io.BytesIO(binary)
  img = np.asarray(Image.open(f))
  
  timestampStr = datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
  filename = 'raw_images/photo_%s.jpeg' %timestampStr
  Image.fromarray(img).save(filename)
  print('Image captured and saved to %s' %filename)


# In[ ]:


img = take_photo() # click the image to capture a frame!


# ## Let's check the contents of our image folder before we start:
# #### (You can always manually delete images by right clicking on them in the file tab)

# In[6]:


from PIL import Image
import os
imgs = sorted(os.listdir('raw_images'))

print("Found %d images in %s" %(len(imgs), 'raw_images'))
if len(imgs) == 0:
  print("Upload images to the \"raw_images\" folder!")
else:
  print(imgs)

for img_path in imgs:
  img = Image.open('raw_images/' + img_path)
  
  w,h = img.size
  rescale_ratio = 256 / min(w,h)
  img = img.resize((int(rescale_ratio*w),int(rescale_ratio*h)), Image.LANCZOS)
  display(img)


# ## Make sure we're using the right TensorFlow version (1.15):

# In[7]:


import tensorflow as tf
print(tf.__version__)


# # II. Auto-Align faces:
# ### This script wil:
# 
# 
# 1.   Look for faces in the images
# 2.   Crop out the faces from the images
# 3.   Align the faces (center the nose and make the eyes horizontal)
# 4.   Rescale the resulting images and save them in "aligned_images" folder
# 
# ### The cell below takes about a minute to run
# 
# 

# In[8]:


get_ipython().system('python align_images.py raw_images/ aligned_images/ --output_size=1024')


# ## Let's take a look at our aligned images:

# In[9]:


def display_folder_content(folder, res = 256):
  if folder[-1] != '/': folder += '/'
  for i, img_path in enumerate(sorted(os.listdir(folder))):
    if '.png' in img_path:
      display(Image.open(folder+img_path).resize((res,res)), 'img %d: %s' %(i, img_path))
      print('\n')
      
display_folder_content('aligned_images')


# # Important, before moving on:
# ### Manually clean the 'aligned_images' directory
# 
# > ### 1. Manually remove all 'bad' images that are not faces / don't look sharp / clear 
# > #####  (Use the image names from the plots above to guide you)
# > ### 2. Make sure you don't have too many faces in this folder (8 at most preferably)
# 
# 
# 

# # Encoding faces into StyleGAN latent space:

# ![title](https://raw.githubusercontent.com/pbaylies/stylegan-encoder/master/mona_example.jpg)

# ## We'll be using pbaylies' awesome encoder repo (building on original work from Puzer): https://github.com/pbaylies/stylegan-encoder
# 

# ## First, let's download a pretrained resnet encoder: (see video for what this does)
# ### --> This model takes an image as input and estimates the corresponding latent code

# In[10]:


get_ipython().system('gdown https://drive.google.com/uc?id=1aT59NFy9-bNyXjDuZOTMl0qX0jmZc6Zb')
get_ipython().system('mkdir data')
get_ipython().system('mv finetuned_resnet.h5 data')
get_ipython().system('rm -rf generated_images latent_representations')


# # III. The actual encoding process:
# > #### Highly recommended: play with the encoding params: they have a huge effect on the latent representations & images!
# > #### Extra encoding options: https://github.com/pbaylies/stylegan-encoder/blob/master/encode_images.py
# 
# #### Note: This script will also download:
# 
# 
# *   The pretrained StyleGAN network from NVIDIA trained on faces
# *   A pretrained VGG-16 network, trained on ImageNet
# 
# #### After guessing the initial latent codes using the pretrained ResNet, it will run gradient descent to optimize the latent faces!
# #### Note that by default, we're optimizing w vectors, not z-vectors!
# 

# In[11]:


print("aligned_images contains %d images ready for encoding!" %len(os.listdir('aligned_images/')))
print("Recommended batch_size for the encode_images process: %d" %min(len(os.listdir('aligned_images/')), 8))


# #### Important: to avoid issues, set the batch_size argument lower than or equal to the number of aligned_images (see previous cell)
# > Keep batch_size<8 or the GPU might run out of memory
# 
# ### Depending on the settings, the encoding process might take a few minutes...

# ## Fast version:

# In[ ]:


get_ipython().system('python encode_images.py --optimizer=lbfgs --face_mask=True --iterations=6 --use_lpips_loss=0 --use_discriminator_loss=0 --output_video=True aligned_images/ generated_images/ latent_representations/')
print("\n************ Latent code optimization finished! ***************")


# ## Slow version:

# In[15]:


get_ipython().system('python encode_images.py --optimizer=adam --lr=0.02 --decay_rate=0.95 --decay_steps=6 --use_l1_penalty=0.3 --face_mask=True --iterations=400 --early_stopping=True --early_stopping_threshold=0.05 --average_best_loss=0.5 --use_lpips_loss=0 --use_discriminator_loss=0 --output_video=True aligned_images/ generated_images/ latent_representations/')
print("\n************ Latent code optimization finished! ***************")


# ## Showtime!
# ### Let's load the StyleGAN network into memory:

# In[13]:


import dnnlib, pickle
import dnnlib.tflib as tflib
tflib.init_tf()
synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=1)

model_dir = 'cache/'
model_path = [model_dir+f for f in os.listdir(model_dir) if 'stylegan-ffhq' in f][0]
print("Loading StyleGAN model from %s..." %model_path)

with dnnlib.util.open_url(model_path) as f:
  generator_network, discriminator_network, averaged_generator_network = pickle.load(f)
  
print("StyleGAN loaded & ready for sampling!")


# In[ ]:


def generate_images(generator, latent_vector, z = True):
    batch_size = latent_vector.shape[0]
    
    if z: #Start from z: run the full generator network
        return generator.run(latent_vector.reshape((batch_size, 512)), None, randomize_noise=False, **synthesis_kwargs)
    else: #Start from w: skip the mapping network
        return generator.components.synthesis.run(latent_vector.reshape((batch_size, 18, 512)), randomize_noise=False, **synthesis_kwargs)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
def plot_imgs(model, rows, columns):
  for i in range(rows):
    f, axarr = plt.subplots(1,columns, figsize = (20,8))
    for j in range(columns):
      img = generate_images(model, np.random.randn(1,512), z = True)[0]
      axarr[j].imshow(img)
      axarr[j].axis('off')
      axarr[j].set_title('Resolution: %s' %str(img.shape))
    plt.show()


# ## Let's plot some random StyleGAN samples:

# In[ ]:


plot_imgs(averaged_generator_network, 3, 3)


# # Let's take a look at the results of our encoding:
# ### If the results don't look great: Play with the encoding arguments!!!
# > 1. Run the optimization for more iterations (eg 500)
# > 2. Decrease the L1 penalty (to eg 0.15)
# > 3. Try a lower initial learning rate (eg 0.02) or play with the decay_rate
# > 4. Find out about the other encoding options here: https://github.com/pbaylies/stylegan-encoder/blob/master/encode_images.py
# > 5. You can find a bunch of good presets on the repo documentation: https://github.com/pbaylies/stylegan-encoder

# In[ ]:


import numpy as np

for f in sorted(os.listdir('latent_representations')):
  w = np.load('latent_representations/' + f).reshape((1,18,-1))
  img = generate_images(averaged_generator_network, w, z = False)[0]
  plt.imshow(img)
  plt.axis('off')
  plt.title("Generated image from %s" %f)
  plt.show()


# ## Let's compare our encoded samples with the original ones:
# 
# **Note: when you optimized with the setting --face_mask=True, the hair will be copied from the source images. If you don't want this, optimize without that setting!**

# In[ ]:


import matplotlib.pyplot as plt

def plot_two_images(img1,img2, img_id, fs = 12):
  f, axarr = plt.subplots(1,2, figsize=(fs,fs))
  axarr[0].imshow(img1)
  axarr[0].title.set_text('Encoded img %d' %img_id)
  axarr[1].imshow(img2)
  axarr[1].title.set_text('Original img %d' %img_id)
  plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
  plt.show()

def display_sbs(folder1, folder2, res = 256):
  if folder1[-1] != '/': folder1 += '/'
  if folder2[-1] != '/': folder2 += '/'
    
  imgs1 = sorted([f for f in os.listdir(folder1) if '.png' in f])
  imgs2 = sorted([f for f in os.listdir(folder2) if '.png' in f])
  if len(imgs1)!=len(imgs2):
    print("Found different amount of images in aligned vs raw image directories. That's not supposed to happen...")
  
  for i in range(len(imgs1)):
    img1 = Image.open(folder1+imgs1[i]).resize((res,res))
    img2 = Image.open(folder2+imgs2[i]).resize((res,res))
    plot_two_images(img1,img2, i)
    print("")
     
display_sbs('generated_images/', 'aligned_images/', res = 512)


# ### Note: 
# If you want to watch the whole thing unfold for yourself, you can **download the optimization videos** from the "videos" folder

# # IV. Cherry pick images & dump their latent vectors to disk
# ### Manipulating latent vectors (Notebook II) is tricky and will only work well if the face encoding looks 'good'
# ### Cherry pick a few images where the optimization worked well
# > (Use the image indices from the plot titles above)

# In[ ]:


good_images = [0,1]  #Change these numbers to pick out latents that worked well (see the image plots)


# ## Save these latent vectors to disk:

# In[ ]:


import numpy as np
latents = sorted(os.listdir('latent_representations'))

out_file = '/content/output_vectors.npy'

final_w_vectors = []
for img_id in good_images:
  w = np.load('latent_representations/' + latents[img_id])
  final_w_vectors.append(w)

final_w_vectors = np.array(final_w_vectors)
np.save(out_file, final_w_vectors)
print("%d latent vectors of shape %s saved to %s!" %(len(good_images), str(w.shape), out_file))


# # V. Manipulating the faces
# ### Everything we downloaded / saved to disk is currently on a temporary VM running on Google Colab
# > We'll want to reuse the latent vectors later, so you should download them manually:
# >> * Go to the root directory using the Files browser
# >> * Richt-click & Download the latent representations: "output_vectors.npy"
# ## Next, let's continue with notebook II:
# > ### Simply open the second notebook from the Drive folder and continue the guide-steps
# > ### (Hint: Notebook II is where all the fun is!)

# ![alt text](https://66.media.tumblr.com/tumblr_mc3hg5VpQP1qcy0p7o1_400.gif)
