#!/usr/bin/env python3.6
# coding: utf-8
import os

# # Notebook II: Playing with Latent Codes

# ![alt text](https://raw.githubusercontent.com/tr1pzz/InterFaceGAN/master/teaser.jpg)

# # OK, first, the really annoying part:
# ### Google Colab uses TensorFlow version 1.14 by default (which comes with Cuda 10.0)
# ### Unfortunately the repo we'll be using requires TF version 1.12 and Cuda 9.0...

# In[1]:


os.system('pip install tensorflow==1.12')
import tensorflow as tf
print(tf.__version__)


# In[2]:


os.system('nvcc --version')


# ## Annoying part 1 of 2: Downgrade to TensorFlow 1.12.2

# In[3]:


os.system('pip install --upgrade tensorflow-gpu==1.12.2')


# ## Annoying part 2 of 2: Install Cuda 9.0 (this can take a few minutes, be patient)

# In[1]:


os.system('wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb')
os.system('dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb')
os.system('apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub')
os.system('apt-get update')
os.system('apt-get install cuda=9.0.176-1')
os.system('echo ****** Cuda reinstall completed. Restart runtime now! *******')


# ## When the above cell has finished executing, restart the kernel runtime to reload everything
# > * At the top, click 'Runtime --> restart runtime'
# > * Continue with the cells below!
# > * In fact, let's just pretend that annoying part never happened...

# ### Safety check to see if everything worked:
# (This might not work as intended due to TF 1.12 being a very old / deprecated version of TensorFlow...) 

# In[3]:


os.system('pip install tensorflow==1.12')
import tensorflow as tf

#print("Now running TensorFlow version %s on Colab!" %tf.VERSION)
#assert tf.VERSION == '1.12.2'


# In[4]:


os.system('nvcc --version')


# ## If the above cells showed TensorFlow version 1.12.2 and Cuda release 9.0, you're good to go!
# ![alt text](https://media.giphy.com/media/CjmvTCZf2U3p09Cn0h/giphy.gif)

# ## Clone my fork of InterFaceGAN
# ### Original Repo: https://github.com/ShenYujun/InterFaceGAN
# 
# ### Paper: https://arxiv.org/abs/1907.10786

# In[5]:


os.system('git clone https://github.com/tr1pzz/InterFaceGAN.git')


# In[6]:


cd /content/InterFaceGAN/


# ## Download the pretrained StyleGAN FFHQ network from NVIDIA:

# In[7]:


os.system('gdown https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ')
os.system('mv /content/InterFaceGAN/karras2019stylegan-ffhq-1024x1024.pkl /content/InterFaceGAN/models/pretrain/karras2019stylegan-ffhq-1024x1024.pkl')


# # I. Let's load our latent space vectors:
# Manually upload the output_vectors.npy file (the one you downloaded at the end of notebook I) to the root of the directory
# *   Right-click anywhere inside the Files browser --> "Upload"
# *   Make sure the filename is "output_vectors.npy"
# 
# 

# In[ ]:


import numpy as np
final_w_vectors = np.load('/content/output_vectors.npy')

print("%d latent vectors of shape %s loaded from %s!" %(final_w_vectors.shape[0], str(final_w_vectors.shape[1:]), 'output_vectors.npy'))


# ## The InterFaceGAN comes with a bunch of pretrained latent directions
# ### (However, you can also train your own!!)
# ### Pick the latent space manipulation we want to use (added it as the -b argument below)

# Boundaries: https://github.com/ShenYujun/InterFaceGAN/tree/master/boundaries
# * stylegan_ffhq_age_w_boundary.npy
# * stylegan_ffhq_eyeglasses_w_boundary.npy
# * stylegan_ffhq_gender_w_boundary.npy
# * stylegan_ffhq_pose_w_boundary.npy
# * stylegan_ffhq_smile_w_boundary.npy
# 

# # II. Let's configure our latent-space interpolation
# ### Change the settings below to morph the faces:

# In[ ]:


latent_direction = 'age'     #### Pick one of ['age', 'eyeglasses', 'gender', 'pose', 'smile']
morph_strength = 3           # Controls how strongly we push the face into a certain latent direction (try 1-5)
nr_interpolation_steps = 48  # The amount of intermediate steps/frames to render along the interpolation path


# # III. Run the latent space manipulation & generate images:

# In[ ]:


boundary_file = 'stylegan_ffhq_%s_w_boundary.npy' %latent_direction

print("Ready to start manipulating faces in the ** %s ** direction!" %latent_direction)
print("Interpolation from %d to %d with %d intermediate frames." %(-morph_strength, morph_strength, nr_interpolation_steps))
print("\nLoading latent directions from %s" %boundary_file)


# 
# ## Final note: The code cell below has a bug I still need to fix...
# ### First time you run it, it will give an error.
# ### ----> Don't worry: just run the same cell again and it should work :p
# ![alt text](https://media1.tenor.com/images/379faefe7d906603844c3c073b290814/tenor.gif?itemid=5108830)

# ## Ready? Set, Go!

# In[ ]:


import subprocess
return_code = subprocess.call("rm -r results/%s" %latent_direction, shell=True)

run_command = "python edit.py \
      -m stylegan_ffhq \
      -b boundaries/stylegan_ffhq_%s_w_boundary.npy \
      -s Wp \
      -i '/content/output_vectors.npy' \
      -o results/%s \
      --start_distance %.2f \
      --end_distance %.2f \
      --steps=%d" %(latent_direction, latent_direction, -morph_strength, morph_strength, nr_interpolation_steps)


print("Running latent interpolations... This should not take longer than ~1 minute")
print("Running: %s" %run_command)
return_code = subprocess.call(run_command, shell=True)

if not return_code:
  print("Latent interpolation successfully dumped to disk!")
else:
  print("Something went wrong, try re-executing this cell...")


# ### If you're still getting errors, run the command with output to see what's wrong: (Adjust the arguments below as needed)
# ### Otherwise ---> just skip this cell

# In[ ]:


if 0:
  latent_direction = 'age'
  os.system('rm -r results/age')
  os.system("python edit.py      -m stylegan_ffhq      -b boundaries/stylegan_ffhq_age_w_boundary.npy      -s Wp      -i '/content/output_vectors.npy'      -o results/age      --start_distance -3.0      --end_distance 3.0      --steps=48")


# # IV. Finally, turn the results into pretty movies!
# Adjust which video to render & at what framerate:

# In[ ]:


image_folder = '/content/InterFaceGAN/results/%s' %latent_direction
video_fps = 12.


# ### Render the videos:

# In[ ]:


from moviepy.editor import *
import cv2

out_path = '/content/output_videos/'

images = [img_path for img_path in sorted(os.listdir(image_folder)) if '.jpg' in img_path]
os.makedirs(out_path, exist_ok=True)

prev_id = None
img_sets = []
for img_path in images:
  img_id = img_path.split('_')[0]
  if img_id == prev_id: #append
    img_sets[-1].append(img_path)
    
  else: #start a new img set
    img_sets.append([])
    img_sets[-1].append(img_path)
  prev_id = img_id

print("Found %d image sets!\n" %len(img_sets))
if image_folder[-1] != '/':
  image_folder += '/'

def make_video(images, vid_name):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(vid_name, fourcc, video_fps, (1024, 1024))
    gen = {}
    for img in images:
      video.write(img)
    video.release()
    print('finished '+ vid_name)
    
    
for i in range(len(img_sets)):
  print("############################")
  print("\nGenerating video %d..." %i)
  set_images = []
  vid_name = out_path + 'out_video_%s_%02d.mp4' %(latent_direction,i)
  for img_path in img_sets[i]:
    set_images.append(cv2.imread(image_folder + img_path))

  set_images.extend(reversed(set_images))
  make_video(set_images, vid_name)


# # So... What did we get?
# ![alt text](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTiWSY0xWfAJvKOBUhvVFcewroGDzAe1lNMfi73EAJp4IBJ-6zi)

# ## Option 1: 
# > ### Navigate to output_videos/
# > (you might have to "REFRESH" the Filebrowser)
# > ### Download the videos to your local pc
# > (This makes viewing a bit easier + you can share them :p)

# ## Option 2:
# > ### Display the videos right here in the notebook

# ## Visualise the resulting videos inside this Notebook:

# In[ ]:


video_file_to_show = 0

clip = VideoFileClip('/content/output_videos/out_video_%s_%02d.mp4' %(latent_direction, video_file_to_show))
clip.ipython_display(height=512, autoplay=1, loop=1)


# In[ ]:


video_file_to_show = 1

clip = VideoFileClip('/content/output_videos/out_video_%s_%02d.mp4' %(latent_direction, video_file_to_show))
clip.ipython_display(height=512, autoplay=1, loop=1)


# # V. Your turn to experiment
# 
# ## You now have all the tools to start exploring the latent Space of StyleGAN: HAVE FUN!
# ### StyleGAN paper link: https://arxiv.org/abs/1812.04948
# 
# ### Some things you could try:
# * You can blend between two faces by doing a linear interpolation in the latent space: very cool!
# *   The StyleGAN vector has 18x512 dimensions, each of those 18 going into a different layer of the generator...
# *   You could eg take the first 9 from person A and the next 9 from person B
# *   This is why it's called "Style-GAN": you can manipulate the style of an image at multiple levels of the Generator!
# *   Try interpolating in Z-space rather than in W-space (see InterFaceGan paper & repo)
# * Have Fun!!
# # Find something cool you wanna share? 
# ## ---> Tag me on Twitter @xsteenbrugge: https://twitter.com/xsteenbrugge
# ## ---> Or simply share it in the comments on YouTube!

# ![alt text](https://media.makeameme.org/created/experiment.jpg)
