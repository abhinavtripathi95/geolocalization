#!/usr/bin/env python
# coding: utf-8

# In[2]:
# for extracting features in train images


import os
import cv2
import numpy as np
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm


# In[3]:


cities = os.listdir('train')
print(cities)


# In[4]:


no_of_imgs = []
for city in cities:
    uav_folder = 'train/' + city + '/' + city + '_uav/uav'
    no_of_imgs.append(len(os.listdir(uav_folder)))
print(no_of_imgs)
print(sum(no_of_imgs))
# plt.bar(range(len(cities)),no_of_imgs)
 #plt.ylabel('Number of Images')
# plt.xticks(range(len(cities)),cities, rotation=90)
# plt.show()


# In[5]:


# function to get image paths in numeric order
import re
def atoi(text):
    if text.isdigit():
        return int(text)
    else:
        return text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]

def save_cache(city, cam, cache, batch_number):
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    file_name = results_dir + '/sift_' + city + '_' + cam + str(batch_number)
    with open(file_name, 'wb') as f:
        pickle.dump(cache, f)


# In[ ]:


sift = cv2.xfeatures2d.SIFT_create()

for city in tqdm(cities):
    cache_uav = []
    cache_sat = []

    uav_folder = 'train/' + city + '/' + city + '_uav/uav'
    sat_folder = 'train/' + city + '/' + city + '_sat/sat300'
    uav_img_paths = os.listdir(uav_folder)
    sat_img_paths = os.listdir(sat_folder)
    uav_img_paths.sort(key = natural_keys)
    sat_img_paths.sort(key = natural_keys)
    
    batch_number = 0
    im_number = 0
    for uav_img_path in tqdm(uav_img_paths):
        img_path = uav_folder + '/' + uav_img_path
        img = cv2.imread(img_path,1) # color
#         print(img_path)
        kp, descr = sift.detectAndCompute(img, None)
        keypoint = cv2.KeyPoint_convert(kp)
        cache_uav.append((city, 'uav', uav_img_path, keypoint, descr))
        im_number = im_number + 1
        if (im_number >= 1000):
            save_cache(city, 'uav', cache_uav, batch_number)
            cache_uav = []
            im_number = 0
            batch_number = batch_number + 1
    
    save_cache(city, 'uav', cache_uav, batch_number)
    del cache_uav
    
    batch_number = 0
    im_number = 0
    for sat_img_path in tqdm(sat_img_paths):
        img_path = sat_folder + '/' + sat_img_path
        img = cv2.imread(img_path,1) # color
        kp, descr = sift.detectAndCompute(img, None)
        keypoint = cv2.KeyPoint_convert(kp)
        cache_sat.append((city, 'sat', sat_img_path, keypoint, descr))
        im_number = im_number + 1
        if (im_number >= 1000):
            save_cache(city, 'sat', cache_sat, batch_number)
            cache_sat = []
            im_number = 0
            batch_number = batch_number + 1
        
    save_cache(city, 'sat', cache_sat, batch_number)
    del cache_sat


# In[ ]:




