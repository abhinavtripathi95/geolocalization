
# In[]:
# for viewing the keypoints in a particular image

import os
import cv2
import numpy as np
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse

# In[]:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='city, cam, and image number')
    parser.add_argument('city', type = str, help = 'type a city name',default='atlanta')
    parser.add_argument('cam', type = str, help = 'choose uav or sat',default='uav')
    parser.add_argument('img_no', type = int, help = 'type img no',default=0)
    args = parser.parse_args()
    print(args)

    # In[]:
    train_cities = ['champaign', 'chicago', 'boston', \
    'springfield', 'austin', 'miami', 'sanfrancisco', 'stlouis', 'atlanta']
    # print(train_cities)
    test_cities = ['detroit', 'portland', 'orlando']



    # In[]:
    # choose the city and cam
    city = args.city #'champaign'
    cam = args.cam # 'uav'

    # In[]:
    # find the number of images in this city and cam
    if city in train_cities:
        t = 'train'
    else:
        t = 'test'
    im_folder = t + '/' + city + '/' + city + '_' + cam + '/'
    if cam == 'uav':
        im_folder = im_folder + cam
    elif t == 'train':
        im_folder = im_folder + cam + '300'
    else:
        im_folder = im_folder + 'all' + cam

    print('Number of images:', len(os.listdir(im_folder)))

    # In[]:
    img_no = args.img_no #431
    file_no = int(img_no/1000)
    # print()
    if t == 'train':
        kp_file = 'results/sift_' + city + '_' + cam + str(file_no)
    else:
        kp_file = 'results/test/sift_' + city + '_' + cam + str(file_no)



    # In[]:
    # Load the file and the image
    with open(kp_file, 'rb') as f:
        kp_batch = pickle.load(f)
    img_no = img_no%1000
    print(kp_batch[img_no])
    print('Number of keypoints:', len(kp_batch[img_no][3]))

    # In[]:
    # View the image with keypoints marked
    img_path = im_folder + '/' + cam + str(img_no) + '.png'
    img = cv2.imread(img_path, 1)
    # cv2.imshow('', img)
    keypoints = kp_batch[img_no][3]
    for i in keypoints:
        cv2.circle(img, (int(i[0]), int(i[1])), 1, (0,255,100), 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    # %%
