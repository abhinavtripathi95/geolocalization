# for viewing the matches between uav and sat image
# with DISTANCE < 200 test + RT

import pickle
import numpy as np 
from scipy.spatial.distance import cdist
import cv2
from matplotlib import pyplot as plt

def draw_keypoints(img_path, keypoints):
    img = cv2.imread(img_path,1)
    for i in keypoints:
        cv2.circle(img, (int(i[0]), int(i[1])), 1, (0,255,100), 2)
    return img

def draw_matches(img, uav_kp, sat_kp):
    print('image_shape', img.shape)
    img_shape = (480,480)
    lineThickness = 2
    sat_kp[:,0] = sat_kp[:,0] + img_shape[1]
    # print(np.max(kp_trg))

    for i in range(len(uav_kp)):
        qry_tuple = (int(uav_kp[i,0]), int(uav_kp[i,1]))
        trg_tuple = (int(sat_kp[i,0]), int(sat_kp[i,1]))
        cv2.line(img, qry_tuple, trg_tuple, (0,255,0), lineThickness)
    return img

def display_matches(uav_img_path, uav_match_kp, sat_img_path, sat_match_kp):
    uav_img = draw_keypoints(uav_img_path, uav_match_kp)
    sat_img = draw_keypoints(sat_img_path, sat_match_kp)
    img = np.concatenate((uav_img, sat_img), axis = 1)
    img2display = draw_matches(img, uav_match_kp, sat_match_kp)
    plt.imshow(cv2.cvtColor(img2display, cv2.COLOR_BGR2RGB))
    plt.show()
    # plt.waitforbuttonpress()

class ImagePair:
    def __init__(self, uav_img_path, uav_kp, uav_descr, sat_img_path, sat_kp, sat_descr, city, matching):
        self.uav_img_path = uav_img_path
        self.uav_kp = uav_kp
        self.uav_descr = uav_descr
        self.sat_img_path = sat_img_path
        self.sat_kp = sat_kp
        self.sat_descr = sat_descr
        self.city = city
        self.matching = matching

    def eval_descr(self):
        # find the number of best descriptor matches
        uav_len = len(self.uav_kp)
        sat_len = len(self.sat_kp)
        # best matches = good performance on RT + <200 distance
        dist_descr = cdist(self.uav_descr, self.sat_descr, 'euclidean')
        # take the best and second best
        min_idx1 = np.argmin(dist_descr, axis = 1)              # indices of the closest descriptors
        min_dist1 = dist_descr[np.arange(uav_len), min_idx1]

        dist_descr[np.arange(uav_len), min_idx1] = float('inf')
        min_idx2 = np.argmin(dist_descr, axis = 1)              # indices of the second best match
        min_dist2 = dist_descr[np.arange(uav_len), min_idx2]

        RT = np.divide(min_dist1, min_dist2)
        mask = np.logical_and(min_dist1<200, RT<0.75)
        print(np.where(mask == True)) # uav image kps index
        print(sum(mask)) # this is the number of matches we get
        print(min_dist1[mask])
        print(min_idx1[mask]) # sat image kps

        uav_match_kp = self.uav_kp[np.where(mask == True)]
        sat_match_kp = self.sat_kp[min_idx1[mask]]
        print(uav_match_kp)
        print(sat_match_kp)

        # display these matches
        display_matches(self.uav_img_path, uav_match_kp, self.sat_img_path, sat_match_kp)
        # img_path = im_folder + '/' + cam + str(img_no) + '.png'
        # img = cv2.imread(img_path, 1)
        # # cv2.imshow('', img)
        # keypoints = kp_batch[img_no][3]
        # for i in keypoints:
        #     cv2.circle(img, (int(i[0]), int(i[1])), 1, (0,255,100), 2)
        # plt.imshow(img)
        # plt.waitforbuttonpress()



if __name__ == "__main__":
    file_path1 = 'results/sift_atlanta_uav0'
    file_path2 = 'results/sift_atlanta_sat0'

    with open(file_path1, 'rb') as f:
        f1 = pickle.load(f)
    img_data_uav = f1[0]
    uav_kp = img_data_uav[3]
    uav_descr = img_data_uav[4]
    del f1
    
    with open(file_path2, 'rb') as f:
        f2 = pickle.load(f)
    img_data_sat = f2[0]
    sat_kp = img_data_sat[3]
    sat_descr = img_data_sat[4]
    del f2

    im_path1 = 'train/atlanta/atlanta_uav/uav/uav5.png'
    im_path2 = 'train/atlanta/atlanta_sat/sat300/sat5.png'

    im_pair = ImagePair(im_path1, uav_kp, uav_descr, im_path2, sat_kp, sat_descr, 'atlanta', True)
    im_pair.eval_descr()