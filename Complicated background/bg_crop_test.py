
import time
import os
import os.path
import cv2
import glob
import numpy as np
import timeit
from PIL import Image

input_folder = "Pre-processed images"
#dilation_output_folder = "Dilation"
output_folder = "Letter extracted images2"
input_files = glob.glob(os.path.join(input_folder, "*"))
counts = {}
start_time = time.time()
"""for (i, input_file) in enumerate(input_files):
  filename = os.path.basename(input_file)  
  img = cv2.imread(input_file) 
  kernel = np.ones((5,5), np.uint8) 
  erosion_image = cv2.erode(img, kernel, iterations=1) 
  dilation_image = cv2.dilate(img, kernel, iterations=1) 
  save_path = os.path.join(dilation_output_folder)
  p = os.path.join(save_path, filename)
  cv2.imwrite(p,dilation_image)
  cv2.waitKey(0)"""

input_files = glob.glob(os.path.join(input_folder, "*"))
for (i, input_file) in enumerate(input_files):
        img = Image.open(input_file)
        filename = os.path.basename(input_file)
        correct_text = os.path.splitext(filename)[0]
        #print (captcha_correct_text)
        width, height = img.size
        left = 0
        top = 0
        right = (width-1)/4
        bottom = height - 1
        im1 = img.crop((left,top,right,bottom))
        save_path1 = os.path.join(output_folder, correct_text[0])
        if not os.path.exists(save_path1):
                os.makedirs(save_path1)

        count = counts.get(correct_text[0], 1)
        p = os.path.join(save_path1, "{}.png".format(str(count).zfill(6)))
        image1 = np.asarray(im1)
        cv2.imwrite(p, image1)

        counts[correct_text[0]] = count + 1

#############################################################

        left = (width-1)/4
        top = 0
        right = (width-1)/6*2.21
        bottom = height - 1
        im1 = img.crop((left,top,right,bottom))
        save_path2 = os.path.join(output_folder, correct_text[1])
        if not os.path.exists(save_path2):
                os.makedirs(save_path2)

        count = counts.get(correct_text[1], 1)
        p = os.path.join(save_path2, "{}.png".format(str(count).zfill(6)))
        image1 = np.asarray(im1)
        cv2.imwrite(p, image1)

        counts[correct_text[1]] = count + 1

#####################################################################

        left = (width-1)/6*2.21
        top = 0
        right = (width-1)/6*3.1
        bottom = height - 1
        im1 = img.crop((left,top,right,bottom))
        save_path3 = os.path.join(output_folder, correct_text[2])
        if not os.path.exists(save_path3):
                os.makedirs(save_path3)

        count = counts.get(correct_text[2], 1)
        p = os.path.join(save_path3, "{}.png".format(str(count).zfill(6)))
        image1 = np.asarray(im1)
        cv2.imwrite(p, image1)

        counts[correct_text[2]] = count + 1

########################################################################

        left = (width-1)/6*3.1
        top = 0
        right = (width-1)/6*3.8
        bottom = height - 1
        im1 = img.crop((left,top,right,bottom))
        save_path4 = os.path.join(output_folder, correct_text[3])
        if not os.path.exists(save_path4):
                os.makedirs(save_path4)

        count = counts.get(correct_text[3], 1)
        p = os.path.join(save_path4, "{}.png".format(str(count).zfill(6)))
        image1 = np.asarray(im1)
        cv2.imwrite(p, image1)

        counts[correct_text[3]] = count + 1

##########################################################################

        """left = (width-1)/6*3.3
        top = 0
        right = (width-1)/6*4
        bottom = height - 1
        im1 = img.crop((left,top,right,bottom))
        save_path5 = os.path.join(output_folder, correct_text[4])
        if not os.path.exists(save_path5):
                os.makedirs(save_path5)

        count = counts.get(correct_text[4], 1)
        p = os.path.join(save_path5, "{}.png".format(str(count).zfill(6)))
        image1 = np.asarray(im1)
        cv2.imwrite(p, image1)

        counts[correct_text[4]] = count + 1"""
