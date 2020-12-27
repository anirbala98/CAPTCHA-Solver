import numpy as np
import argparse
import cv2
import os
import os.path
import glob
import time
import imutils


input_folder = "Letter extracted images"
output_folder = "Upright images"
for x in range(2,10):
 x = str(x)
 input_files = glob.glob(os.path.join(input_folder, x, "*"))
 for (i, input_file) in enumerate(input_files):
            rimage = cv2.imread(input_file)
            filename1 = os.path.basename(input_file)
            #print (filename1)
            gray = cv2.cvtColor(rimage, cv2.COLOR_BGR2GRAY)
            gray = cv2.bitwise_not(gray)
            thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            coords = np.column_stack(np.where(thresh > 0))
            angle = cv2.minAreaRect(coords)[-1]

            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = rimage.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(rimage, M, (w, h),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            save_path = os.path.join(output_folder, x)
            if not os.path.exists(save_path):
              os.makedirs(save_path)
            p = os.path.join(save_path, filename1)
            cv2.imwrite(p, rotated)

            
alpha = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
for x in alpha:
 input_files1 = glob.glob(os.path.join(input_folder, x, "*"))
 for (i, input_file1) in enumerate(input_files1):
            rimage = cv2.imread(input_file1)
            filename1 = os.path.basename(input_file1)
            gray = cv2.cvtColor(rimage, cv2.COLOR_BGR2GRAY)
            gray = cv2.bitwise_not(gray)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            coords = np.column_stack(np.where(thresh > 0))
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            (h, w) = rimage.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(rimage, M, (w, h),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            save_path = os.path.join(output_folder, x)
            if not os.path.exists(save_path):
              os.makedirs(save_path)
            p = os.path.join(save_path, filename1)
            cv2.imwrite(p, rotated)



