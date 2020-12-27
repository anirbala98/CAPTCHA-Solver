import os
import os.path
import glob
import cv2
import numpy as np

input_dest = "overall dataset"
output_dest = "Pre-processed images"

imagefiles = glob.glob(os.path.join(input_dest, "*"))

for (i, imagefile) in enumerate(imagefiles):
    
    filename = os.path.basename(imagefile)
    img = cv2.imread(imagefile,2)
    kernel = np.ones((3,3), np.uint8)
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,bw_img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    save_path = os.path.join(output_dest)
    p = os.path.join(save_path, filename)
    cv2.imwrite(p,bw_img)

    cv2.waitKey(0)
