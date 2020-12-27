import numpy 
import math
import cv2
import os
import glob
input_image = "overall dataset"
output_image = "Pre-processed images"
sum1 = 0
image_files = glob.glob(os.path.join(input_image, "*"))

for (i, image_file) in enumerate(image_files):
   filename = os.path.basename(image_file)
   original = cv2.imread(image_file)
   contrast = cv2.imread(os.path.join(output_image,filename))
   def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
     return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

   d=psnr(original,contrast)
   sum1 = sum1+d
avg = sum1/1070
print("Average PSNR ratio",avg)
