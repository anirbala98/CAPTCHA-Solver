import os
import os.path
import cv2
import glob
import imutils
import time

input_folder = "Rotated images"
output_folder = "Letter extracted images"

#start_time = time.time()


input_files = glob.glob(os.path.join(input_folder, "*"))
counts = {}

for (i, input_file) in enumerate(input_files):
    filename = os.path.basename(input_file)
    text = os.path.splitext(filename)[0]
    image = cv2.imread(input_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if imutils.is_cv2() else contours[1]
    letter_region = []

    for contour in contours:

        (x, y, w, h) = cv2.boundingRect(contour)
        if w / h > 1.25:
            half_width = int(w / 2)
            letter_region.append((x, y, half_width, h))
            letter_region.append((x + half_width, y, half_width, h))
        else:
        
            letter_region.append((x, y, w, h))

    if len(letter_region) != 4:
        continue

    letter_region = sorted(letter_region, key=lambda x: x[0])

    for bounding_box, letter_text in zip(letter_region, text):

        x, y, w, h = bounding_box
        letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]
        save_path = os.path.join(output_folder, letter_text)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)

  
        counts[letter_text] = count + 1

#end_time = time.time()

#print (end_time - start_time)
