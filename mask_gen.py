"""
Reference article: https://towardsdatascience.com/detecting-face-features-with-python-30385aee4a8e

Requires Python 3.5 (or older)
"""

import cv2
import dlib
import numpy as np
import os
from PIL import Image
import sys


def generate_mask(file, count, convert):
    
    file_noext = file[:file.index(".")]

    # Convert image from jpg to png
    if (convert):
        im1 = Image.open(file)
        new_file = file_noext + ".png"
        im1.save(new_file)
        file = new_file

    # Initial Setup
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("Insights-Mask-Off/shape_predictor_68_face_landmarks.dat")
    # print("Processing: ", file)
    img = cv2.imread(file)
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    face = detector(gray)[0]
    landmarks = predictor(image=img, box=face)

    # Obtain points of interest
    right_jaw_x = landmarks.part(3).x
    right_jaw_y = landmarks.part(3).y

    bot_chin_x = landmarks.part(8).x
    bot_chin_y = landmarks.part(8).y

    left_jaw_x = landmarks.part(13).x
    left_jaw_y = landmarks.part(13).y

    middle_nose_x = landmarks.part(29).x
    middle_nose_y = landmarks.part(29).y

    major_axis_len = left_jaw_x - right_jaw_x
    minor_axis_len = bot_chin_y - middle_nose_y

    # cv2.ellipse(img,(middle_nose_x, middle_nose_y),
    #                (major_axis_len, minor_axis_len),0,0,180,255,-1)

    curve_face = []
    for i in range(1, 16):
        curve_face.append([landmarks.part(i).x, landmarks.part(i).y])

    curve_face.append([middle_nose_x, middle_nose_y])

    #pts = np.array([[100,350],[165,350],[165,240]], np.int32)
    pts = np.array(curve_face, np.int32)
    mask = cv2.fillPoly(img, [pts], 255)

    # print("Processed: ", file)
    # cv2.imshow(winname="Face", mat=mask)
    cv2.imwrite(file_noext + '_m.png', mask)
    # cv2.waitKey(delay=0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        # creating a folder named data
        if not os.path.exists('data'):
            os.makedirs('data')

        inputName = sys.argv[1]

        # if data not created then raise error
    except OSError:
        print('Error: Creating directory of data')

    except IndexError:
        print('Error: Provide an input folder')

    count = 0
    for file in (os.listdir(inputName)):
        # print(file)
        if "_m.png" not in file:
            ext = file.split(".")[-1]
            # if (ext == "png"):
            #     generate_mask(inputName + "/" + file, count, False)
            #     count += 1
            try: 
                if (ext == "jpg"):
                    generate_mask(inputName + "/" + file, count, True)
                    count += 1
            except BaseException:
                print('File broke ', file)
                # os.remove(inputName + "/" + file)
