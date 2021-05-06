

import cv2
import dlib
import numpy as np
import os
from PIL import Image
import sys

inputName = sys.argv[1]
for file in (os.listdir(inputName)):
    # print(file)
    if "_m.png" not in file and ("png" in file):
        ext = file.split(".")[0]
        if ext + "_m.png" not in os.listdir(inputName):
            os.remove(inputName + "/" + file)

        
    



 