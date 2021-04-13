import sys
import os
import random
import pandas as pd

if __name__ == '__main__':

    try:
        inputName = sys.argv[1]

    except IndexError:
        print('Error: Provide an input folder')

    allFiles = os.listdir(inputName)
    subset = allFiles[:100]

    noMask = []
    for file in allFiles:
        if ("png" in file and "_m.png" not in file):
            noMask.append(file)

    print(noMask)
    print("len no mask",len(noMask))
    trainSize = 13000

    train = random.sample(noMask, trainSize)
    trainMask = []

    for file in train:
        corrFile = file[:-4] + "_m.png"
        trainMask.append(corrFile)

    print(train) #train mask
    print(trainMask) # corresponding with mask


    first10_mask = train[:6500]
    first10_nomask = trainMask[:6500]
    ones = [1 for i in range(6500)]
    df = (pd.DataFrame([first10_mask, first10_nomask, ones])).transpose()
    df.to_csv('out.csv',index = False)

    other_mask = train[6500:]
    print(len(other_mask))
    other_nomask = trainMask[6500:]
    print(len(other_nomask))

    different_mask = []
    different_no_mask = []


    while (len(other_mask) != 0):
        first_image = other_mask[0]
        other_mask = other_mask[1:]

        # Pick a random second image
        second_image = other_nomask[random.randint(0, len(other_nomask)-1)]

        second_image_prefix = second_image[:second_image.rindex("_m.png")]
        second_image_prefix = second_image[:second_image_prefix.rindex("_")]

        first_image_prefix = second_image[:second_image.rindex("_")]

        # first_middle_last_#_m.png
        while (second_image_prefix == first_image_prefix):
            second_image = other_nomask[random.randint(0, len(other_nomask)-1)]
            second_image_prefix = second_image[:second_image.rindex("_m.png")]
            second_image_prefix = second_image[:second_image.rindex("_")]
    
        print("second image prefix", second_image_prefix)
        print("first image prefix", first_image_prefix)

        different_mask.append(first_image)
        different_no_mask.append(second_image)
    
    zeros = [0 for i in range(len(different_mask))]
    df = (pd.DataFrame([different_mask, different_no_mask, zeros])).transpose()
    df.to_csv('out2.csv',index = False)