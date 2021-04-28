import sys
import os
import random
from sklearn.model_selection import train_test_split
import pandas as pd

partition_amt = int(13141* .8)

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
    trainSize = 13141

    train = random.sample(noMask, trainSize)
    trainMask = []

    for file in train:
        corrFile = file[:-4] + "_m.png"
        trainMask.append(corrFile)

    print(train) #train mask
    print(trainMask) # corresponding with mask


    first10_mask = train[:partition_amt]
    first10_nomask = trainMask[:partition_amt]
    ones = [1 for i in range(partition_amt)]
    df = (pd.DataFrame([first10_mask, first10_nomask, ones])).transpose()
    df.to_csv('out.csv',index = False)

    other_mask = train[partition_amt:]
    print(len(other_mask))
    other_nomask = trainMask[partition_amt:]
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
    df_2 = (pd.DataFrame([different_mask, different_no_mask, zeros])).transpose()
    # df.to_csv('out2.csv',index = False)
    df_out = pd.concat([df, df_2])
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(df[[0, 1]], df[[2]], random_state = 10, test_size = 0.2)
    X_train, X_test, y_train, y_test = train_test_split(df_2[[0, 1]], df_2[[2]], random_state = 10, test_size = 0.2)

    X_train_p['info'] = 1
    X_train['info'] = 0
    trains = pd.concat([X_train_p, X_train])

    X_test_p['info'] = 1
    X_test['info'] = 0
    tests = pd.concat([X_test_p, X_test])

    trains.to_csv('trains.csv', index = False)
    tests.to_csv('tests.csv', index = False)


    # X_train_p.to_csv('x_train_p.csv')
    # y_train_p.to_csv('y_train_p.csv')




    # df_out.to_csv('concatenated_dfs.csv')
