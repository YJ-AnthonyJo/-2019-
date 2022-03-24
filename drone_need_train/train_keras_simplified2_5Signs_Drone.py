#!/usr/bin/env python3
import os
import glob

import cv2
import numpy as np
from keras.models import Model
from keras.layers import Dense, Activation, MaxPool2D, Conv2D, Flatten, Dropout, Input, BatchNormalization, Add
from keras.optimizers import Adam

# Worker function for custom model
def conv_block(x, filters):
    x = BatchNormalization() (x)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same') (x)

    x = BatchNormalization() (x)
    shortcut = x
    x = Conv2D(filters, (3, 3), activation='relu', padding='same') (x)
    x = Add() ([x, shortcut])
    x = MaxPool2D((2, 2), strides=(2, 2)) (x)

    return x

# DIY model for training (instead of using standard model package)
def custom_model(input_shape, n_classes):

    input_tensor = Input(shape=input_shape)

    x = conv_block(input_tensor, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 256)
    x = conv_block(x, 512)

    x = Flatten() (x)
    x = BatchNormalization() (x)
    x = Dense(512, activation='relu') (x)
    x = Dense(512, activation='relu') (x)

    output_layer = Dense(n_classes, activation='softmax') (x)

    inputs = [input_tensor]
    model = Model(inputs, output_layer)

    return model


# main loop
def main():

    # Data parameter
    input_height = 48
    input_width = 48

    input_channel = 3
    input_shape = (input_height, input_width, input_channel)
    n_classes = 6 # 4 objects  #######################

    # Modeling
    # 'custom':
    model = custom_model(input_shape, n_classes)

    adam = Adam()
    model.compile(
        optimizer=adam,
        loss='categorical_crossentropy',
        metrics=['acc'],
    )

    # Search all images
    data_dir = 'DataforAI'
    match_obj1 = os.path.join('DataforAI', 'left', '*.jpg')
    paths_obj1 = glob.glob(match_obj1)

    match_obj2 = os.path.join('DataforAI', 'right', '*.jpg')
    paths_obj2 = glob.glob(match_obj2)

    match_obj3 = os.path.join('DataforAI', 'land', '*.jpg')
    paths_obj3 = glob.glob(match_obj3)

    match_obj4 = os.path.join('DataforAI', 'forward', '*.jpg')
    paths_obj4 = glob.glob(match_obj4)

    match_obj5 = os.path.join('DataforAI', 'backward', '*.jpg')
    paths_obj5 = glob.glob(match_obj5)

    match_obj6 = os.path.join('DataforAI', 'other', '*.jpg')
    paths_obj6 = glob.glob(match_obj6)

    #print(paths_obj5)

    match_test = os.path.join('DataforAI', 'test', '*.jpg')
    paths_test = glob.glob(match_test)

    n_train = len(paths_obj1) + len(paths_obj2) + len(paths_obj3) + len(paths_obj4) + len(paths_obj5) +len(paths_obj6)
    n_test = len(paths_test)

    # Initialization dataset matrix
    trainset = np.zeros(
        shape=(n_train, input_height, input_width, input_channel),
        dtype='float32',
    )
    label = np.zeros(
        shape=(n_train, n_classes),
        dtype='float32',
    )
    testset = np.zeros(
        shape=(n_test, input_height, input_width, input_channel),
        dtype='float32',
    )

    # Read image and resize to data set
    paths_train = paths_obj1 + paths_obj2 + paths_obj3 + paths_obj4 + paths_obj5 + paths_obj6

    for ind, path in enumerate(paths_train):
        try:
        #if True:
            #print("000")
            image = cv2.imread(path)
            #print("-1")
            ###################################
            """
            height, width = image.shape[:2]
            #print("0")
            
            img_color = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            #print("1")

            # 원본 영상을 HSV 영상으로 변환합니다.
            img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

            # 범위 값으로 HSV 이미지에서 마스크를 생성합니다.
            lower_blue1 = np.array([105,  81,  81])
            upper_blue1 = np.array([115, 255, 255])

            lower_blue2 = np.array([95, 81, 81])
            upper_blue2 = np.array([105, 255, 255])
            
            lower_blue3 = np.array([95, 81, 81])
            upper_blue3 = np.array([105, 255, 255])
            ######
            img_mask1 = cv2.inRange(img_hsv, lower_blue1, upper_blue1)
            img_mask2 = cv2.inRange(img_hsv, lower_blue2, upper_blue2)
            img_mask3 = cv2.inRange(img_hsv, lower_blue3, upper_blue3)
            img_mask = img_mask1 | img_mask2 | img_mask3

            kernel = np.ones((5,5),np.uint8)
            img_mask= cv2.morphologyEx(img_mask,cv2.MORPH_OPEN,kernel)
            img_mask= cv2.morphologyEx(img_mask,cv2.MORPH_CLOSE,kernel)
            #print('2')

            # 마스크 이미지로 원본 이미지에서 범위값에 해당되는 영상 부분을 획득합니다.
            img_result = cv2.bitwise_and(img_color, img_color, mask=img_mask)
            numOfLabels, img_label,stats,centroids =cv2.connectedComponentsWithStats(img_mask)

            for idx, centroid in enumerate(centroids):
                if stats[idx][0] == 0 and stats[idx][1] ==0:
                    continue
                if np.any(np.isnan(centroid)):
                    continue
                x,y,width,height,area=stats[idx]
                centerX, cneterY= int(centroid[0]), int(centroid[1])

                if area>20000:
                    #cv.rectangle(img_color,(x,y),(x+width,y+height),(0,0,255))
                    image=image[y:y+height ,x:x+width]
                    ##############################3
            
            #print('3')
            """
            resized_image = cv2.resize(image, (input_width, input_height))
            # 컴퓨터는 항상 같은 데이터 크기가 들어와야 하기에 아니면 컴퓨터 입장에서는 알기 어렵다.
            trainset[ind] = resized_image

        except Exception as e:
            print(path) # print out the Image that cause exception error

    for ind, path in enumerate(paths_test):
        try:
            image = cv2.imread(path)
            resized_image = cv2.resize(image, (input_width, input_height))
            testset[ind] = resized_image

        except Exception as e:
            print(path) # print out the Image that cause exception error

    # Set the mark of the training set
    n_obj1 = len(paths_obj1)
    n_obj2 = len(paths_obj2)
    n_obj3 = len(paths_obj3)
    n_obj4 = len(paths_obj4)
    n_obj5 = len(paths_obj5)
    n_obj6 = len(paths_obj6)
    #print(n_obj5)

    begin_ind = 0
    end_ind = n_obj1
    label[begin_ind:end_ind, 0] = 1.0

    begin_ind = n_obj1
    end_ind = n_obj1 + n_obj2
    label[begin_ind:end_ind, 1] = 1.0

    begin_ind = n_obj1 + n_obj2
    end_ind = n_obj1 + n_obj2 + n_obj3
    label[begin_ind:end_ind, 2] = 1.0

    begin_ind = n_obj1 + n_obj2 + n_obj3
    end_ind = n_obj1 + n_obj2 + n_obj3 + n_obj4
    label[begin_ind:end_ind, 3] = 1.0

    begin_ind = n_obj1 + n_obj2 + n_obj3 + n_obj4
    end_ind = n_obj1 + n_obj2 + n_obj3 + n_obj4 + n_obj5
    #print(begin_ind, end_ind)
    label[begin_ind:end_ind, 4] = 1.0

    begin_ind = n_obj1 + n_obj2 + n_obj3 + n_obj4 + n_obj5
    end_ind = n_obj1 + n_obj2 + n_obj3 + n_obj4 + n_obj5 + n_obj6
    #print(begin_ind, end_ind)
    label[begin_ind:end_ind, 5] = 1.0

    # Normalize the value between 0 and 1
    trainset = trainset / 255.0
    testset = testset / 255.0

    # Training model
    model.fit(
        trainset,
        label,
        epochs=9,  # no. of rounds of training => 8 rounds
        validation_split=0.2,   # percentage of dataset use for validation at trainiing => 20% (2000 images, 1600 for training, 400 for validation)
    )

    # Saving model architecture and weights (parameters)
    model_desc = model.to_json()
    model_file = 'DataforAI/model.json'
    with open(model_file, 'w') as file_model:
        file_model.write(model_desc)

    weights_file = 'DataforAI/weights.h5'
    model.save_weights(weights_file )

    # Execution predication
    if testset.shape[0] != 0:
        result_onehot = model.predict(testset)
        result_sparse = np.argmax(result_onehot, axis=1)
    else:
        result_sparse = list()

    # Print predication results
    print('File name\t forecast category')

    for path, label_id in zip(paths_test, result_sparse):
        filename = os.path.basename(path)
        if label_id == 0:
            label_name = 'left'
        elif label_id == 1:
            label_name = 'right'
        elif label_id == 2:
            label_name = 'land'
        elif label_id == 3:
            label_name = 'forward'
        elif label_id == 4:
            label_name = 'backward'
        elif label_id ==5:
            label_name ='other'
        

        print('%s\t%s' % (filename, label_name))

if __name__ == '__main__':
    main()
