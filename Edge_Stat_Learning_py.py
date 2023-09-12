import cv2
# import time
# import os
# import numpy as np
# import random
# import pickle
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

def main():
    while True:

        train_path = "./Images/Train_edge"
        valid_path = "./Images/Validation_edge"
        classes_train_valid = ["both", "empty", "human", "vehicle"]

        train_generator = ImageDataGenerator().flow_from_directory(train_path, classes=classes_train_valid, target_size=(590,445), batch_size=30)
        valid_generator = ImageDataGenerator().flow_from_directory(valid_path, classes=classes_train_valid, target_size=(590,445), batch_size=20)


        base_model = MobileNetV3Large(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        preds = Dense(4, activation='softmax')(x)  # liczba wyjsc
        model = Model(inputs=base_model.input, outputs=preds)

        epochs = 1
        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.fit(train_generator, validation_data=valid_generator, epochs=epochs)
        model.save('edge_model5.h5')
        cv2.waitKey(1)

if __name__ == "__main__":
    main()