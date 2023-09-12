import cv2
import time
import numpy as np

import tensorflow as tf
from keras.applications import imagenet_utils
from keras.applications.mobilenet_v3 import decode_predictions
from keras.applications import imagenet_utils
from tensorflow.keras.applications import MobileNetV3Small

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture('C:/Users/Admin/PycharmProjects/pythonProject/Video/WIN_20220517_13_57_15_Pro.mp4')
    model = tf.keras.models.load_model('edge_model5.h5')
    while True:
        success, img = cap.read()
        frame = img
        final_image = cv2.resize(frame, (590, 445))
        final_image = np.expand_dims(final_image, axis=0)
        predictions = model.predict(final_image)

        # top_five = tf.keras.applications.imagenet_utils.decode_predictions(predictions, top=5)
        # print(top_five)

        classes = np.argmax(predictions, axis=1)
        print(classes)


        # print(predictions[0][0],'  ',predictions[0][1],'  ',predictions[0][2])
        # print(predictions[0][2])

        if classes == 0:
            status = "both"
            cv2.putText(img, status, (520, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6, cv2.LINE_4)
        if classes == 1:
            status = "empty"
            cv2.putText(img, status, (520, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0,), 6, cv2.LINE_4)
        if classes == 2:
            status = "human"
            cv2.putText(img, status, (520, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6, cv2.LINE_4)
        if classes == 3:
            status = "vehicle"
            cv2.putText(img, status, (520, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6, cv2.LINE_4)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()