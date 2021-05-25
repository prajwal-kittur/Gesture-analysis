import numpy as np
from keras.models import model_from_json
import operator
import cv2
import sys, os

import pymysql

conn = pymysql.connect(host="localhost", user="root", passwd="", db="test3")
mycursor = conn.cursor()

# Loading the model
json_file = open("model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("model-bw.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)

# Category dictionary
categories = {0: 'CLOSED_FIST', 1: 'THUMBS_UP', 2: 'THUMBS_DOWN'}

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Got this from collect-data.py
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    
    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (64, 64))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("test", test_image)
    # Batch of 1
    result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))
    prediction = {"STATUS : "'CLOSED_FIST': result[0][0],
                  "STATUS : "'THUMBS_UP': result[0][1],
                  "STATUS : "'THUMBS_DOWN': result[0][2]
                  }

    # Sorting based on top prediction
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    status = prediction[0][0]
    # Displaying the predictions

    cv2.putText(frame, "Stock Managment ", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "PRODUCT ", (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, " 1: Perfume ", (10, 195), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, " 2: Soap ", (10, 210), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, " 3: Shampoo ", (10, 225), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)

    cv2.putText(frame, " 4: Face Wash ", (10, 240), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.imshow("Frame", frame)

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:  # esc key
        break
    if interrupt & 0xFF == ord('1'):
        mycursor.execute("INSERT INTO stock(id,state) VALUES (1, %s);", status)
        conn.commit()
    if interrupt & 0xFF == ord('2'):
        mycursor.execute("INSERT INTO stock(id,state) VALUES (2, %s);", status)
        conn.commit()
    if interrupt & 0xFF == ord('3'):
        mycursor.execute("INSERT INTO stock(id,state) VALUES (3, %s);", status)
        conn.commit()
    if interrupt & 0xFF == ord('4'):
        mycursor.execute("INSERT INTO stock(id,state) VALUES (4, %s);", status)
        conn.commit()

conn.close()
cap.release()
cv2.destroyAllWindows()
