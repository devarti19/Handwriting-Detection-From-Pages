import cv2
import numpy as np
import imutils
from keras.models import load_model

model = load_model(r'C:\Users\91758\Downloads\ML CW2\vehicle\model-030.model')
cap = cv2.VideoCapture('video.mp4')
currentframe = 0
while(True):

    ret, frames = cap.read()
    frames = imutils.resize(frames, width=400, height=200)
    print(np.array(frames).shape)

    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(frames, (256, 256))

    normalized = gray / 255.0
    reshaped = np.reshape(normalized, (1, 256, 256, 3))
    y_pred = model.predict(reshaped)
    y_pred = y_pred*255.0

    for (x, y, w, h) in y_pred:
        plate = cv2.rectangle(frames, (int(x), int(y)), (int(x) + int(w), int(y) + int(h)), (0, 255, 0), 2)

        ro = frames[int(y):int(y) + int(h), int(x):int(x) + int(w)]

    cv2.imshow('frames', frames)

    currentframe += 1

    if cv2.waitKey(33) == 27:
        break

# De-allocate any associated memory usage
cv2.destroyAllWindows()