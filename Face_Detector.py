import cv2
import random
from random import randrange

# loading pre-trained data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#choose an image
#choose the webcam to obtain the image
#The (0) argument indicates that we want to use the default camera of the device to obtain the frame
webcam = cv2.VideoCapture(0)

#iterate over the video frames
while True:

    #read current frame
    successfull_frame_read, frame = webcam.read()

    #convert to grayscale
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    

    #detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

    #get the coordinates and trace a rectangle arounf it
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(100, 256), randrange(100, 256), randrange(100, 256)), 5)

    cv2.imshow('Clever Face Detector', frame)
    
    key = cv2.waitKey(1)

    if key == 81 or key==113:
        break


print('Code completed! ')


