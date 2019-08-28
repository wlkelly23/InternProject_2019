import cv2
import sys
import time
import math
import numpy as np
from collections import Counter
from selenium import webdriver
from selenium.webdriver.common.keys import Keys


browser = webdriver.Chrome()
browser.get('https://poki.com/en/g/slime-road')
time.sleep(15)

while True:
    try:
        browser.find_element_by_name('gameFrame').click()
        print('got gameframe and click')
        break
    except Exception as ex:
        print(str(ex))
        pass
while True:
    try:
        browser.switch_to.frame(browser.find_element_by_name("gameFrame"))
        break
    except Exception as ex:
        print(str(ex))
        pass


cascPath = './haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
# ret = video_capture.set(3,640)
# ret = video_capture.set(4,480)

curr_time = time.time()
face_center_points = []
center_point_distance = []
center_point_rate = []
face_location = []
thr_sec = 0.5
last_thr_time = time.time()

while True:

    ret, frame = video_capture.read()
    # frame.shape:(480,640,3)

    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

 
    face_detection = False
  
    for (x, y, w, h) in faces:
        # detect the main face
        if w > frame.shape[1]/4 and h > frame.shape[0]/4:
            face_x_min = x/frame.shape[1]
            face_x_max = (x + w)/frame.shape[1]
            face_y_min = y/frame.shape[0]
            face_y_max = (y + h)/frame.shape[0]

            face_x_avg = (face_x_min + face_x_max)/2
            face_y_avg = (face_y_min + face_y_max)/2


            # 如果要執行遊戲使用下面這行:
            face_center_points.append((x+(w/2), y+(h/2)))

            # 如果要推算速度使用下面這行:
            # face_center_points.append((face_x_avg, face_y_avg))

            face_detection = True
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # print(face_center_points)
            # identify whether the face is moving
            # if time.time() - last_thr_time > thr_sec:

            if (len(face_center_points) >= 2):
                if (face_center_points[-1][0] - face_center_points[-2][0]) > 5 :
                    browser.find_element_by_id("gameframe").send_keys(Keys.ARROW_LEFT)
                    face_location.append("左")
                elif (face_center_points[-1][0] - face_center_points[-2][0]) < -5:
                    browser.find_element_by_id("gameframe").send_keys(Keys.ARROW_RIGHT)
                    face_location.append("右")
                else:
                    face_location.append("不動")
                # last_thr_time = time.time()

        

            # distance between face's center points
            dis = 0
            if (len(face_center_points) >= 2):
                dis = math.sqrt((face_center_points[-1][0]-face_center_points[-2][0])**2 + (face_center_points[-1][1]-face_center_points[-2][1])**2)
                center_point_distance.append(dis)

            break
        


    # delete the first element when len(face_center_points)>30:
    if (len(face_center_points) > 30):
        face_center_points.pop(0)

    # delete the first element when len(center_point_distance)>30:
    if (len(center_point_distance) > 30):
        center_point_distance.pop(0)
           

    if face_detection:
        time_diff = time.time() - curr_time
        # print("time_diff: ",time_diff)

    # print(face_center_points)
    # print(center_point_distance)
    # print(len(center_point_distance))


    speed_factor = 25

    if(len(center_point_distance) != 0 and face_detection == True):
        center_point_rate.append(center_point_distance[-1]/time_diff)
        speed_rate = [x * speed_factor for x in center_point_rate]
        # print(center_point_rate)
        # print("time_diff: ",time_diff, center_point_rate[-1], speed_rate[-1])
        if len(speed_rate) > 1:
            if speed_rate[-1] > 1:
                print(int(speed_rate[-1]), "km/hr")
        # print(int(speed_rate))


    # delete the first element when len(center_point_rate)>30:    
    if(len(center_point_rate) > 30):
        center_point_rate.pop(0)

    # print(center_point_rate)
    # print(len(center_point_rate))




    curr_time = time.time()

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()