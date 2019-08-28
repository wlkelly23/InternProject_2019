######## Webcam Object Detection Using Tensorflow-trained Classifier #########

# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a webcam feed.
# It draws boxes and scores around the objects of interest in each frame from
# the webcam.

import os
import cv2 
import numpy as np
import tensorflow as tf
import sys
import time

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

cascPath = './haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)



# Name of the directory containing the object detection module we're using
MODEL_NAME = '存放匯出model的資料夾名稱'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training資料夾名稱','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 2

## Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize webcam feed
video = cv2.VideoCapture(0)
#ret = video.set(3,1280)
#ret = video.set(4,720)
ret = video.set(3,640)
ret = video.set(4,480)

curr_time = time.time()
while(True):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (img_tensor, boxes, scores, classes, num) = sess.run(
        [image_tensor, detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    # print(img_tensor.shape,'\n',boxes,'\n',scores)


    # # Draw the results of the detection (aka 'visulaize the results')

    
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.99)


    
    hats_list = []
    for idx in range(len(boxes[0])):
        box = boxes[0][idx]
        # print(box)
        box[0] = box[0] * img_tensor.shape[1]
        box[1] = box[1] * img_tensor.shape[2]
        box[2] = box[2] * img_tensor.shape[1]
        box[3] = box[3] * img_tensor.shape[2]
    
        hat_x_avg = (box[1] + box[3])/2
        hat_y_avg = (box[0] + box[2])/2

        if (scores[0][idx] >= 0.99):
            hats_list.append([hat_x_avg, hat_y_avg])
    print('hat list = ', hats_list)



    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # print(faces)

    faces_list = []
    for (x, y, w, h) in faces:
        face_x_min = x
        face_x_max = x + w
        face_y_min = y
        face_y_max = y + h

        face_x_avg = (face_x_min + face_x_max)/2
        face_y_avg = (face_y_min + face_y_max)/2
        # print(face_x_avg, face_y_avg)
        new_face_y_min = y - h 
        faces_list.append([face_x_min, new_face_y_min, face_x_max, face_y_max])
    print('face list = ', faces_list)


    if faces_list != []:
        for f in faces_list:
            hat_detected = False
            for h in hats_list:
                if (h[0]>f[0] and h[0]<f[2]) and (h[1]>f[1] and h[1]<f[3]):
                    cv2.rectangle(frame, (f[0], f[1]), (f[2], f[3]), (0, 255, 0), 2)
                    cv2.putText(frame, 'people_hat', (f[2], f[3]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
                    hat_detected = True
            if not hat_detected:
                cv2.rectangle(frame, (f[0], int((f[3]+f[1])/2)), (f[2], f[3]), (0, 255, 255), 2)
                cv2.putText(frame, 'face_only', (f[2], f[3]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 255), 2)



        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # print(x, y, x+w, y+h)



    


    time_diff = time.time() - curr_time
    print("time_diff",time_diff)
    cv2.putText(frame,
            "FPS: %f" % (1.0 / (time_diff)),
            (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 2)
    curr_time = time.time()

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()

