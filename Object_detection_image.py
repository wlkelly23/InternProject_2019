######## Image Object Detection Using Tensorflow-trained Classifier #########
 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on an image.
# It draws boxes and scores around the objects of interest in the image.

import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import csv
import math

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
# IMAGE_NAME = 'test1.jpg'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to image
# PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 34

# Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
# print(category_index)



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

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value



#create a dictionary to put every photo's ziva_value, x_avg, and y_avg.
csv_path = ['C:\\models\\research\\object_detection\\images\\train_ziva_1_new_tag_labels.csv',
            'C:\\models\\research\\object_detection\\images\\train_ziva_2_new_tag_labels.csv']
csv_dict = {}
ziva_value_count = {}
row_count = 0
for c_path in csv_path:
    with open( c_path , newline='') as csvfile:
        rows = csv.reader(csvfile)
        headers = next(rows)
        for row in rows:
            row_count += 1
            if (c_path == csv_path[0]):
                row[0] = 'train_ziva_1_'+ row[0]
            else:
                row[0] = 'train_ziva_2_'+ row[0] 

            file_name = row[0]
            ziva_value = row[3]
            x_avg = (int(row[4]) + int(row[6]))/2
            y_avg = (int(row[5]) + int(row[7]))/2

            if file_name not in csv_dict:
                csv_dict.update({file_name:[[ziva_value, x_avg, y_avg]]})
            else:
                csv_dict[file_name].append([ziva_value, x_avg, y_avg])

            if ziva_value not in ziva_value_count:
                ziva_value_count[ziva_value] = 1
            else:
                ziva_value_count[ziva_value] += 1
# print(row_count)
# print(csv_dict)

# for k,v in sorted(ziva_value_count.items(), key=lambda d: d[1]):
#     print(k ,'\t', v)
# sys.exit(0)


img_path = ['C:\\models\\research\\object_detection\\images\\train_ziva_1_test', 
            'C:\\models\\research\\object_detection\\images\\train_ziva_2_test']
# img_path = ['C:\\models\\research\\object_detection\\images\\train_ziva_2_test']

correct_img = []
wrong_img_with_equal_boxes = []
wrong_img_with_zero_box = []
wrong_img_with_no_equal_boxes = []
wrong_img = []
Total_pred_ans_boxes = 0
correct_boxes = 0
wrong_boxes = 0
    
for i_path in img_path:
    files = os.listdir(i_path)
    for image_name in files:
        # image_name = "  "
        if image_name.endswith(".jpg"):
            # print(image_name)
            img = cv2.imread(os.path.join(i_path, image_name))
            # print(os.path.join(i_path, image_name))

            #resize the photo
            scale_percent = 100
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            res = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

            if (i_path == img_path[0]):
                csv_dict_value = csv_dict['train_ziva_1_'+ image_name]
            else:
                csv_dict_value = csv_dict['train_ziva_2_'+ image_name]

            raw_img = res.copy()

            true_ans = []
            for x in csv_dict_value:
                cv2.putText(raw_img, x[0], (int(x[1]* scale_percent / 100), int(x[2]* scale_percent / 100)), cv2.FONT_HERSHEY_TRIPLEX, 4, (0, 255, 255), 8)
                true_ans.append([x[0], int(x[1]* scale_percent / 100), int(x[2])* scale_percent / 100])
            # print(true_ans)


            
            img_expanded = np.expand_dims(res, axis=0)
            # # Perform the actual detection by running the model with the image as input
            pred_classes = []
            for i in range(1):
                (img_tensor, boxes, scores, classes) = sess.run(
                    [image_tensor,detection_boxes, detection_scores, detection_classes],
                    feed_dict={image_tensor: img_expanded})
                pred_classes.append(classes[0][0])

            # print(scale_percent,res.shape,img_tensor.shape,boxes[0][:5], '\n' ,scores[0][:5], '\n' ,classes[0][:5])
            # print(scale_percent, '\n' ,scores[0][:5], '\n' ,classes[0][:5], '\n')


            thr = 0.9
            # Draw the results of the detection (aka 'visulaize the results')
            vis_util.visualize_boxes_and_labels_on_image_array(
                res,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=thr)

            height_max = raw_img.shape[0]
            tot_img = np.zeros((height_max,raw_img.shape[1]+res.shape[1], 3), dtype = np.uint8)
            tot_img[:res.shape[0],:res.shape[1]] = res
            tot_img[:raw_img.shape[0],res.shape[1]:res.shape[1]+raw_img.shape[1]] = raw_img

            # cv2.imwrite(os.path.join("C:\\models\\research\\object_detection\\images\\train_ziva_pred_2", image_name), tot_img)
            # print(res.shape,img_tensor.shape,boxes[0][:5], scores[0][:5], classes[0][:5])
            # sys.exit(0)





            if True:
                first_n = 10
                pred_ans = []

                for idx in range(len(boxes[0][:first_n])):
                    box = boxes[0][:first_n][idx]
                    box[0] = box[0] * img_tensor.shape[1]
                    box[1] = box[1] * img_tensor.shape[2]
                    box[2] = box[2] * img_tensor.shape[1]
                    box[3] = box[3] * img_tensor.shape[2]

                    box_x_avg = (box[1]+box[3])/2
                    box_y_avg = (box[0]+box[2])/2
                    class_name = category_index[classes[0][idx]]['name']  

                    # print(boxes[0][:1], scores[0][:1], classes[0][:1],box_x_avg,box_y_avg)
                    # cv2.circle(res,(int(box_x_avg), int(box_y_avg)), 10, (255, 0, 0), -1)

                    if scores[0][idx] >= thr:
                        pred_ans.append([class_name, box_x_avg, box_y_avg])
                    
                Total_pred_ans_boxes += len(pred_ans)


                # counting_wrong_boxes_based_on_pred_ans
                count = 0
                for x in pred_ans:
                    distance_smallest = 50
                    distance_smallest_ziva_value = None
                    for y in true_ans:
                        dis = math.sqrt((y[1]-x[1])**2 + (y[2]-x[2])**2)
                        if (dis <= distance_smallest):
                            # distance_smallest = dis
                            distance_smallest_ziva_value = y[0]
                            break

                    if (x[0] == distance_smallest_ziva_value):
                        correct_boxes += 1
                        
                    else:
                        wrong_boxes += 1
                        count += 1

                if  (count != 0):
                    cv2.imwrite(os.path.join("C:\\models\\research\\object_detection\\images\\train_ziva_check_2", image_name), tot_img)
                    wrong_img.append(image_name)
                    # print(image_name)
                    



                # counting_wrong_boxes_based_on_true_ans

                # if len(pred_ans) == 0:
                #     wrong_boxes = wrong_boxes + len(true_ans)
                # else:
                #     for y in true_ans:
                #         distance_smallest = 50
                #         distance_smallest_ziva_value = None
                #         for x in pred_ans:

                #             dis = math.sqrt((y[1]-x[1])**2 + (y[2]-x[2])**2)
                #             if (dis <= distance_smallest):
                #                 # distance_smallest = dis
                #                 distance_smallest_ziva_value = x[0]
                #                 break

                #         if (y[0] == distance_smallest_ziva_value):
                #                 correct_boxes = correct_boxes + 1
                #         else:
                #             wrong_boxes = wrong_boxes + 1        




                # folder for training_ziva classifiation 
                if len(pred_ans) == 0:
                    wrong_img_with_zero_box.append(image_name)
                    cv2.imwrite(os.path.join("C:\\models\\research\\object_detection\\images\\train_ziva_zero_box", image_name), tot_img)
                elif len(pred_ans) != len(true_ans):
                    wrong_img_with_no_equal_boxes.append(image_name)
                    cv2.imwrite(os.path.join("C:\\models\\research\\object_detection\\images\\train_ziva_no_equal_box", image_name), tot_img)
                    
                else:
                    
                    correct = 0
                    
                    for x in pred_ans:
                        distance_smallest = 100000000
                        distance_smallest_ziva_value = None

                        for y in true_ans:
                            dis = (y[1]-x[1])**2 + (y[2]-x[2])**2
                            if (dis < distance_smallest):
                                distance_smallest = dis
                                distance_smallest_ziva_value = y[0]

                        if (x[0] == distance_smallest_ziva_value):
                            correct = correct + 1

                    if (correct == len(pred_ans)):
                        # print(image_name +': correct!')
                        correct_img.append(image_name)
                        cv2.imwrite(os.path.join("C:\\models\\research\\object_detection\\images\\train_ziva_correct", image_name), tot_img)
                    else:
                        # print(image_name +': wrong!')
                        wrong_img_with_equal_boxes.append(image_name)
                        cv2.imwrite(os.path.join("C:\\models\\research\\object_detection\\images\\train_ziva_wrong", image_name), tot_img)




# for "counting_wrong_boxes_based_on_pred_ans"
print(len(wrong_img))
print('Total_pred_ans_boxes: ',Total_pred_ans_boxes)
print('correct_boxes: ',correct_boxes)
print('wrong_boxes: ',wrong_boxes)    
print(wrong_boxes/Total_pred_ans_boxes)




# for "folder for training_ziva classifiation"
print('Correct images: \n', correct_img)
print(len(correct_img),'\n')

print('Wrong images with equal boxes: \n', wrong_img_with_equal_boxes)
print(len(wrong_img_with_equal_boxes),'\n')

print('Wrong images with zero box (min score thr = 0.9): \n', wrong_img_with_zero_box)
print(len(wrong_img_with_zero_box),'\n')

print('Wrong images with no equal boxes (min score thr = 0.9): \n', wrong_img_with_no_equal_boxes)
print(len(wrong_img_with_no_equal_boxes))




            # # # All the results have been drawn on image. Now display the image.
#             cv2.imshow('Object detector', res)

#             # # # # # Press any key to close the image
#             cv2.waitKey(0)

#                 # # # # # Clean up
# cv2.destroyAllWindows()
