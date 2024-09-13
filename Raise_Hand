import cv2
import numpy as np
import tensorflow as tf

# Load YOLOv4 model and weights
yolov4 = tf.keras.models.load_model(r"D:\Yolo-v4\tensorflow-yolov4-master")  # Assuming you have the pre-trained YOLOv4 model file

classes = ["student_raise_hand"] 


def detect_student_raise_hand(image_path):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (608, 608))  

    image = image / 255.0  # Normalize pixel values


    detections = yolov4.predict(np.expand_dims(image, axis=0))


    for detection in detections:
        for detected_object in detection:
            class_index = int(detected_object[4])
            confidence = detected_object[5]
            if confidence > 0.5 and class_index == 0:  # Set confidence threshold and class index
                class_name = classes[class_index]
                print("Detected:", class_name, "with confidence:", confidence)

image_path = r"#ENTER THE FILE PATH"
detect_student_raise_hand(image_path)









