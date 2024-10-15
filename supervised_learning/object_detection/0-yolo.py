#!/usr/bin/env python3
""" This module defines the Yolo class for object detection """
import keras as K


class Yolo:
    """ Yolo class for object detection."""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializer for the Yolo class.
        Args:
            model_path (str): The path to where a Darknet Keras model is
                stored.
            classes_path (str): The path to where the list of class names
                used for the Darknet model, listed in order of index, can be
                found.
            class_t (float): The box score threshold for the initial filtering
                step.
            nms_t (float): The IOU threshold for non-max suppression.
            anchors (numpy.ndarray): Is a numpy.ndarray of shape (outputs,
                anchor_boxes, 2) containing all of the anchor boxes.
                outputs: Is the number of outputs (predictions) made by the
                    Darknet model.
                anchor_boxes: Is the number of anchor boxes used for each
                    prediction.
                2: [anchor_box_width, anchor_box_height].
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as classes:
            self.class_names = [line.strip() for line in classes]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
