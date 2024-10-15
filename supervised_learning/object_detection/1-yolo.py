#!/usr/bin/env python3
""" This module defines the Yolo class for object detection """
import keras as K
import numpy as np


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
        Method:
            process_outputs(self, outputs, image_size): Process Darknet
                outputs.
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as classes:
            self.class_names = [line.strip() for line in classes]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Process Darknet outputs.
        Args:
            outputs (list(numpy.ndarray)): List of predictions from the Darknet
                model for a single image.
                Each output will have the shape (grid_height, grid_width,
                anchor_boxes, 4 + 1 + classes).
                    grid_height & grid_width: The height and width of the grid
                        used for the output.
                    anchor_boxes: The number of anchor boxes used.
                    4: (t_x, t_y, t_w, t_h).
                    1: box_confidence.
                    classes: class probabilities for all classes.
            image_size (numpy.ndarray): numpy.ndarray containing the image's
                original size [image_height, image_width].
        Returns:
            boxes, box_confidences, box_class_probs.
            boxes (list(numpy.ndarray)): List of arrays containing bounding
                boxes at each output level.
            box_confidences (list(numpy.ndarray)): List of arrays containing
                box confidences at each output level.
            box_class_probs (list(numpy.ndarray)): List of arrays containing
                box's class probabilities at each output level.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []
        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Extracting secrets from boxes and class probabilities
            # Sigmoid for box confidence
            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            # Sigmoid for box class probabilities
            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))

            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

            # Calculate bounding boxes
            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]

            # Widths and heights of anchors
            anchor_w = self.anchors[i][:, 0]
            anchor_h = self.anchors[i][:, 1]

            # Sigmoid for tx and ty
            cy = (np.arange(grid_height).reshape(
                grid_height, 1, 1) + ty) / grid_height
            cx = (np.arange(grid_width).reshape(
                1, grid_width, 1) + tx) / grid_width

            # Calculate the center of the box (relative to the grid)
            bw = np.exp(tw) * anchor_w / image_width
            bh = np.exp(th) * anchor_h / image_height

            # Convert to coordinates of (x1, y1) and (x2, y2)
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2

            # Concatenate the coordinates into a single array
            box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(box)

        return boxes, box_confidences, box_class_probs
