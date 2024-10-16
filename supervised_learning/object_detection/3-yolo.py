#!/usr/bin/env python3
""" This module defines the Yolo class for object detection """
import keras as K
import numpy as np
import tensorflow as tf


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

    def sigmoid(self, x):
        """A simple sigmoid method"""
        return 1 / (1 + np.exp(-x))

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
            grid_height, grid_width, _, _ = output.shape

            # Extracting secrets from boxes and class probabilities
            # Sigmoid for box confidence
            box_confidence = self.sigmoid(output[..., 4:5])
            # Sigmoid for box class probabilities
            box_class_prob = self.sigmoid(output[..., 5:])

            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

            # Calculate bounding boxes
            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]

            # Widths and heights of anchors
            anchor_w = self.anchors[i, :, 0]
            anchor_h = self.anchors[i, :, 1]
            # grid cells coordinates for width and height
            cx, cy = np.meshgrid(np.arange(grid_width),
                                 np.arange(grid_height))

            # Add axis to match dimensions of t_x & t_y
            cx = np.expand_dims(cx, axis=-1)
            cy = np.expand_dims(cy, axis=-1)

            # Calculate bounding box coordinates
            bx = (self.sigmoid(tx) + cx) / grid_width
            by = (self.sigmoid(ty) + cy) / grid_height
            # Calculate the center of the box (relative to the grid)
            bw = (np.exp(tw) * anchor_w) / self.model.input.shape[1]
            bh = (np.exp(th) * anchor_h) / self.model.input.shape[2]

            # Convert to coordinates of (x1, y1) and (x2, y2)
            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            # Concatenate the coordinates into a single array
            box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(box)

        return boxes, box_confidences, box_class_probs

    def nms(self, boxes, scores, iou_threshold):
        """
        Performs non-max suppression using TensorFlow's built-in function.
        Args:
            boxes: numpy.ndarray of shape (m, 4) containing the bounding boxes.
            scores: numpy.ndarray of shape (m,) containing the scores of the
            boxes.
            iou_threshold: The IoU threshold used to discard overlapping boxes.
        Returns:
            keep: numpy.ndarray of shape (n,) containing the indices of the
            boxes to keep.
        """
        # Convert boxes and scores to tensors
        boxes_tensor = tf.convert_to_tensor(boxes, dtype=tf.float32)
        scores_tensor = tf.convert_to_tensor(scores, dtype=tf.float32)

        # Perform Non-Maximum Suppression using TensorFlow
        selected_indices = tf.image.non_max_suppression(
            boxes_tensor, scores_tensor, max_output_size=boxes.shape[0],
            iou_threshold=iou_threshold)

        # Convert the selected indices back to numpy array and return
        return selected_indices.numpy()

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter boxes.
        Args:
            boxes (list(numpy.ndarray)): List of arrays containing bounding
                boxes at each output level.
            box_confidences (list(numpy.ndarray)): List of arrays containing
                box confidences at each output level.
            box_class_probs (list of numpy.ndarray): List of arrays containing
                box class probabilities at each output level.
        Returns: Tuple of (filtered_boxes, box_classes, box_scores).
            filtered_boxes (numpy.ndarray): Shape (m, 4) containing all of the
                filtered bounding boxes.
            box_classes (numpy.ndarray): Shape (m,) containing the class number
                that each box in filtered_boxes predicts.
            box_scores (numpy.ndarray): Shape (m,) containing the box scores
                for each box in filtered_boxes.
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            # Get box confidences and class probabilities
            box_confidence = box_confidences[i]
            box_class_prob = box_class_probs[i]

            # Compute box scores: confidence * class probability
            box_scores_level = box_confidence * box_class_prob

            # Find the class index with the maximum box score
            box_class_argmax = np.argmax(box_scores_level, axis=-1)
            box_class_score = np.max(box_scores_level, axis=-1)

            # Apply the mask based on the class threshold
            mask = box_class_score >= self.class_t

            # Filter boxes, classes, and scores
            filtered_boxes.append(boxes[i][mask])
            box_classes.append(box_class_argmax[mask])
            box_scores.append(box_class_score[mask])

        # Concatenate all filtered results across outputs
        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """ This function performs non-max suppression on the boundary boxes.
        Args:
            filtered_boxes (numpy.ndarray): Shape (m, 4) containing all of the
                filtered bounding boxes.
            box_classes (numpy.ndarray): Shape (m,) containing the class number
                that each box in filtered_boxes predicts.
            box_scores (numpy.ndarray): Shape (m,) containing the box scores
                for each box in filtered_boxes.
        Returns:
            Tuple of (box_predictions, predicted_box_classes,
            predicted_box_scores).
            box_predictions (numpy.ndarray): Shape (m, 4) containing all of the
                predicted bounding boxes ordered by class and box score.
            predicted_box_classes (numpy.ndarray): Shape (m,) containing the
                class number for box_predictions.
            predicted_box_scores (numpy.ndarray): Shape (m) containing the box
                scores for box_predictions.
        """
        # Get the unique classes
        unique_classes = np.unique(box_classes)

        # Initialize the list to hold the results
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for cls in unique_classes:
            # Create a mask for each class and filter boxes
            mask = np.where(box_classes == cls)
            boxes_of_class = filtered_boxes[mask]
            class_scores = box_scores[mask]
            class_classes = box_classes[mask]

            # Perform NMS on the class boxes and scores
            keep = self.nms(boxes_of_class, class_scores, self.nms_t)

            # Append the results after NMS to the results list
            box_predictions.append(boxes_of_class[keep])
            predicted_box_classes.append(class_classes[keep])
            predicted_box_scores.append(class_scores[keep])

        # Concatenate the results into single arrays
        box_predictions = np.concatenate(box_predictions, axis=0)
        predicted_box_classes = np.concatenate(predicted_box_classes, axis=0)
        predicted_box_scores = np.concatenate(predicted_box_scores, axis=0)

        return box_predictions, predicted_box_classes, predicted_box_scores
