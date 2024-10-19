#!/usr/bin/env python3
""" This module defines the Yolo class for object detection """
import keras as K
import numpy as np
import tensorflow as tf
import os
import cv2


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

    @staticmethod
    def load_images(folder_path):
        """
        This method loads images.
        Args:
            folder_path (str): The path to the folder where the images to load
                are stored.
        Returns:
            Tuple of (images, image_paths).
            images (list): List of images as numpy.ndarrays.
            image_paths (list): List of paths to the individual images.
        """
        # Initialize lists to hold the images and their paths
        images = []
        image_paths = []

        # Loop through all files in the folder
        for filename in os.listdir(folder_path):
            # Construct the full path to the image
            file_path = os.path.join(folder_path, filename)

            # Check if it's a valid image file (cv2 can read it)
            image = cv2.imread(file_path)

            if image is not None:  # Check if the image was successfully loaded
                images.append(image)  # Append the loaded image
                image_paths.append(file_path)  # Append the path of the image

        # Return the tuple of images and their corresponding paths
        return images, image_paths

    def preprocess_images(self, images):
        """
        This method resizes and rescales the images before running them through
        the Darknet model.
        Args:
            images (list): List of images as numpy.ndarrays.
        Returns:
            Tuple of (pimages, image_shapes).
            pimages (numpy.ndarray): numpy.ndarray containing the preprocessed
                images.
            image_shapes (numpy.ndarray): numpy.ndarray containing the original
                shapes of the images.
        """
        # Initialize lists to hold the preprocessed images and their shapes
        pimages = []
        image_shapes = []

        # Loop through all images
        for image in images:
            # Get the original shape of the image
            image_shapes.append(np.array([image.shape[0], image.shape[1]]))
            # Resize the image to match the input shape of the Darknet model
            pimage = cv2.resize(image, (self.model.input.shape[1],
                                        self.model.input.shape[2]),
                                interpolation=cv2.INTER_CUBIC)
            # Rescale all pixel values between 0 and 1
            pimage = pimage / 255
            # Add the preprocessed image to the list
            pimages.append(pimage)

        # Convert the list of preprocessed images to a numpy array
        pimages = np.array(pimages)

        # Convert the list of image shapes to a numpy array
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        This method displays the image with all boundary boxes, class names,
        and box scores.
        Args:
            image (numpy.ndarray): Containing an unprocessed image to display.
            boxes (numpy.ndarray): Containing the boundary boxes for the image.
            box_classes (numpy.ndarray): Containing the class indices for each
                box.
            box_scores (numpy.ndarray): Containing the box scores for each box.
            file_name (str): the file path where the original image is stored.
        Returns:
            None.
        """
        # Loop through all the boxes and draw them on the image
        for i, box in enumerate(boxes):
            # Extract the coordinates of the box converted to integers
            x1, y1, x2, y2 = box.astype(int)
            # Draw the box on the image (blue, thickness 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Prepare the text (class name and score rounded to 2 decimals)
            class_name = self.class_names[box_classes[i]]
            score = box_scores[i]
            text = "{} {:.2f}".format(class_name, score)
            # Determine where to place the text (5 pixels above the top-left
            # corner)
            text_pos = (x1, y1 - 5)
            # Draw the text on the image (red, scale 0.5, thickness 1,
            # anti-aliased)
            cv2.putText(image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)
        # Display the image with the file_name as the window title
        cv2.imshow(file_name, image)
        # Wait for a key press
        key = cv2.waitKey(0)
        # If 's' key is pressed, save the image in the 'detections' folder
        if key == ord('s'):
            # Create the detections directory if it doesn't exist
            if not os.path.exists('detections'):
                os.makedirs('detections')
            # Save the image with the same file_name in the detections folder
            save_path = os.path.join('detections', file_name)
            cv2.imwrite(save_path, image)
        # Close the image window
        cv2.destroyAllWindows()
        return None

    def predict(self, folder_path):
        """
        This method performs object detection on a folder of images.
        Args:
            folder_path (str): a string representing the path to the folder
            holding all the images to predict.
        Returns:
            Tuple of (predictions, image_paths).
            predictions (list): a list of tuples for each image of
                (boxes, box_classes, box_scores).
                boxes (numpy.ndarray): a numpy.ndarray of shape (grid_height,
                    grid_width, anchor_boxes, 4) containing the processed
                    boundary boxes for each output, respectively.
                box_classes (numpy.ndarray): a numpy.ndarray of shape
                    (grid_height, grid_width, anchor_boxes, classes) containing
                    the class indices for each output, respectively.
                box_scores (numpy.ndarray): a numpy.ndarray of shape
                    (grid_height, grid_width, anchor_boxes, 1) containing
                    the box scores for each output, respectively.
            image_paths (list): a list of image paths corresponding to each
            prediction in predictions.
        """
        # Load the images and their paths
        images, image_paths = self.load_images(folder_path)
        # Preprocess the images to make them compatible with YOLO
        pimages, image_shapes = self.preprocess_images(images)
        predictions = []
        # Load model predictions on preprocessed images
        model_predictions = self.model.predict(pimages)

        # Iterate over the model predictions and corresponding image details
        for img, img_path, img_shape, idx in zip(images, image_paths,
                                                 image_shapes,
                                                 range(len(pimages))):
            # Process the model predictions
            output = [model_predictions[j][idx]
                      for j in range(len(model_predictions))]
            # Process the model outputs
            boxes, box_confidences, box_class_probs = self.process_outputs(
                output, img_shape)
            # Filter boxes
            filtered_boxes, box_classes, box_scores = self.filter_boxes(
                boxes, box_confidences, box_class_probs)
            # Perform non-max suppression
            box_preds, pred_box_classes, pred_box_scores = \
                self.non_max_suppression(
                    filtered_boxes, box_classes, box_scores)
            # Append the predictions to the list
            predictions.append(
                (box_preds, pred_box_classes, pred_box_scores))
            # Get the image filename (without path)
            file_name = os.path.basename(img_path)
            # Display the image with all boundary boxes
            self.show_boxes(image=img, boxes=box_preds,
                            box_classes=pred_box_classes,
                            box_scores=pred_box_scores,
                            file_name=file_name)

        return predictions, image_paths
