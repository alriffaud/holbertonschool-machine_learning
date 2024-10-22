#!/usr/bin/env python3
""" This module defines the class NST that performs Neural Style Transfer """
import numpy as np
import tensorflow as tf


class NST:
    """ This class performs tasks for Neural Style Transfer """
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor for NST class that initializes the following public
        instance attributes:
        - style_image (numpy.ndarray): the image used as a style reference.
        - content_image (numpy.ndarray): the image used as a content reference.
        - alpha (float): the weight for content cost.
        - beta (float): the weight for style cost.
        """
        error1 = "style_image must be a numpy.ndarray with shape (h, w, 3)"
        error2 = "content_image must be a numpy.ndarray with shape (h, w, 3)"
        if (not isinstance(style_image, np.ndarray) or style_image.ndim != 3
                or style_image.shape[-1] != 3):
            raise TypeError(error1)
        if (not isinstance(content_image, np.ndarray)
                or content_image.ndim != 3 or content_image.shape[-1] != 3):
            raise TypeError(error2)
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        # Load the model
        self.load_model()

    @staticmethod
    def scale_image(image):
        """
        This method rescales the image such that its pixels values are between
        0 and 1 and its largest side is 512 pixels.
        Args:
            image (numpy.ndarray): the image to be rescaled.
        """
        error = "image must be a numpy.ndarray with shape (h, w, 3)"
        if (not isinstance(image, np.ndarray) or len(image.shape) != 3
                or image.shape[-1] != 3):
            raise TypeError(error)
        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = int(w * h_new / h)
        else:
            w_new = 512
            h_new = int(h * w_new / w)
        # Resize the image with bicubic interpolation
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.image.resize(image, (h_new, w_new), method="bicubic")
        # Normalize the image pixel values to be in the range [0, 1]
        image = tf.clip_by_value(image / 255.0, 0, 1)
        return image[tf.newaxis, ...]

    def load_model(self):
        """
        This method loads the VGG19 model for Neural Style Transfer.
        It returns the model.
        """
        # Load the VGG19 model.
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet')
        # Replace MaxPooling layers with Average Pooling
        for layer in vgg.layers:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                layer.__class__ = tf.keras.layers.AveragePooling2D
        # Make sure that the model is non-trainable
        vgg.trainable = False
        # Get output layers corresponding to style and content layers
        outputs = [vgg.get_layer(layer).output for layer in self.style_layers]
        outputs.append(vgg.get_layer(self.content_layer).output)
        # Create a model that returns the outputs of the VGG19 model
        self.model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """
        This method calculates the gram matrices for the input layer.
        It returns the gram matrix.
        Args:
            input_layer (tf.Tensor): the layer from which to calculate the gram
            matrix.
        Returns:
            A tf.Tensor of shape (1, c, c) containing the gram matrix of the
            input layer.
        """
        error = "input_layer must be a tensor of rank 4"
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable))
                or len(input_layer.shape) != 4):
            raise TypeError(error)

        _, h, w, c = input_layer.shape
        # Reshape the features of the layer to a 2D matrix
        F = tf.reshape(input_layer, (h * w, c))
        # Calculate the gram matrix
        gram = tf.matmul(F, F, transpose_a=True)
        # Expand dimensions to have shape (1, c, c)
        gram = tf.expand_dims(gram, axis=0)

        # Normalize by number of locations (h * w) then return gram tensor
        input_shape = tf.shape(input_layer)
        nb_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)

        return gram / nb_locations
