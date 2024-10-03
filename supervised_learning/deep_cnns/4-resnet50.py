#!/usr/bin/env python3
""" This module defines the resnet50 function. """
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    This function builds the ResNet-50 architecture as described in Deep
    Residual Learning for Image Recognition (2015).
    All convolutions inside the blocks are followed by batch normalization
    along the channels axis and a rectified linear activation (ReLU),
    respectively.
    All weights use he normal initialization.
    Returns: the Keras model.
    """
    he_normal = K.initializers.he_normal(seed=0)
    X = K.Input(shape=(224, 224, 3))
    # Stage 1
    conv1 = K.layers.Conv2D(filters=64, kernel_size=7, padding='same',
                            strides=2,
                            kernel_initializer=he_normal)(X)
    batch_norm1 = K.layers.BatchNormalization(axis=3)(conv1)
    activation1 = K.layers.Activation('relu')(batch_norm1)
    max_pool1 = K.layers.MaxPooling2D(pool_size=3, strides=2,
                                      padding='same')(activation1)
    # Stage 2
    projection_block1 = projection_block(max_pool1, [64, 64, 256], s=1)
    identity_block1 = identity_block(projection_block1, [64, 64, 256])
    identity_block2 = identity_block(identity_block1, [64, 64, 256])
    # Stage 3
    projection_block2 = projection_block(identity_block2, [128, 128, 512])
    identity_block3 = identity_block(projection_block2, [128, 128, 512])
    identity_block4 = identity_block(identity_block3, [128, 128, 512])
    identity_block5 = identity_block(identity_block4, [128, 128, 512])
    # Stage 4
    projection_block3 = projection_block(identity_block5, [256, 256, 1024])
    identity_block6 = identity_block(projection_block3, [256, 256, 1024])
    identity_block7 = identity_block(identity_block6, [256, 256, 1024])
    identity_block8 = identity_block(identity_block7, [256, 256, 1024])
    identity_block9 = identity_block(identity_block8, [256, 256, 1024])
    identity_block10 = identity_block(identity_block9, [256, 256, 1024])
    # Stage 5
    projection_block4 = projection_block(identity_block10, [512, 512, 2048])
    identity_block11 = identity_block(projection_block4, [512, 512, 2048])
    identity_block12 = identity_block(identity_block11, [512, 512, 2048])
    # Average pooling
    avg_pool = K.layers.AveragePooling2D(pool_size=7, strides=None,
                                         padding='same')(identity_block12)

    # Fully connected layer
    dense = K.layers.Dense(units=1000, activation='softmax',
                           kernel_initializer=he_normal)(avg_pool)
    model = K.models.Model(inputs=X, outputs=dense)
    return model
