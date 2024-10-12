#!/usr/bin/env python3
"""
This module transfers knowledge using ResNet152V2 with optimizations for
CIFAR-10.
"""

from tensorflow import keras as K
import tensorflow as tf


def preprocess_data(X, Y):
    """
    Pre-processes the data for your model:
    X is a numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR-10 data,
        where m is the number of data points
    Y is a numpy.ndarray of shape (m,) containing the CIFAR-10 labels for X

    Returns: X_p, Y_p
        X_p is a numpy.ndarray containing the preprocessed X
        Y_p is a numpy.ndarray containing the preprocessed Y
    """
    # Use the preprocess function of ResNet152V2
    X_p = K.applications.resnet_v2.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, num_classes=10)
    return X_p, Y_p


def build_model():
    """
    Builds a ResNet152V2 model with 10 classes, using ImageNet weights.
    The model is fine-tuned for CIFAR-10.
    """
    # Load ResNet152V2 pre-trained on ImageNet
    base_model = K.applications.ResNet152V2(weights='imagenet',
                                            include_top=False,
                                            input_shape=(224, 224, 3))

    # Freeze base model to retain pre-trained weights
    base_model.trainable = False

    # Input layer for CIFAR-10
    inputs = K.Input(shape=(32, 32, 3))
    resized_inputs = K.layers.Lambda(
        lambda x: tf.image.resize(x, (224, 224)))(inputs)

    # Get the base model output
    base_model_output = base_model(resized_inputs, training=False)

    # Add custom layers for CIFAR-10 classification
    x = K.layers.GlobalAveragePooling2D()(base_model_output)
    x = K.layers.Dense(512, activation='relu',
                       kernel_regularizer=K.regularizers.l2(0.001))(x)
    x = K.layers.Dropout(0.3)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    return K.Model(inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    # Preprocess data for ResNet152V2
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    # Data augmentation
    datagen = K.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(x_train)

    # Build and compile the model
    model = build_model()

    # Optimizer with learning rate schedule
    initial_learning_rate = 0.001
    lr_schedule = K.callbacks.LearningRateScheduler(
        lambda epoch: initial_learning_rate * (0.1 ** (epoch // 20)))
    optimizer = K.optimizers.Adam(learning_rate=initial_learning_rate)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # Callbacks for early stopping and saving the best model
    early_stopping = K.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5, restore_best_weights=True)
    checkpoint = K.callbacks.ModelCheckpoint(filepath='cifar10.keras',
                                             monitor='val_accuracy',
                                             save_best_only=True,
                                             mode='max')

    # Train the model with data augmentation
    history = model.fit(datagen.flow(x_train, y_train, batch_size=512),
                        epochs=50,
                        validation_data=(x_test, y_test),
                        callbacks=[early_stopping, checkpoint, lr_schedule],
                        verbose=1)

    # Now, we fine-tune the model by unfreezing the last few layers
    base_model = model.layers[2]  # MobileNetV2 is the 3rd layer in the model

    # Unfreeze the base model and train specific layers
    base_model.trainable = True
    for layer in base_model.layers[:-20]:  # Freeze all but the last 30 layers
        layer.trainable = False
    fine_tune_optimizer = K.optimizers.Adam(learning_rate=1e-5)
    model.compile(optimizer=fine_tune_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Fine-tune model with all layers trainable
    finetuning_history = model.fit(datagen.flow(x_train, y_train,
                                                batch_size=512),
                                   epochs=50,
                                   validation_data=(x_test, y_test),
                                   callbacks=[early_stopping, checkpoint],
                                   verbose=1)

    # Save final model
    model.save(filepath='cifar10.h5', save_format='h5')

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f'Test accuracy: {accuracy * 100:.2f}%')
