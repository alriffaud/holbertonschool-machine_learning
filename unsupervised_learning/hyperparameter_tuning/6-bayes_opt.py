#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from keras import layers, models
import GPyOpt
import os
import matplotlib.pyplot as plt


# Define the objective function
def objective_function(hyperparams):
    """
    Objective function to optimize.
    hyperparams: numpy array of shape (1, 5) representing the hyperparameters.
    hyperparams[0][0]: learning_rate
    hyperparams[0][1]: units
    hyperparams[0][2]: dropout_rate
    hyperparams[0][3]: l2_regularization
    hyperparams[0][4]: batch_size
    """
    # Extract hyperparameters
    learning_rate = hyperparams[0][0]
    units = int(hyperparams[0][1])
    dropout_rate = hyperparams[0][2]
    l2_reg = hyperparams[0][3]
    batch_size = int(hyperparams[0][4])

    # Define the model
    model = models.Sequential([
        layers.Dense(units, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5)

    # Checkpoint callback
    checkpoint_filepath = f"model_lr{learning_rate:.4f}_units{units}_dropout\
{dropout_rate:.2f}_l2{l2_reg:.4f}_bs{batch_size}.h5"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, save_best_only=True,
        save_weights_only=True)

    # Create a dummy dataset
    X_train = np.random.rand(1000, 20)
    y_train = np.random.randint(0, 2, 1000)
    X_val = np.random.rand(200, 20)
    y_val = np.random.randint(0, 2, 200)

    # Train the model
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        batch_size=batch_size,
                        epochs=50,
                        verbose=0,
                        callbacks=[early_stopping, checkpoint_callback])

    # Get the best validation loss
    val_loss = min(history.history['val_loss'])

    # Return the negative validation loss (since GPyOpt minimizes the objective
    # function)
    return val_loss


# Define the bounds of the hyperparameters
bounds = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-4, 1e-1)},
    {'name': 'units', 'type': 'discrete', 'domain': (16, 32, 64, 128, 256)},
    {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.1, 0.5)},
    {'name': 'l2_regularization', 'type': 'continuous', 'domain': (
        1e-4, 1e-2)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (16, 32, 64, 128)}
]

# Perform Bayesian Optimization
optimizer = GPyOpt.methods.BayesianOptimization(
    f=objective_function,
    domain=bounds,
    maximize=False
)

# Run the optimization
optimizer.run_optimization(max_iter=30)

# Save the results
optimizer.plot_convergence()
plt.savefig("convergence_plot.png")

# Write optimization results to file
with open("bayes_opt.txt", "w") as file:
    file.write(f"Best hyperparameters: {optimizer.x_opt}\n")
    file.write(f"Best objective value: {optimizer.fx_opt}\n")
    file.write(f"Convergence plot saved as 'convergence_plot.png'\n")
