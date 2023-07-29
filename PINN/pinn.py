import tensorflow as tf
import numpy as np
import  logging
from logging import config
import os
import sys
root_path = sys.path[0]
sys.path.append(root_path)
config.fileConfig(os.path.join(root_path,'log.conf'),defaults={'logfilename': 'pinn_conf.log'})


logging.info('=' * 80)
logging.info('Start')

num_points=1000
x = np.linspace(0, 1, num_points)
y = np.linspace(0, 1, num_points)
x, y = np.meshgrid(x, y)

# Flatten and stack to create a (num_points, 2) array
inputs = np.stack([x.flatten(), y.flatten()], axis=-1)

# Convert to a TensorFlow tensor
inputs = tf.constant(inputs, dtype=tf.float32)

# Let's say you have boundary conditions u(x, 0) = 0, u(x, 1) = 1, u(0, y) = 0, u(1, y) = 1

# Boundary points
boundary_points = np.array([[0, y] for y in np.linspace(0, 1, num_points)] + 
                           [[1, y] for y in np.linspace(0, 1, num_points)] + 
                           [[x, 0] for x in np.linspace(0, 1, num_points)] + 
                           [[x, 1] for x in np.linspace(0, 1, num_points)])

# Corresponding boundary values
boundary_values = np.array([0]*num_points + [1]*num_points + [0]*num_points + [1]*num_points)

# Convert to TensorFlow tensors
boundary_points = tf.constant(boundary_points, dtype=tf.float32)
boundary_values = tf.constant(boundary_values, dtype=tf.float32)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, input_shape=(2,), activation=tf.nn.tanh),
    tf.keras.layers.Dense(20, activation=tf.nn.tanh),
    tf.keras.layers.Dense(1)
])

def loss(model, inputs, boundary_points, boundary_values):
    with tf.GradientTape(persistent=True) as outer_tape:
        outer_tape.watch(inputs)
        with tf.GradientTape(persistent=True) as inner_tape:
            inner_tape.watch(inputs)
            model_outputs = model(inputs)
        # Compute first order derivatives
        grads = inner_tape.gradient(model_outputs, inputs)
        u_x = grads[:, 0]
        u_y = grads[:, 1]
    # Compute second order derivatives
    u_xx = outer_tape.gradient(u_x, inputs)[:, 0]
    u_yy = outer_tape.gradient(u_y, inputs)[:, 1]
    del outer_tape, inner_tape

    # Compute physics loss
    # Assuming Laplace's equation: ∆u = 0
    physics_loss = tf.reduce_mean(tf.square(u_xx + u_yy))

    # Compute boundary loss
    boundary_outputs = model(boundary_points)
    boundary_loss = tf.reduce_mean(tf.square(boundary_outputs - boundary_values))

    return physics_loss + boundary_loss


optimizer = tf.keras.optimizers.Adam()

def train_step(model, inputs, boundary_points, boundary_values):
    with tf.GradientTape() as tape:
        current_loss = loss(model, inputs, boundary_points, boundary_values)
    gradients = tape.gradient(current_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return current_loss

# Training loop
for epoch in range(1000):
    current_loss = train_step(model, inputs, boundary_points, boundary_values)
    if epoch % 100 == 0:
        logging.info(f"Epoch: {epoch}, Loss: {current_loss}")