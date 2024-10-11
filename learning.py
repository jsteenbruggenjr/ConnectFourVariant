import os
import tensorflow as tf
from tensorflow.keras import layers, models

# the model will take in a board of shape (BOARD_WIDTH,BOARD_HEIGHT,2) and returns (2) scalars, the first being win chance and the second being loss chance
def new_model():
    input_shape = (7, 7, 2)
    input_layer = layers.Input(shape=input_shape)
    conv_output = layers.Conv2D(filters=50, kernel_size=(4, 4), activation='relu')(input_layer)
    conv_flatten = layers.Flatten()(conv_output)
    input_flatten = layers.Flatten()(input_layer)
    combined = layers.Concatenate()([conv_flatten, input_flatten])
    x = layers.Dense(100, activation='relu')(combined)
    for _ in range(15):
        x = layers.Dense(100, activation='relu')(x)
    output = layers.Dense(3)(x)
    prob_output = layers.Activation('softmax')(output)
    model = models.Model(inputs=input_layer, outputs=prob_output)
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model

def get_model(path):
    if os.path.exists(path):
        print("model found, loading...")
        return models.load_model(path)
    print("no model found. creating new one...")
    return new_model()