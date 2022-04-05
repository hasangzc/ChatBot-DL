# Import modules
import numpy as np
import tensorflow as tf
from datapreprocessing import traning, output

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

try:
    model = tf.keras.models.load_model("model.pkl")
except:
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Dense(128, input_shape=(len(traning[0]),), activation="relu")
    )
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(len(output[0]), activation="softmax"))

    # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    # fitting and saving the model
    model.fit(traning, output, epochs=200, batch_size=30, verbose=1)
    model.save("model.pkl")
