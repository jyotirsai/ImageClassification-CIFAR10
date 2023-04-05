import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras

from data import Data

# return CIFAR-10 data
X_train, X_val, X_test, y_train, y_val, y_test = Data()

# define baseline model
def BaselineCNN():
  model = keras.Sequential()
  model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", input_shape=(32, 32, 3)))
  model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu"))
  model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

  model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same"))
  model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu"))
  model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

  model.add(keras.layers.Flatten())

  model.add(keras.layers.Dense(units=512, activation="relu"))
  model.add(keras.layers.Dense(units=10, activation="softmax"))

  opt = keras.optimizers.Adam(learning_rate=0.01)
  model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=['accuracy'])

  return model

# baseline results
model = BaselineCNN()
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20)
model.evaluate(X_test, y_test)

# define tuning class
class MyHyperModel(kt.HyperModel):
  def build(self, hp):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(
        filters=hp.Int("conv_1_layers", 32, 96, step=32),
        kernel_size=(3,3),
        activation="relu", 
        input_shape=(32, 32, 3)
    ))

    model.add(keras.layers.Conv2D(
        filters=hp.Int("conv_2_layers", 32, 96, step=32),
        kernel_size=(3,3),
        activation="relu"
    ))

    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

    model.add(keras.layers.Dropout(hp.Float('dropout_2', min_value=0, max_value=1, step=0.2)))

    model.add(keras.layers.Conv2D(
        filters=64, 
        kernel_size=(3,3), 
        activation="relu",
        padding="same",
    ))

    model.add(keras.layers.Conv2D(
        filters=64, 
        kernel_size=(3,3), 
        activation="relu",
    ))

    model.add(keras.layers.Dropout(hp.Float('dropout_3', min_value=0, max_value=1, step=0.2)))

    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=hp.Int("dense_1", 64, 512, step=32), activation="relu"))
    model.add(keras.layers.Dense(units=10, activation="softmax"))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model
  
  def fit(self, hp, model, *args, **kwargs):
    return model.fit(
        *args,
        batch_size=hp.Int("batch_size", 32, 256, step=32),
        **kwargs,
    )

# tuning results
cnn_tuner = kt.Hyperband(MyHyperModel(), objective='val_accuracy', overwrite=True)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
cnn_tuner.search(X_train, y_train, epochs=20, validation_data=(X_val, y_val), callbacks=[stop_early])
cnn_tuner.results_summary()

# evaluate best model
best_model = cnn_tuner.get_best_models()[0]
best_model.evaluate(X_test, y_test)