import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras

from data import Data

# return CIFAR-10 data
X_train, X_val, X_test, y_train, y_val, y_test = Data()

# define baseline model
def BaselineDNN():
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Flatten(input_shape=(32,32,3)))
  model.add(keras.layers.Dense(units=1536, activation='relu'))
  model.add(keras.layers.Dense(units=768, activation='relu'))
  model.add(keras.layers.Dense(units=384, activation='relu'))
  model.add(keras.layers.Dense(units=128, activation='relu'))
  model.add(keras.layers.Dense(units=10, activation='softmax'))

  opt = keras.optimizers.Adam(learning_rate=0.01)
  model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=['accuracy'])

  return model

# baseline results
model = BaselineDNN()
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=256)
model.evaluate(X_test, y_test)

# define tuning class
class MyHyperModel(kt.HyperModel):
  def build(self, hp):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(32,32,3)))
    model.add(keras.layers.Dense(units=hp.Int('layer_1', 1000, 2280, step=128), activation='relu'))
    model.add(keras.layers.Dense(units=768, activation='relu'))
    model.add(keras.layers.Dropout(hp.Float('dropout_1', min_value=0, max_value=1, step=0.2)))
    model.add(keras.layers.Dense(units=384, activation='relu'))
    model.add(keras.layers.Dense(units=128, activation='relu'))
    model.add(keras.layers.Dropout(hp.Float('dropout_2', min_value=0, max_value=1, step=0.2)))
    model.add(keras.layers.Dense(units=10, activation='softmax'))

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
dnn_tuner = kt.Hyperband(MyHyperModel(), objective='val_accuracy', overwrite=True)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
dnn_tuner.search(X_train, y_train, epochs=20, validation_data=(X_val, y_val), callbacks=[stop_early])
dnn_tuner.results_summary()

# evaluate best model
best_model = dnn_tuner.get_best_models()[0]
best_model.evaluate(X_test, y_test)