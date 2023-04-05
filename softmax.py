import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras 

from data import Data

# return CIFAR-10 data
X_train, X_val, X_test, y_train, y_val, y_test = Data()

# define baseline model
def BaselineSoftmax():
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(32,32,3)))
    model.add(keras.layers.Dense(units=10, activation="softmax"))

    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    return model

# baseline results
model = BaselineSoftmax()
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=256)
model.evaluate(X_test, y_test)

# define tuning class
class TuningSoftmax(kt.HyperModel):
  def build(self, hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(32,32,3)))

    hp_l2reg = hp.Float("l2reg", min_value=0.001, max_value=10, step=10, sampling="log")
    model.add(keras.layers.Dense(units=10, activation="softmax", activity_regularizer=keras.regularizers.L2(hp_l2reg)))

    hp_learning_rate = hp.Float("learning_rate", min_value=0.0001, max_value=0.1, step=10, sampling="log")

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
softmax_tuner = kt.Hyperband(TuningSoftmax(), objective='val_accuracy', overwrite=True)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
softmax_tuner.search(X_train, y_train, epochs=20, validation_data=(X_val, y_val))
softmax_tuner.summary()

# evaluate best model
best_model = softmax_tuner.get_best_models()[0]
best_model.evaluate(X_train, y_train)
