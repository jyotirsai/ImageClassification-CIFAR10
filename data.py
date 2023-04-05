import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

def Data():
    """
        Normalize CIFAR-10 dataset with some adjustsments

        returns (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # split training into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    return (X_train, X_val, X_test, y_train, y_val, y_test)