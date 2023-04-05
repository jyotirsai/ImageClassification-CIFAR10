# Description

A study of common image classification algorithms on the CIFAR-10 dataset from the University of Toronto.

The following machine learning models were used:

1. Softmax (Multi-class Logistic Regression)
2. Deep Neural Network (DNN)
3. Convolutional Neural Network (CNN)

A train-validation-test framework was implemented along with systematic hyperparamater tuning using the Keras-Tuner library. The following results were
achieved:

|         | Baseline | Tuned |
| ------- | -------- | ----- |
| Softmax | 38.57    | 39.08 |
| DNN     | 45.35    | 54.23 |
| CNN     | 64.80    | 78.96 |

Due to a lack of computational resources, more complex CNNs were not explored.

## Requirements

- keras_tuner==1.3.4
- scikit_learn==1.2.2
- tensorflow==2.10.0
