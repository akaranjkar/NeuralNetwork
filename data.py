import numpy as np

class Dataset:
    class XOR:
        """ XOR data set """
        input = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])

        target = np.array([
            [0],
            [1],
            [1],
            [0]
        ])

    class Identity:
        """ Identity data set """
        # Identity set
        input = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])

        target = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])

    class Tennis:
        """ Tennis dats set """
        # Outlook: Sunny, Overcast, Rain
        # Temperature: Hot, Mild, Cool
        # Humidity: High, Normal
        # Wind: Weak, Strong
        # PlayTennis: Yes, No

        input = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [2, 1, 0, 0],
            [2, 2, 1, 0],
            [2, 2, 1, 1],
            [1, 2, 1, 1],
            [0, 1, 0, 0],
            [0, 2, 1, 0],
            [2, 1, 1, 0]
        ])

        target = np.array([
            [1],
            [1],
            [0],
            [0],
            [0],
            [1],
            [0],
            [1],
            [0],
            [0]
        ])

        test_input = np.array([
            [0, 1, 1, 1],
            [1, 1, 0, 1],
            [1, 0, 1, 0],
            [2, 1, 0, 1]
        ])

        test_target = np.array([
            [0],
            [0],
            [0],
            [1]
        ])

    class Iris:
        """ Iris data set """
        # sepal - length: continuous
        # sepal - width: continuous
        # petal - length: continuous
        # petal - width: continuous
        # Iris: Iris-setosa, Iris-versicolor, Iris-virginica

        input = np.array([
            [4.3, 3.0, 1.1, 0.1],
            [4.4, 2.9, 1.4, 0.2],
            [5.9, 3.2, 4.8, 1.8],
            [6.0, 2.2, 4.0, 1.0],
            [6.0, 2.2, 5.0, 1.5],
            [6.0, 2.7, 5.1, 1.6],
            [6.0, 2.9, 4.5, 1.5],
            [4.4, 3.0, 1.3, 0.2],
            [6.7, 3.3, 5.7, 2.1],
            [6.7, 3.3, 5.7, 2.5],
            [6.8, 2.8, 4.8, 1.4],
            [6.8, 3.0, 5.5, 2.1],
            [6.8, 3.2, 5.9, 2.3],
            [6.9, 3.1, 4.9, 1.5],
            [5.1, 3.4, 1.5, 0.2],
            [5.1, 3.5, 1.4, 0.2],
            [5.6, 3.0, 4.1, 1.3],
            [5.6, 3.0, 4.5, 1.5],
            [5.7, 2.5, 5.0, 2.0],
            [5.7, 2.6, 3.5, 1.0],
            [5.7, 2.8, 4.1, 1.3],
            [5.7, 2.8, 4.5, 1.3],
            [5.8, 4.0, 1.2, 0.2],
            [5.9, 3.0, 4.2, 1.5],
            [5.9, 3.0, 5.1, 1.8],
            [6.9, 3.1, 5.1, 2.3],
            [6.9, 3.1, 5.4, 2.1],
            [6.9, 3.2, 5.7, 2.3],
            [7.0, 3.2, 4.7, 1.4],
            [7.1, 3.0, 5.9, 2.1],
            [4.8, 3.4, 1.6, 0.2],
            [4.8, 3.4, 1.9, 0.2],
            [4.9, 2.4, 3.3, 1.0],
            [4.9, 2.5, 4.5, 1.7],
            [4.9, 3.0, 1.4, 0.2],
            [4.9, 3.1, 1.5, 0.1],
            [4.9, 3.1, 1.5, 0.1],
            [4.9, 3.1, 1.5, 0.1],
            [5.0, 2.0, 3.5, 1.0],
            [5.0, 2.3, 3.3, 1.0],
            [5.0, 3.0, 1.6, 0.2],
            [5.0, 3.2, 1.2, 0.2],
            [7.2, 3.6, 6.1, 2.5],
            [7.3, 2.9, 6.3, 1.8],
            [7.4, 2.8, 6.1, 1.9],
            [7.6, 3.0, 6.6, 2.1],
            [7.7, 2.6, 6.9, 2.3],
            [5.1, 3.8, 1.6, 0.2],
            [5.1, 3.8, 1.9, 0.4],
            [5.2, 2.7, 3.9, 1.4],
            [5.2, 3.4, 1.4, 0.2],
            [5.2, 3.5, 1.5, 0.2],
            [5.2, 4.1, 1.5, 0.1],
            [5.3, 3.7, 1.5, 0.2],
            [5.4, 3.0, 4.5, 1.5],
            [5.4, 3.4, 1.5, 0.4],
            [5.4, 3.4, 1.7, 0.2],
            [5.4, 3.7, 1.5, 0.2],
            [5.4, 3.9, 1.3, 0.4],
            [5.4, 3.9, 1.7, 0.4],
            [5.5, 2.3, 4.0, 1.3],
            [5.5, 2.4, 3.7, 1.0],
            [5.5, 2.4, 3.8, 1.1],
            [6.3, 2.9, 5.6, 1.8],
            [6.3, 3.3, 4.7, 1.6],
            [6.3, 3.3, 6.0, 2.5],
            [6.3, 3.4, 5.6, 2.4],
            [6.4, 2.7, 5.3, 1.9],
            [4.6, 3.6, 1.0, 0.2],
            [5.0, 3.5, 1.3, 0.3],
            [5.0, 3.5, 1.6, 0.6],
            [5.0, 3.6, 1.4, 0.2],
            [5.1, 2.5, 3.0, 1.1],
            [7.7, 2.8, 6.7, 2.0],
            [7.7, 3.0, 6.1, 2.3],
            [7.7, 3.8, 6.7, 2.2],
            [7.9, 3.8, 6.4, 2.0],
            [5.0, 3.3, 1.4, 0.2],
            [6.1, 2.8, 4.0, 1.3],
            [6.1, 2.8, 4.7, 1.2],
            [6.1, 2.9, 4.7, 1.4],
            [6.1, 3.0, 4.6, 1.4],
            [6.1, 3.0, 4.9, 1.8],
            [6.2, 2.2, 4.5, 1.5],
            [6.2, 2.8, 4.8, 1.8],
            [6.4, 2.8, 5.6, 2.1],
            [6.4, 2.8, 5.6, 2.2],
            [6.4, 2.9, 4.3, 1.3],
            [6.4, 3.1, 5.5, 1.8],
            [6.4, 3.2, 4.5, 1.5],
            [6.4, 3.2, 5.3, 2.3],
            [6.5, 2.8, 4.6, 1.5],
            [6.5, 3.0, 5.2, 2.0],
            [5.1, 3.5, 1.4, 0.3],
            [5.1, 3.7, 1.5, 0.4],
            [5.1, 3.8, 1.5, 0.3],
            [5.1, 3.3, 1.7, 0.5],
            [6.0, 3.0, 4.8, 1.8],
            [6.0, 3.4, 4.5, 1.6],
            [6.1, 2.6, 5.6, 1.4]
        ])

        target = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1]
        ])

        test_input = np.array([
            [6.5, 3.0, 5.5, 1.8],
            [6.7, 3.1, 4.4, 1.4],
            [6.7, 3.1, 4.7, 1.5],
            [6.7, 3.1, 5.6, 2.4],
            [4.4, 3.2, 1.3, 0.2],
            [4.5, 2.3, 1.3, 0.3],
            [6.5, 3.0, 5.8, 2.2],
            [6.5, 3.2, 5.1, 2.0],
            [6.6, 2.9, 4.6, 1.3],
            [6.6, 3.0, 4.4, 1.4],
            [6.7, 2.5, 5.8, 1.8],
            [6.7, 3.0, 5.0, 1.7],
            [6.7, 3.0, 5.2, 2.3],
            [4.6, 3.1, 1.5, 0.2],
            [4.6, 3.2, 1.4, 0.2],
            [4.6, 3.4, 1.4, 0.3],
            [6.2, 2.9, 4.3, 1.3],
            [6.2, 3.4, 5.4, 2.3],
            [6.3, 2.3, 4.4, 1.3],
            [6.3, 2.5, 4.9, 1.5],
            [4.7, 3.2, 1.3, 0.2],
            [4.7, 3.2, 1.6, 0.2],
            [4.8, 3.0, 1.4, 0.1],
            [5.5, 2.5, 4.0, 1.3],
            [5.5, 2.6, 4.4, 1.2],
            [5.5, 3.5, 1.3, 0.2],
            [5.5, 4.2, 1.4, 0.2],
            [5.6, 2.5, 3.9, 1.1],
            [5.6, 2.7, 4.2, 1.3],
            [5.6, 2.8, 4.9, 2.0],
            [5.6, 2.9, 3.6, 1.3],
            [5.0, 3.4, 1.5, 0.2],
            [5.0, 3.4, 1.6, 0.4],
            [6.3, 2.5, 5.0, 1.9],
            [6.3, 2.7, 4.9, 1.8],
            [6.3, 2.8, 5.1, 1.5],
            [4.8, 3.0, 1.4, 0.3],
            [4.8, 3.1, 1.6, 0.2],
            [7.2, 3.0, 5.8, 1.6],
            [7.2, 3.2, 6.0, 1.8],
            [5.7, 2.9, 4.2, 1.3],
            [5.7, 3.0, 4.2, 1.2],
            [5.7, 3.8, 1.7, 0.3],
            [5.7, 4.4, 1.5, 0.4],
            [5.8, 2.6, 4.0, 1.2],
            [5.8, 2.7, 3.9, 1.2],
            [5.8, 2.7, 4.1, 1.0],
            [5.8, 2.7, 5.1, 1.9],
            [5.8, 2.7, 5.1, 1.9],
            [5.8, 2.8, 5.1, 2.4]
        ])

        test_target = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1]
        ])

    class IrisNoisy:
        """ Iris noisy data set """
        # sepal - length: continuous
        # sepal - width: continuous
        # petal - length: continuous
        # petal - width: continuous
        # Iris: Iris-setosa, Iris-versicolor, Iris-virginica

        input = np.array([
            [4.3, 3.0, 1.1, 0.1],
            [4.4, 2.9, 1.4, 0.2],
            [5.9, 3.2, 4.8, 1.8],
            [6.0, 2.2, 4.0, 1.0],
            [6.0, 2.2, 5.0, 1.5],
            [6.0, 2.7, 5.1, 1.6],
            [6.0, 2.9, 4.5, 1.5],
            [4.4, 3.0, 1.3, 0.2],
            [6.7, 3.3, 5.7, 2.1],
            [6.7, 3.3, 5.7, 2.5],
            [6.8, 2.8, 4.8, 1.4],
            [6.8, 3.0, 5.5, 2.1],
            [6.8, 3.2, 5.9, 2.3],
            [6.9, 3.1, 4.9, 1.5],
            [5.1, 3.4, 1.5, 0.2],
            [5.1, 3.5, 1.4, 0.2],
            [5.6, 3.0, 4.1, 1.3],
            [5.6, 3.0, 4.5, 1.5],
            [5.7, 2.5, 5.0, 2.0],
            [5.7, 2.6, 3.5, 1.0],
            [5.7, 2.8, 4.1, 1.3],
            [5.7, 2.8, 4.5, 1.3],
            [5.8, 4.0, 1.2, 0.2],
            [5.9, 3.0, 4.2, 1.5],
            [5.9, 3.0, 5.1, 1.8],
            [6.9, 3.1, 5.1, 2.3],
            [6.9, 3.1, 5.4, 2.1],
            [6.9, 3.2, 5.7, 2.3],
            [7.0, 3.2, 4.7, 1.4],
            [7.1, 3.0, 5.9, 2.1],
            [4.8, 3.4, 1.6, 0.2],
            [4.8, 3.4, 1.9, 0.2],
            [4.9, 2.4, 3.3, 1.0],
            [4.9, 2.5, 4.5, 1.7],
            [4.9, 3.0, 1.4, 0.2],
            [4.9, 3.1, 1.5, 0.1],
            [4.9, 3.1, 1.5, 0.1],
            [4.9, 3.1, 1.5, 0.1],
            [5.0, 2.0, 3.5, 1.0],
            [5.0, 2.3, 3.3, 1.0],
            [5.0, 3.0, 1.6, 0.2],
            [5.0, 3.2, 1.2, 0.2],
            [7.2, 3.6, 6.1, 2.5],
            [7.3, 2.9, 6.3, 1.8],
            [7.4, 2.8, 6.1, 1.9],
            [7.6, 3.0, 6.6, 2.1],
            [7.7, 2.6, 6.9, 2.3],
            [5.1, 3.8, 1.6, 0.2],
            [5.1, 3.8, 1.9, 0.4],
            [5.2, 2.7, 3.9, 1.4],
            [5.2, 3.4, 1.4, 0.2],
            [5.2, 3.5, 1.5, 0.2],
            [5.2, 4.1, 1.5, 0.1],
            [5.3, 3.7, 1.5, 0.2],
            [5.4, 3.0, 4.5, 1.5],
            [5.4, 3.4, 1.5, 0.4],
            [5.4, 3.4, 1.7, 0.2],
            [5.4, 3.7, 1.5, 0.2],
            [5.4, 3.9, 1.3, 0.4],
            [5.4, 3.9, 1.7, 0.4],
            [5.5, 2.3, 4.0, 1.3],
            [5.5, 2.4, 3.7, 1.0],
            [5.5, 2.4, 3.8, 1.1],
            [6.3, 2.9, 5.6, 1.8],
            [6.3, 3.3, 4.7, 1.6],
            [6.3, 3.3, 6.0, 2.5],
            [6.3, 3.4, 5.6, 2.4],
            [6.4, 2.7, 5.3, 1.9],
            [4.6, 3.6, 1.0, 0.2],
            [5.0, 3.5, 1.3, 0.3],
            [5.0, 3.5, 1.6, 0.6],
            [5.0, 3.6, 1.4, 0.2],
            [5.1, 2.5, 3.0, 1.1],
            [7.7, 2.8, 6.7, 2.0],
            [7.7, 3.0, 6.1, 2.3],
            [7.7, 3.8, 6.7, 2.2],
            [7.9, 3.8, 6.4, 2.0],
            [5.0, 3.3, 1.4, 0.2],
            [6.1, 2.8, 4.0, 1.3],
            [6.1, 2.8, 4.7, 1.2],
            [6.1, 2.9, 4.7, 1.4],
            [6.1, 3.0, 4.6, 1.4],
            [6.1, 3.0, 4.9, 1.8],
            [6.2, 2.2, 4.5, 1.5],
            [6.2, 2.8, 4.8, 1.8],
            [6.4, 2.8, 5.6, 2.1],
            [6.4, 2.8, 5.6, 2.2],
            [6.4, 2.9, 4.3, 1.3],
            [6.4, 3.1, 5.5, 1.8],
            [6.4, 3.2, 4.5, 1.5],
            [6.4, 3.2, 5.3, 2.3],
            [6.5, 2.8, 4.6, 1.5],
            [6.5, 3.0, 5.2, 2.0],
            [5.1, 3.5, 1.4, 0.3],
            [5.1, 3.7, 1.5, 0.4],
            [5.1, 3.8, 1.5, 0.3],
            [5.1, 3.3, 1.7, 0.5],
            [6.0, 3.0, 4.8, 1.8],
            [6.0, 3.4, 4.5, 1.6],
            [6.1, 2.6, 5.6, 1.4]
        ])

        target = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ])