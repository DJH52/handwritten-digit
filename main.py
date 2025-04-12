# Imports
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Functions 
def display_images(train_images, train_labels, number_of_images=25):
    """[A concise one-line summary of the function's purpose.]

    Args:
        [param_name] ([type hint]): [A clear description of the parameter.
            Include details about its purpose, expected values, and any
            constraints.]
        [another_param] ([type hint], optional): [Description of an optional
            parameter. Indicate that it's optional and specify its default
            value.]
        *[keyword_only_param] ([type hint]): [Description of a keyword-only
            parameter. Emphasize that it must be passed using its keyword.]

    Returns:
        [return_type hint]: [A description of the value returned by the
            function. If the function returns multiple values, describe each
            one.]

    Raises:
        [ExceptionType]: [A description of the specific exception that can be
            raised and the conditions under which it occurs.]
        [AnotherExceptionType]: [Description of another potential exception.]

    Examples:
        >>> [A simple example demonstrating a typical use case of the function.]
        [Expected output of the example.]
        >>> [Another example showcasing a different scenario or edge case.]
        [Expected output of the second example.]
    """
    

    plt.figure(figsize=(10, 10))
    for i in range(number_of_images):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(train_labels[i])
    plt.show()


def main():
    # hyperparameters = dict([('sape', 4139)])

    # Load dataset
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Inital dataset investigation
    display_images(x_train, y_train)
    
    
    
    pass

if __name__ == "__main__":
    pass