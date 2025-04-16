# Imports
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from preprocess import Preprocess

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

    # # Load dataset
    mnist = tf.keras.datasets.mnist    

    # # Split the dataset into training and test sets
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # # Normalise the images 
    # x_train, x_test, y_train, y_test = x_train / 255.0, x_test / 255.0, y_train / 255.0, y_test / 255.0

    preprocessed_data = Preprocess(mnist)
    x_train, x_test, y_train, y_test = preprocessed_data.split_data()
    x_train, x_test, y_train, y_test = preprocessed_data.normalise_images()
    
    # Inital dataset investigation
    # display_images(x_train, y_train)

    # Setup model layers
    model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
    ])

    # Compile model with optimiser, loss function and metrics 
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    # Train model
    model.fit(x_train, y_train, epochs=5)

    # Evaluate model on test data using metrics
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
    
    pass

if __name__ == "__main__":
    main()