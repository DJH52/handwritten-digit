# Imports
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from preprocess import Preprocess

from PIL import Image
import numpy as np

def test_custom_image(model, image_path):
    # Load the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels

    # Preprocess the image
    img_array = np.array(img).astype('float32') / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 28, 28)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    print(f'Predicted class for the custom image: {predicted_class}')

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
    x_train, x_test = preprocessed_data.normalise_images()
    
    # Inital dataset investigation
    # display_images(x_train, y_train)

    # Setup model layers
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28)),  # Define the input shape explicitly
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile model with optimiser, loss function and metrics 
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
        
    # Train model
    model.fit(x_train, y_train, epochs=5)        

    # Evaluate model on test data using metrics
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

    print('\nTest accuracy:', test_acc)
    print('\nTest loss:', test_loss)

    probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
    
    predictions = probability_model.predict(x_test)

    test_image_position = 0
    predicted_class = np.argmax(predictions[test_image_position])
    print(f'Predicted class for the first test image:{predicted_class}')
    print(f'Actual Class:{y_test[test_image_position]}')

    custom_image_path = "7.png"  # Replace with your image path
    test_custom_image(probability_model, custom_image_path)

    pass

if __name__ == "__main__":
    main()