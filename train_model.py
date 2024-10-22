import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load and preprocess the dataset
(X_train, y_train), _ = mnist.load_data()
X_train = X_train / 255.0
y_train = to_categorical(y_train, 10)

# Build a simple neural network model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
print("Training the model...")
model.fit(X_train, y_train, epochs=2, batch_size=32)
print("Model training completed.")

# Save the trained model
print("Saving the model as 'mnist_model.h5'...")
model.save('mnist_model.h5')
print("Model saved successfully.")
