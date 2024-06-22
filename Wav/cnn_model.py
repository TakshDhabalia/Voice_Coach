# cnn_model.py

import tensorflow as tf

def create_cnn_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(13, 32, 1)),  # Define input shape explicitly
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

if __name__ == '__main__':
    num_classes = 10  # Placeholder, update this dynamically in train_model.py
    model = create_cnn_model(num_classes)
    model.save('voice_coach_model.keras')  # Save using the .keras extension
