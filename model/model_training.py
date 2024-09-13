import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import requests
from tqdm import tqdm

# List of classes to download
classes = ['cat', 'dog', 'apple', 'banana', 'car', 'airplane', 'alarm clock', 'bathtub', 'bed', 'door',
           'dishwasher', 'helicopter', 'headphones', 'fence', 'garden hose', 'microwave', 'pizza', 'sun', 'whale']

# Directory to save the downloaded .npy files
data_dir = 'quickdraw_data'
os.makedirs(data_dir, exist_ok=True)

def download_data(classes, data_dir):
    base_url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
    for cls in classes:
        cls_name = cls.replace('_', '%20')
        url = base_url + cls_name + '.npy'
        filepath = os.path.join(data_dir, cls + '.npy')
        if not os.path.exists(filepath):
            print(f'Downloading {cls} data...')
            response = requests.get(url, stream=True)
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            with open(filepath, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print("ERROR: Something went wrong with the download")
        else:
            print(f'{cls} data already downloaded.')


def load_data(classes, data_dir, max_items_per_class=20000):
    images = []
    labels = []
    for idx, cls in enumerate(classes):
        filepath = os.path.join(data_dir, cls + '.npy')
        print(f'Loading {cls} data...')
        data = np.load(filepath)
        data = data[:max_items_per_class]
        images.append(data)
        labels.extend([idx] * len(data))
    images = np.concatenate(images, axis=0)
    labels = np.array(labels)
    return images, labels


def preprocess_data(images, labels):
    # Normalize pixel values
    images = images.astype('float32') / 255.0
    # Reshape images to (28, 28, 1)
    images = images.reshape(-1, 28, 28, 1)
    # Convert labels to one-hot encoding
    labels = tf.keras.utils.to_categorical(labels, num_classes=len(classes))
    return images, labels


def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Flatten(),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    download_data(classes, data_dir)
    images, labels = load_data(classes, data_dir)
    images, labels = preprocess_data(images, labels)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels.argmax(axis=1))

    input_shape = (28, 28, 1)
    num_classes = len(classes)
    model = create_model(input_shape, num_classes)

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=128,
              validation_data=(X_test, y_test))

    # Save the model
    model.save('DoodleVision.keras')
    print('Model training complete and saved as DoodleVision.keras')


if __name__ == "__main__":
    main()