import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt

# List of classes to download
classes = ['cat', 'dog', 'apple', 'car', 'airplane', 'alarm clock', 'bathtub', 'bed', 'door',
           'dishwasher', 'helicopter', 'headphones', 'fence', 'garden hose', 'microwave', 'pizza', 'sun', 'whale']

# Directory to save the downloaded .npy files
data_dir = 'quickdraw_data'
os.makedirs(data_dir, exist_ok=True)

def download_data(classes, data_dir):
    base_url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
    for cls in classes:
        cls_name = cls.replace(' ', '%20')
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
        print(f'Loading {cls} data with label {idx}...')
        data = np.load(filepath)
        data = data[:max_items_per_class]
        images.append(data)
        labels.extend([idx] * len(data))
        print(f"{cls} data shape: {data.shape}, labels assigned: {len(labels)}")
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
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    download_data(classes, data_dir)
    images, labels = load_data(classes, data_dir)
    images, labels = preprocess_data(images, labels)
    print(f'Total images: {images.shape}')
    print(f'Total labels: {labels.shape}')

    # Shuffle the data
    from sklearn.utils import shuffle
    images, labels = shuffle(images, labels, random_state=42)

    # Check class distribution
    import collections
    label_indices = np.argmax(labels, axis=1)
    class_counts = collections.Counter(label_indices)
    print("Class distribution:")
    for idx, count in class_counts.items():
        print(f"Class {classes[idx]}: {count} samples")

    # Split the data into training and testing sets
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=label_indices)

    # Further split training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=np.argmax(y_train_full, axis=1))

    # Data Augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.10,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )
    datagen.fit(X_train)

    input_shape = (28, 28, 1)
    num_classes = len(classes)
    model = create_model(input_shape, num_classes)

    # Implement Early Stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # TensorBoard Callback
    log_dir = "logs"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # ReduceLROnPlateau Callback
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    # Train the model with data augmentation
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=128),
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, tensorboard_callback, reduce_lr]
    )

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_acc:.4f}')

    # Plot accuracy and loss graphs
    plt.figure(figsize=(12, 4))
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Save the model
    model.save('DoodleVision.keras')
    print('Model training complete and saved as DoodleVision.keras')

if __name__ == "__main__":
    main()
