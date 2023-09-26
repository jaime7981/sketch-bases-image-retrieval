import os
import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import random
from sklearn.model_selection import train_test_split
from PIL import Image
import cairosvg
import io

# get path of running script
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

SKETCH_STATS_PATH = os.path.join(SCRIPT_PATH, 'info', 'stats.csv')
PHOTO_PATH = os.path.join(SCRIPT_PATH, 'rendered_256x256', '256x256', 'photo', 'tx_000000000000')
SKETCH_PATH = os.path.join(SCRIPT_PATH, 'rendered_256x256', '256x256', 'sketch', 'tx_000000000000')

image_size = (224, 224, 3)

def load_sketch_stats():
    return pd.read_csv(SKETCH_STATS_PATH, index_col=None)


def load_images_paths(df_sketch):
    anchors = []
    positives = []
    
    for index, row in df_sketch.iterrows():
        category_id = row['CategoryID']
        category = row['Category']
        image_id = row['ImageNetID']
        sketch_id = row['SketchID']

        photo_path = os.path.join(
            PHOTO_PATH, 
            category, 
            image_id + '.jpg'
        )
        sketch_path = os.path.join(
            SKETCH_PATH, 
            category, 
            image_id + '-' + str(sketch_id) + '.png'
        )

        anchors.append(photo_path)
        positives.append(sketch_path)

    return anchors, positives


def load_and_preprocess_image(file_path):
    if isinstance(file_path, bytes):
        file_path = file_path.decode("utf-8")

    file_extension = file_path.split('.')[-1]

    file_path = file_path.replace(" ", "_")

    if file_extension not in ['jpg', 'png']:
        raise ValueError('Invalid file extension')
    
    try:
        image = tf.io.read_file(file_path)
    except:
        print('Error reading file: {}'.format(file_path))
        return None

    if tf.equal(file_extension, 'jpg'):
        image = tf.image.decode_jpeg(image, channels=3)
    elif tf.equal(file_extension, 'png'):
        image = tf.image.decode_png(image, channels=3)

    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [image_size[0], image_size[1]])

    return image


# Now only generates anchor and positives
def generate_triplets(anchor_path, positive_path, triplet_paths):
    anchor_image = load_and_preprocess_image(anchor_path)
    positive_image = load_and_preprocess_image(positive_path)

    negative_path = None
    negative_image = None

    while negative_image is None:
        negative_path = triplet_paths[random.randint(0, len(triplet_paths) - 1)][0]
        if negative_path != anchor_path and negative_path != positive_path:
            negative_image = load_and_preprocess_image(negative_path)

    return anchor_image, positive_image, negative_image


def triplet_generator(triplet_paths):
    for triplet_path in triplet_paths:
        yield generate_triplets(triplet_path[0], triplet_path[1], triplet_paths)


def tensorflow_dataset(anchor_paths, positive_paths, batch_size=128):
    triplet_paths = list(zip(anchor_paths, positive_paths))

    dataset = tf.data.Dataset.from_generator(
        triplet_generator,
        args=[triplet_paths],
        output_types=(tf.float32, tf.float32, tf.float32),
        output_shapes=(image_size, image_size, image_size)
    )

    dataset = dataset.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def split_dataset(dataset, batch_size = 128,train_size=0.8, shuffle=True):
    dataset_list = list(dataset)

    train_triplets, val_triplets = train_test_split(dataset_list, test_size=0.2, random_state=42)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_triplets)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_triplets)

    train_dataset = train_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, val_dataset


def visualize_triplets(dataset, num_triplets=3):
    for anchor, positive, negative in dataset.take(num_triplets):
        plt.figure(figsize=(6, 2))
        plt.subplot(1, 3, 1)
        plt.imshow(anchor[0])
        plt.title("Anchor")

        plt.subplot(1, 3, 2)
        plt.imshow(positive[0])
        plt.title("Positive")

        plt.subplot(1, 3, 3)
        plt.imshow(negative[0])
        plt.title("Negative")

        plt.show()


# Load and preprocess the SVG file
def preprocess_svg(svg_file_path, target_size=(224, 224)):
    # Load the SVG file using a library like lxml or svgwrite.
    # Extract the path data and convert it to an image.
    # For simplicity, let's assume you have an SVG to PNG conversion function.
    # Here's an example:
    # svg_to_png(svg_file_path, output_png_path)

    # svg_file_path = os.path.join(SCRIPT_PATH, svg_file_path)

    # Load the PNG image
    

    try:
        # Convert SVG to PNG using cairosvg
        svg_data = open(svg_file_path, 'rb').read()
        png_data = cairosvg.svg2png(bytestring=svg_data)

        # Open the PNG image with Pillow
        img = Image.open(io.BytesIO(png_data))
        img = img.convert("RGB")
        # img = Image.open(svg_file_path)
    except:
        print('Error reading file: {}'.format(svg_file_path))
        return None
    
    # Resize and preprocess the image
    # img = img.resize(target_size)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.image.resize(img, target_size)
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    img = tf.image.convert_image_dtype(img, tf.float32)
    # img = tf.keras.applications.resnet50.preprocess_input(img)  # Preprocess for ResNet model
    return img

def main():
    df_sketch = load_sketch_stats()
    anchors, positives = load_images_paths(df_sketch)

    dataset = tensorflow_dataset(anchors, positives)

    visualize_triplets(dataset)


if __name__ == '__main__':
    main()
