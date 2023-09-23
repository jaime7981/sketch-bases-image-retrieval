import os
import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import random
from pathlib import Path
import shlex

# get path of running script
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

SKETCH_STATS_PATH = os.path.join(SCRIPT_PATH, 'info', 'stats.csv')
PHOTO_PATH = os.path.join(SCRIPT_PATH, 'rendered_256x256', '256x256', 'photo', 'tx_000000000000')
SKETCH_PATH = os.path.join(SCRIPT_PATH, 'rendered_256x256', '256x256', 'sketch', 'tx_000000000000')

image_size = (256, 256, 3)

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

    return image


def generate_triplets(anchor_path, positive_path, triplet_paths):
    anchor_image = load_and_preprocess_image(anchor_path)
    positive_image = load_and_preprocess_image(positive_path)
    print(anchor_image)

    negative_path = None
    negative_image = None

    while True:
        negative_path = triplet_paths[random.randint(0, len(triplet_paths) - 1)][0]
        
        negative_image = load_and_preprocess_image(negative_path)

        if negative_path != anchor_path and negative_path != positive_path and negative_path is not None and negative_image is not None:
            break

    return anchor_image, positive_image, negative_image


def triplet_generator(triplet_paths):
    for triplet_path in triplet_paths:
        print(triplet_path)
        yield generate_triplets(triplet_path[0], triplet_path[1], triplet_paths)


def tensorflow_dataset(anchor_paths, positive_paths, batch_size=128):
    triplet_paths = list(zip(anchor_paths, positive_paths))

    dataset = tf.data.Dataset.from_generator(
        triplet_generator,
        args=[triplet_paths],
        output_types=(tf.float32, tf.float32, tf.float32),
        output_shapes=(image_size, image_size, image_size)
    )

    return dataset


def visualize_triplets(dataset, num_triplets=3):
    for triplet in range(num_triplets):
        for anchor, positive, negative in dataset.take(1):
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(anchor)
            ax[1].imshow(positive)
            ax[2].imshow(negative)
            plt.show()


def main():
    df_sketch = load_sketch_stats()
    anchors, positives = load_images_paths(df_sketch)

    dataset = tensorflow_dataset(anchors, positives)

    visualize_triplets(dataset)


if __name__ == '__main__':
    main()
