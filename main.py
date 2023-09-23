import tensorflow as tf
import configparser
import os
import numpy as np

from models.siamese import Siamese

AUTO = tf.data.AUTOTUNE

EPOCHS = 100
BATCH_SIZE = 128

def map_func(example_serialized):    
#     features_map=tfds.features.FeaturesDict({'image': tfds.features.Image(shape=(None, None, 3)),
#                                               'label': tfds.features.ClassLabel(names=range(100))})
#     features = tf.io.parse_example(example_serialized, features_map)
    image_anchor = example_serialized['image-anchor']
    image_positive = example_serialized['image_positive']    
    image_anchor = tf.image.resize_with_pad(image_anchor, 256, 256)
    image_positive = tf.image.resize_with_pad(image_positive, 256, 256)
    image_anchor = tf.image.random_crop(image_anchor, size = [224, 224, 3])
    image_positive = tf.image.random_crop(image_positive, size = [224, 224, 3])
    image_positive = tf.cast(image_positive, tf.float32)    
    image_anchor = tf.cast(image_anchor, tf.float32)
    return image_anchor, image_positive 


def main():

    # TODO: load dataset
    dataset = tf.data.Dataset.from_tensor_slices((np.random.rand(100, 224, 224, 3), np.random.rand(100, 224, 224, 3)))

    train_dataset = dataset
    train_dataset = (
        train_dataset.shuffle(1024)
        .map(map_func, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO) )

    siamese_model = Siamese()
    siamese_model.compile(optimizer=tf.keras.optimizers.SGD(momentum=0.9))                            
    history = siamese_model.fit(train_dataset, epochs = EPOCHS)

    history_file = os.path.join('model_history.npy')
    np.save(history_file, history.history)

    #model.save_weights(model_file)
    siamese_model.encoder.save('siamese_model.h5')


if __name__ == '__main__':
    main()