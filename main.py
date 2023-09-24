import tensorflow as tf
import configparser
import os
import numpy as np

from models.siamese import Siamese

from datasets.load_datasets import load_sketch_stats, load_images_paths, tensorflow_dataset

AUTO = tf.data.AUTOTUNE

EPOCHS = 10
BATCH_SIZE = 10

def main():
    df_sketch = load_sketch_stats()
    anchors, positives = load_images_paths(df_sketch)

    dataset = tensorflow_dataset(anchors, positives, BATCH_SIZE)

    print(dataset)

    siamese_model = Siamese()
    siamese_model.compile(optimizer=tf.keras.optimizers.SGD(momentum=0.9))                            
    history = siamese_model.fit(dataset, epochs = EPOCHS)

    history_file = os.path.join('model_history.npy')
    np.save(history_file, history.history)

    #model.save_weights(model_file)
    siamese_model.encoder.save('siamese_model.h5')


if __name__ == '__main__':
    main()