import tensorflow as tf
import os
import numpy as np

from models.siamese_alt import Siamese
from datasets.load_datasets import load_sketch_stats, load_images_paths, tensorflow_dataset, visualize_triplets, split_dataset

EPOCHS = 3
BATCH_SIZE = 30

image_size = (224, 224, 3)

def main():
    df_sketch = load_sketch_stats()
    anchors, positives = load_images_paths(df_sketch)

    num_samples = len(anchors)
    steps_per_epoch = num_samples // BATCH_SIZE

    dataset = tensorflow_dataset(anchors, positives, BATCH_SIZE)
    train_dataset, val_dataset = split_dataset(dataset, BATCH_SIZE)

    siamese_model = Siamese(image_size)

    siamese_model.compile(
        optimizer=tf.optimizers.Adam(0.0001)
    )

    history = siamese_model.fit(
        train_dataset,
        validation_data=val_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        verbose=1,
        use_multiprocessing=True,
        workers=8,
    )

    history_file = os.path.join('model_history.npy')
    np.save(history_file, history.history)

    siamese_model.save_weights('siamese_model_weights.h5')
    siamese_model.save('siamese_model.h5')

if __name__ == '__main__':
    main()