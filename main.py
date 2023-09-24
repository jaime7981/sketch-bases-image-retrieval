import tensorflow as tf
import os
import numpy as np

from models.siamese_alt import Siamese
from datasets.load_datasets import load_sketch_stats, load_images_paths, tensorflow_dataset, visualize_triplets, triplet_generator

EPOCHS = 10
BATCH_SIZE = 250

image_size = (224, 224, 3)

def main():
    df_sketch = load_sketch_stats()
    anchors, positives = load_images_paths(df_sketch)

    triplet_paths = list(zip(anchors, positives))
    num_samples = len(triplet_paths)
    steps_per_epoch = num_samples // BATCH_SIZE

    siamese = Siamese(image_size)

    siamese_model = siamese.get_model()

    siamese_model.compile(
        optimizer=tf.optimizers.Adam(0.0001),
        loss=siamese.triplet_loss
    )

    generator = triplet_generator(triplet_paths, BATCH_SIZE, image_size)

    history = siamese_model.fit(
        generator,
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