import tensorflow as tf
import os
import numpy as np

from models.siamese_alt import Siamese
from datasets.load_datasets import load_sketch_stats, load_images_paths, tensorflow_dataset, visualize_triplets, split_dataset

EPOCHS = 1
BATCH_SIZE = 20

image_size = (224, 224, 3)

def main():
    df_sketch = load_sketch_stats()
    anchors, positives = load_images_paths(df_sketch)

    num_samples = len(anchors)
    steps_per_epoch = num_samples // (BATCH_SIZE * 10)

    dataset = tensorflow_dataset(anchors, positives, BATCH_SIZE)
    dataset = dataset.repeat(EPOCHS)

    siamese_model = Siamese()

    siamese_model.compile(
        optimizer=tf.optimizers.Adam(0.0001)
    )

    siamese_model.build(input_shape=image_size)

    siamese_model.summary()
    # siamese_model.embbeding_summary()
    siamese_model.resnet_summary()

    # exit()

    history = siamese_model.fit(
        dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        verbose=1,
        use_multiprocessing=True,
        workers=8,
    )

    history_file = os.path.join('model_history.npy')
    np.save(history_file, history.history)

    try:
        siamese_model.save_weights('siamese_model_weights.h5')
    except:
        print("Error saving model weights")

    try:
        siamese_model.save('siamese_model.h5')
    except:
        print("Error saving model")

if __name__ == '__main__':
    main()