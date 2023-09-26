import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt

from models.siamese_alt import Siamese
from datasets.load_datasets import load_sketch_stats, load_images_paths, tensorflow_dataset, visualize_triplets, split_dataset, preprocess_svg

svg_test_path = "/media/jaimefdz/3DC6172F5E0FA9A6/202302/deep_learning/tarea_2/datasets/sketches-06-04/sketches/airplane/n02691156_10151-1.svg"

EPOCHS = 1
BATCH_SIZE = 20

image_size = (224, 224, 3)

def main():
    df_sketch = load_sketch_stats()
    anchors, positives = load_images_paths(df_sketch)

    svg_image = preprocess_svg(svg_test_path)

    num_samples = len(anchors)
    steps_per_epoch = num_samples // (BATCH_SIZE * 10)

    dataset = tensorflow_dataset(anchors, positives, BATCH_SIZE)
    dataset = dataset.repeat(EPOCHS)

    siamese_model = Siamese()
    siamese_model.compile(optimizer=tf.keras.optimizers.SGD(momentum=0.9))
    siamese_model.build(input_shape=(None, *image_size))

    siamese_model.summary()
    siamese_model.resnet_summary()

    history = siamese_model.fit(
        dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        use_multiprocessing=True,
        workers=8,
    )

    history_file = os.path.join('model_history.npy')
    np.save(history_file, history.history)

    prediction = siamese_model.predict(svg_image)

    print(prediction)

    plt.imshow(svg_image[0])
    plt.show()

    plt.imshow(prediction)
    plt.show()

    try:
        siamese_model.save_weights('siamese_model_weights.h5')
        print('h5 whights model saved')
    except:
        print("Error saving model weights")

    try:
        siamese_model.save('siamese_model.keras')
        print('keras model saved')
    except Exception as e:
        print("Error saving keras model")
        print(e)

    try:
        siamese_model.save('siamese_model', save_format='tf')
        print('tf model saved')
    except Exception as e:
        print("Error saving tf model")
        print(e)

if __name__ == '__main__':
    main()