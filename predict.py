from models.siamese_alt import Siamese
from datasets.load_datasets import load_sketch_stats, load_images_paths, tensorflow_dataset, visualize_triplets, split_dataset, preprocess_svg
import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt

svg_test_path = "/media/jaimefdz/3DC6172F5E0FA9A6/202302/deep_learning/tarea_2/datasets/sketches-06-04/sketches/airplane/n02691156_10151-1.svg"


def show_epoch_history_from_file(file_path = 'model_history.npy'):
    data = np.load(file_path, allow_pickle=True)
    print(data)
    plt.plot(data.item().get('loss'))
    plt.plot(data.item().get('dist_pos'))
    plt.plot(data.item().get('dist_neg'))
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'dist_pos', 'dist_neg'], loc='upper left')
    plt.show()


def show_embedding(embedding_path = 'embeddings.npy'):
    embeddings = np.load(embedding_path)
    print(embeddings.shape)
    plt.imshow(embeddings[0])
    plt.show()


def show_images(image_list):
    for image in image_list:
        plt.imshow(image)
        plt.show()

def load_keras_model():
    try:
        model = tf.keras.models.load_model('siamese_model.keras')
        print('model loaded')
        return model
    except Exception as e:
        print("Error loading model")
        print(e)
        return None


def load_h5_weights_model():
    model = Siamese()
    model.build(input_shape=(None, 224, 224, 3))
    model.load_weights('siamese_model_weights.h5')
    print('model loaded')
    return model


def load_tf_model():
    model = tf.keras.models.load_model('siamese_model')
    print('model loaded')
    return model


def load_dataset():
    df_sketch = load_sketch_stats()
    anchors, positives = load_images_paths(df_sketch)
    dataset = tensorflow_dataset(anchors, positives)
    return dataset


def find_similar_images(query_prediction, embeddings, dataset):
    similarities = cosine_similarity(query_prediction, embeddings)

    # Rank images based on similarity scores
    sorted_indices = np.argsort(similarities[0])  # Sort in ascending order

    # Retrieve top N similar images (e.g., top 5)
    top_n = 5
    top_n_indices = sorted_indices[:top_n]

    print(top_n_indices)

    # Now 'top_n_indices' contains the indices of the most similar images in your dataset
    # You can use these indices to retrieve the corresponding images from your dataset
    similar_images = []

    for i in top_n_indices:
        dataset_tensors = dataset.take(i)

        # there is a tuple with 3 tensors, we only need the first one
        # the data type is _TakeDataset

        anchor_tensor = list(dataset_tensors)[0][0]
        print(anchor_tensor)
        image = tf.keras.preprocessing.image.array_to_img(anchor_tensor)
        similar_images.append(image)

    return similar_images


def make_prediction(model, svg_image):
    dataset = load_dataset()

    visualize_triplets(dataset)

    # embeddings = create_embedding(model, dataset)
    embeddings = np.load('embeddings.npy')
    #print(embeddings)

    prediction = model.predict(svg_image)
    prediction = prediction.reshape(1, -1)
    # print(prediction)

    similar_images = find_similar_images(prediction, embeddings, dataset)
    print(similar_images)

    show_images(similar_images)


def create_embedding(model, dataset):
    embeddings = []
    counter = 0

    # iterate through all the images in the dataset
    for anchor, positive, negative in dataset:
        anchor_embedding = model(anchor)
        embeddings.append(anchor_embedding[0])
        counter += 1

    print(counter)

    # Convert the list of embeddings to a NumPy array
    embeddings_array = np.array(embeddings)

    # Save the embeddings to a file
    np.save('embeddings.npy', embeddings_array)

    return embeddings


def main():
    model = load_h5_weights_model()
    svg_image = preprocess_svg(svg_test_path)

    make_prediction(model, svg_image)


if __name__ == '__main__':
    main()