from models.siamese_alt import Siamese
from datasets.load_datasets import load_sketch_stats, load_images_paths, tensorflow_dataset, visualize_triplets, split_dataset, preprocess_svg
import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt

svg_test_path = "/media/jaimefdz/3DC6172F5E0FA9A6/202302/deep_learning/tarea_2/datasets/sketches-06-04/sketches/airplane/n02691156_10151-1.svg"

# png_test_path = '/media/jaimefdz/3DC6172F5E0FA9A6/202302/deep_learning/tarea_2/datasets/rendered_256x256/256x256/sketch/tx_000000000000/axe/n02764044_176-2.png'
png_test_path = '/media/jaimefdz/3DC6172F5E0FA9A6/202302/deep_learning/tarea_2/datasets/rendered_256x256/256x256/sketch/tx_000000000000/airplane/n02691156_58-4.png'
# png_test_path = '/media/jaimefdz/3DC6172F5E0FA9A6/202302/deep_learning/tarea_2/datasets/rendered_256x256/256x256/photo/tx_000000000000/airplane/n02691156_359.jpg'

def load_png_image(path, target_size=(224, 224)):
    image = tf.keras.preprocessing.image.load_img(path, target_size=target_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.resnet.preprocess_input(image)
    image = tf.expand_dims(image, axis=0)
    return image


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


def show_images(image_list, main_image):
    # show the main image and all the similar images in one plot
    n_rows = (len(image_list) // 2) + 1
    n_cols = 2
    index = 1

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(n_rows, n_cols, index)
    ax.imshow(main_image[0])
    ax.set_title('Main image')

    for image in image_list:
        index += 1
        ax = fig.add_subplot(n_rows, n_cols, index)
        ax.imshow(image)
        ax.set_title('Similar image')
    
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
    dataset = tensorflow_dataset(anchors, positives, False, False)
    return dataset


def find_similar_images(query_prediction, embeddings, dataset, top_n = 2):
    similarities = cosine_similarity(query_prediction, embeddings)

    # Rank images based on similarity scores
    similarities_sorted = similarities[0]  # Sort in ascending order

    print(similarities_sorted)

    top_n_indices = {}

    for i in range(len(similarities_sorted)):
        similarities_value = similarities_sorted[i]

        # get the lower value in top_n_indices
        # if similarities_value is greater than the lower value, replace it
        if len(top_n_indices) < top_n:
            top_n_indices[i] = similarities_value
        else:
            for index, value in top_n_indices.items():
                if similarities_value > value:
                    top_n_indices.pop(index)
                    top_n_indices[i] = similarities_value
                    break

    print(top_n_indices)
    # Now 'top_n_indices' contains the indices of the most similar images in your dataset
    # You can use these indices to retrieve the corresponding images from your dataset
    similar_images = []

    for index, value in top_n_indices.items():
        print(index)
        # dataset_tensors = dataset.take(i)
        skipped_dataset = dataset.skip(index)

        # Use take to take the desired element
        extracted_triplet = skipped_dataset.take(1)

        # there is a tuple with 3 tensors, we only need the first one
        # the data type is _TakeDataset

        # print(list(extracted_triplet)[0])

        anchor_tensor = list(extracted_triplet)[0][0]
        # print(anchor_tensor)

        image = tf.keras.preprocessing.image.array_to_img(anchor_tensor)
        similar_images.append(image)
        print(image)

    return similar_images


def make_prediction(model, svg_image):
    dataset = load_dataset()

    # visualize_triplets(dataset)

    embeddings = create_embedding(model, dataset)
    # embeddings = np.load('embeddings.npy')
    #print(embeddings)

    png_image = load_png_image(png_test_path)
    png_prediction = model.predict(png_image)
    png_prediction = png_prediction.reshape(1, -1)
    # print(png_prediction)

    # prediction = model.predict(svg_image)
    # prediction = prediction.reshape(1, -1)
    # print(prediction)

    similar_images = find_similar_images(
        png_prediction, 
        embeddings, 
        dataset,
        top_n = 10
    )
    print(similar_images)

    image = png_image

    show_images(similar_images, image)


def create_embedding(model, dataset):
    embeddings = []
    counter = 0

    # iterate through all the images in the dataset
    for anchor, positive, negative in dataset:
        # chage dimention to (1, 224, 224, 3)
        anchor = tf.expand_dims(anchor, axis=0)
        # anchor = tf.expand_dims(positive, axis=0)

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
    model = load_tf_model()
    svg_image = preprocess_svg(svg_test_path)

    make_prediction(model, svg_image)
    # show_epoch_history_from_file()


if __name__ == '__main__':
    main()