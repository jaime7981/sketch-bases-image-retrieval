import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

class Siamese(tf.keras.Model):
    def __init__(self, input_shape_, alpha=0.2):
        super(Siamese, self).__init__()

        self.input_shape_ = input_shape_
        self.alpha = alpha
        self.siamese_network = self.build_siamese_network()

    def build_siamese_network(self):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape_)
        x = base_model.output
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        output = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)  # L2 normalize embeddings
        model = Model(inputs=base_model.input, outputs=output)
        return model

    def call(self, inputs):
        anchor_input, positive_input, negative_input = inputs

        anchor_embedding = self.siamese_network(anchor_input)
        positive_embedding = self.siamese_network(positive_input)
        negative_embedding = self.siamese_network(negative_input)

        # Calculate distances
        pos_dist = tf.reduce_sum(tf.square(anchor_embedding - positive_embedding), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor_embedding - negative_embedding), axis=-1)

        # Calculate loss
        basic_loss = pos_dist - neg_dist + self.alpha
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))

        return loss

    def triplet_loss(self, y_true, y_pred):
        return y_pred  # Loss is already computed in the call method

    def get_model(self):
        anchor_input = Input(shape=self.input_shape_)
        positive_input = Input(shape=self.input_shape_)
        negative_input = Input(shape=self.input_shape_)

        loss = self.call([anchor_input, positive_input, negative_input])

        siamese_model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=loss)

        return siamese_model
