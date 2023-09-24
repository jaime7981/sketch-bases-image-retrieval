import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import resnet
from tensorflow.keras import layers


class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

class Siamese(tf.keras.Model):
    def __init__(self, input_shape_, margin=0.5):
        super().__init__()

        self.input_shape_ = input_shape_
        self.siamese_network = self.build_siamese_network()
        self.margin = margin

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.dist_pos_tracker = tf.keras.metrics.Mean(name="dist_pos")
        self.dist_neg_tracker = tf.keras.metrics.Mean(name="dist_neg")

    def build_siamese_network(self):
        anchor_input = layers.Input(name="anchor", shape=self.input_shape_)
        positive_input = layers.Input(name="positive", shape=self.input_shape_)
        negative_input = layers.Input(name="negative", shape=self.input_shape_)

        base_model = ResNet50(
            weights='imagenet', 
            include_top=False, 
            input_shape=self.input_shape_
        )

        flatten = layers.Flatten()(base_model.output)
        dense1 = layers.Dense(512, activation="relu")(flatten)
        dense1 = layers.BatchNormalization()(dense1)
        dense2 = layers.Dense(256, activation="relu")(dense1)
        dense2 = layers.BatchNormalization()(dense2)
        output = layers.Dense(256)(dense2)

        embedding = Model(base_model.input, output, name="Embedding")

        trainable = False
        for layer in base_model.layers:
            if layer.name == "conv5_block1_out":
                trainable = True
            layer.trainable = trainable
        
        distances = DistanceLayer()(
            embedding(resnet.preprocess_input(anchor_input)),
            embedding(resnet.preprocess_input(positive_input)),
            embedding(resnet.preprocess_input(negative_input)),
        )

        siamese_model = Model(
            inputs=[anchor_input, positive_input, negative_input], 
            outputs=distances
        )
        return siamese_model

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss, dist_pos, dist_neg = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        self.dist_pos_tracker.update_state(dist_pos)
        self.dist_neg_tracker.update_state(dist_neg)
        
        return {"loss": self.loss_tracker.result(), "dist_pos": self.dist_pos_tracker.result(), "dist_neg": self.dist_neg_tracker.result()}
    
    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss, ap_distance, an_distance

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]
