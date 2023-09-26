import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import resnet
from tensorflow.keras import layers
import models.resnet as m_resnet


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
    def __init__(self, margin=0.5, weight_decay = 0.0005):
        super().__init__()

        # self.input_shape_ = input_shape_
        self.margin = margin
        self.weight_decay = weight_decay

        self.encoder = None

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.dist_pos_tracker = tf.keras.metrics.Mean(name="dist_pos")
        self.dist_neg_tracker = tf.keras.metrics.Mean(name="dist_neg")

    '''
    def get_encoder(self):
        anchor_input = tf.keras.layers.Input(name="anchor", shape=self.input_shape_)
        positive_input = tf.keras.layers.Input(name="positive", shape=self.input_shape_)
        negative_input = tf.keras.layers.Input(name="negative", shape=self.input_shape_)


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
    '''
    
    def get_encoder(self):       
        inputs = tf.keras.layers.Input(name="image", shape=self.input_shape_)                
        x = inputs
        bkbone = m_resnet.ResNetBackbone(
            [3,4,6,3], 
            [64,128, 256, 512], 
            kernel_regularizer = tf.keras.regularizers.l2(self.weight_decay)
        )        
        x = bkbone(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(self.input_shape_[0])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(self.input_shape_[0])(x)
        
        outputs = tf.math.l2_normalize(x, axis=1)
        
        return tf.keras.Model(inputs, outputs, name="Encoder")

    def call(self, inputs):
        return self.encoder(inputs)

    def build(self, input_shape):
        inputs = tf.keras.layers.Input(name="image", shape=input_shape)                
        x = inputs
        bkbone = m_resnet.ResNetBackbone(
            [3,4,6,3], 
            [64,128, 256, 512], 
            kernel_regularizer = tf.keras.regularizers.l2(self.weight_decay)
        )        
        x = bkbone(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(input_shape[0])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(input_shape[0])(x)
        
        outputs = tf.math.l2_normalize(x, axis=1)

        self.encoder = tf.keras.Model(inputs, outputs, name="Encoder")

    def summary(self):
        return self.encoder.summary()
    
    def encoder_summary(self):
        return self.encoder.get_layer("Encoder").summary()

    def resnet_summary(self):
        return self.encoder.get_layer("res_net_backbone").summary()

    '''
    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss, dist_pos, dist_neg = self.compute_loss(data)

        gradients = tape.gradient(loss, self.encoder.trainable_weights)

        self.optimizer.apply_gradients(
            zip(gradients, self.encoder.trainable_weights)
        )

        self.loss_tracker.update_state(loss)
        self.dist_pos_tracker.update_state(dist_pos)
        self.dist_neg_tracker.update_state(dist_neg)
        
        return {"loss": self.loss_tracker.result(), "dist_pos": self.dist_pos_tracker.result(), "dist_neg": self.dist_neg_tracker.result()}
    '''

    def train_step(self, batch):
        # Unpack the data.
        anchors, positives, negatives = batch
        # select negatives
        '''
        n = tf.shape(anchors)[0]
        pos = tf.range(n)
        perm = tf.random.shuffle(pos)
        perm = tf.where(perm == pos, (perm + 1) % n, perm)
        negatives = tf.gather(anchors, perm)
        '''
        # view(anchors, positives, negatives)
        # training one step
        with tf.GradientTape() as tape:
            xa = self.encoder(anchors)
            xp = self.encoder(positives)
            xn = self.encoder(negatives)            
            loss, dist_pos, dist_neg = self.compute_loss(xa, xp, xn)
        
        # Compute gradients and update the parameters.
        learnable_params = (
            self.encoder.trainable_variables
        )
        gradients = tape.gradient(loss, learnable_params)
        self.optimizer.apply_gradients(zip(gradients, learnable_params))

        # tracking status.        
        self.loss_tracker.update_state(loss)
        self.dist_pos_tracker.update_state(dist_pos)
        self.dist_neg_tracker.update_state(dist_neg)
        
        return {"loss": self.loss_tracker.result(), "dist_pos": self.dist_pos_tracker.result(), "dist_neg": self.dist_neg_tracker.result()}


    def test_step(self, data):
        loss = self.compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result(), "dist_pos": self.dist_pos_tracker.result(), "dist_neg": self.dist_neg_tracker.result()}


    '''
    def compute_loss(self, data):
        ap_distance, an_distance = self.encoder(data)

        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss, ap_distance, an_distance
    '''

    def compute_loss(self, xa, xp, xn):                            
        margin = self.margin
        #dist_pos = tf.sqrt(tf.reduce_sum(tf.math.square(xa - xp), axis = 1))
        #dist_neg = tf.sqrt(tf.reduce_sum(tf.math.square(xa - xn), axis = 1))
        dist_pos  = tf.math.sqrt(2.0 - 2.0*tf.reduce_sum((xa * xp), axis = 1))
        dist_neg  = tf.math.sqrt(2.0 - 2.0*tf.reduce_sum((xa * xn), axis = 1))
        loss = tf.math.maximum(0.0, dist_pos - dist_neg + margin)
                        
        return tf.reduce_mean(loss), tf.reduce_mean(dist_pos), tf.reduce_mean(dist_neg)


    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker, self.dist_pos_tracker, self.dist_neg_tracker]
