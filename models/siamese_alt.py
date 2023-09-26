import tensorflow as tf
import models.resnet as m_resnet

class Siamese(tf.keras.Model):
    def __init__(self, input_shape_ = (224, 224, 3), margin=0.5, weight_decay = 0.0005, **kwargs):
        super(Siamese, self).__init__(**kwargs)

        self.input_shape_ = input_shape_
        self.margin = margin
        self.weight_decay = weight_decay

        self.encoder = self.get_encoder()

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.dist_pos_tracker = tf.keras.metrics.Mean(name="dist_pos")
        self.dist_neg_tracker = tf.keras.metrics.Mean(name="dist_neg")

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

    def compute_loss(self, xa, xp, xn):                            
        margin = self.margin
        dist_pos = tf.sqrt(tf.reduce_sum(tf.math.square(xa - xp), axis = 1))
        dist_neg = tf.sqrt(tf.reduce_sum(tf.math.square(xa - xn), axis = 1))
        loss = tf.math.maximum(0.0, dist_pos - dist_neg + margin)
                        
        return tf.reduce_mean(loss), tf.reduce_mean(dist_pos), tf.reduce_mean(dist_neg)

    def train_step(self, batch):
        anchors, positives, negatives = batch
        
        with tf.GradientTape() as tape:
            xa = self.encoder(anchors)
            xp = self.encoder(positives)
            xn = self.encoder(negatives)            
            loss, dist_pos, dist_neg = self.compute_loss(xa, xp, xn)
        
        learnable_params = (
            self.encoder.trainable_variables
        )
        gradients = tape.gradient(loss, learnable_params)
        self.optimizer.apply_gradients(zip(gradients, learnable_params))

        self.loss_tracker.update_state(loss)
        self.dist_pos_tracker.update_state(dist_pos)
        self.dist_neg_tracker.update_state(dist_neg)
        
        return {"loss": self.loss_tracker.result(), "dist_pos": self.dist_pos_tracker.result(), "dist_neg": self.dist_neg_tracker.result()}

'''
    def summary(self):
        return self.encoder.summary()
    
    def encoder_summary(self):
        return self.encoder.get_layer("Encoder").summary()

    def resnet_summary(self):
        return self.encoder.get_layer("res_net_backbone").summary()

    @property
    def metrics(self):
        return [self.loss_tracker, self.dist_pos_tracker, self.dist_neg_tracker]
'''
