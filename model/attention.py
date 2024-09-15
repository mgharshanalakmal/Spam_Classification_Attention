import keras
import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)


@keras.saving.register_keras_serializable(package="ComplexModels")
class Attention(tf.keras.Model):

    def __init__(self, units, **kwargs):

        super(Attention, self).__init__()
        self.units = units
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["units"] = config["units"]

        return cls(units=config["units"])
