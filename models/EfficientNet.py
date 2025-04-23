import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetB5

class EfficientNet(Model):
    def __init__(self, output_layer_names):
        """
        Initializes an EfficientNetB0 backbone that outputs a tuple of intermediate feature maps.

        Arguments:
            output_layer_names: A list of layer names from EfficientNetB0 to be used as outputs.
                Example: ['block2a_activation', 'block3a_activation', 'block4a_activation', 'block6a_activation']
        """
        super(EfficientNet, self).__init__()
        base_model = EfficientNetB5(include_top=False, weights='imagenet')
        outputs = [base_model.get_layer(name).output for name in output_layer_names]
        self.model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

    def call(self, inputs):
        return self.model(inputs)
