import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50

class ResNet(Model):
    def __init__(self, output_layer_names):
        """
        Initializes a ResNet50 backbone that outputs a tuple of intermediate feature maps.
        
        Arguments:
            output_layer_names: A list of layer names from ResNet50 to be used as outputs.
                For example: ['conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']
        """
        super(ResNet, self).__init__()
        # Load the ResNet50 model with ImageNet weights and without the top classification layers.
        base_model = ResNet50(include_top=False, weights='imagenet')
        # Create a new model that takes the original inputs and outputs the selected intermediate layers.
        outputs = [base_model.get_layer(name).output for name in output_layer_names]
        self.model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    
    def call(self, inputs):
        return self.model(inputs)
    
    def load_pretrained_resnet(self, input_shape):
        # We are using tf.keras.applications.ResNet50 with weights='imagenet',
        # so the weights are already loaded automatically.
        pass