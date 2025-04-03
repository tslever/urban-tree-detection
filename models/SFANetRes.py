import tensorflow as tf
from tensorflow.keras import Model, layers
# Import our custom ResNet backbone.
from .ResNet import ResNet
# Reuse the BaseConv layer from the VGG-based implementation.
from .VGG import BaseConv

# Optionally, if you wish to perform channel reduction in case your input has more than 3 channels:
class ChannelReduction(Model):
    def __init__(self):
        super(ChannelReduction, self).__init__()
        self.conv = layers.Conv2D(3, kernel_size=1, strides=1, padding='same',
                                  kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))
    def call(self, inputs):
        return self.conv(inputs)

class BackEnd(Model):
    def __init__(self, half_res=False):
        super(BackEnd, self).__init__()
        self.half_res = half_res
        self.upsample = layers.UpSampling2D(2, interpolation='bilinear')
        self.conv1 = BaseConv(256, 1, 1, activation='relu', use_bn=True)
        self.conv2 = BaseConv(256, 3, 1, activation='relu', use_bn=True)
        self.conv3 = BaseConv(128, 1, 1, activation='relu', use_bn=True)
        self.conv4 = BaseConv(128, 3, 1, activation='relu', use_bn=True)
        self.conv5 = BaseConv(64, 1, 1, activation='relu', use_bn=True)
        self.conv6 = BaseConv(64, 3, 1, activation='relu', use_bn=True)
        self.conv7 = BaseConv(32, 3, 1, activation='relu', use_bn=True)
        if not self.half_res:
            self.conv8 = BaseConv(32, 1, 1, activation='relu', use_bn=True)
            self.conv9 = BaseConv(32, 3, 1, activation='relu', use_bn=True)
            self.conv10 = BaseConv(32, 3, 1, activation='relu', use_bn=True)
    
    def call(self, inputs):
        # If half_res is True, we expect 4 feature maps; otherwise, 5.
        if self.half_res:
            f1, f2, f3, f4 = inputs
        else:
            f0, f1, f2, f3, f4 = inputs
        
        x = self.upsample(f4)
        x = tf.concat([x, f3], axis=-1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        
        x = tf.concat([x, f2], axis=-1)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.upsample(x)
        
        x = tf.concat([x, f1], axis=-1)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        
        if not self.half_res:
            x = self.upsample(x)
            x = tf.concat([x, f0], axis=-1)
            x = self.conv8(x)
            x = self.conv9(x)
            x = self.conv10(x)
        
        return x

class SFANetRes(Model):
    def __init__(self, half_res=True):
        super(SFANetRes, self).__init__()
        # Choose the ResNet50 output layers corresponding to multiple scales.
        output_layer_names = ['conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']
        # Instantiate our ResNet backbone.
        self.resnet = ResNet(output_layer_names=output_layer_names)
        # Instantiate the two decoding branches (attention and confidence branches) using BackEnd.
        self.amp = BackEnd(half_res=half_res)
        self.dmp = BackEnd(half_res=half_res)
        # Define 1x1 convolutions for attention and final output.
        self.conv_att = BaseConv(1, 1, 1, activation='sigmoid', use_bn=True)
        self.conv_out = BaseConv(1, 1, 1, activation=None, use_bn=False)
        # Optional: if the preprocessed input has more than 3 channels, reduce it.
        self.channel_reduction = ChannelReduction()

    def call(self, inputs):
        # First, reduce input channels from 5 to 3 if needed.
        x = self.channel_reduction(inputs)
        # Extract multi-scale feature maps from the ResNet backbone.
        features = self.resnet(x)
        # Pass the features through both decoder branches.
        amp_out = self.amp(features)
        dmp_out = self.dmp(features)
        # Compute an attention map.
        amp_out = self.conv_att(amp_out)
        # Multiply the attention map with the confidence branch’s output.
        dmp_out = amp_out * dmp_out
        # Compute the final confidence map.
        dmp_out = self.conv_out(dmp_out)

        # Add this line to resize `dmp_out` to match the required size (256, 256)
        dmp_out = tf.image.resize(dmp_out, (256, 256), method='bilinear')

        return dmp_out, amp_out

def build_model(input_shape, preprocess_fn=None, bce_loss_weight=0.1, half_res=True):
    image = layers.Input(input_shape)
    image_preprocessed = layers.Lambda(lambda x: preprocess_fn(x))(image)
    
    # If needed, reduce channels to 3 for ResNet
    if image_preprocessed.shape[-1] != 3:
        x = layers.Conv2D(3, (1, 1), padding='same', name='channel_reduction')(image_preprocessed)
    else:
        x = image_preprocessed
    
    # Build our SFANetRes model
    sfanet = SFANetRes(half_res=half_res)
    dmp, amp = sfanet(x)
    
    # Ensure dmp and amp are resized to (256, 256)
    dmp_out = layers.Lambda(lambda x: tf.image.resize(x, (256, 256), method='bilinear'), name='dmp_resize')(dmp)
    amp_out = layers.Lambda(lambda x: tf.image.resize(x, (256, 256), method='bilinear'), name='amp_resize')(amp)

    # Squeeze out the last dimension
    dmp_squeezed = layers.Lambda(lambda x: tf.squeeze(x, axis=-1), name='dmp_squeeze')(dmp_out)
    amp_squeezed = layers.Lambda(lambda x: tf.squeeze(x, axis=-1), name='amp_squeeze')(amp_out)
    
    
    # Build training and testing models
    training_model = tf.keras.Model(inputs=image, outputs=[dmp_squeezed, amp_squeezed])
    testing_model = tf.keras.Model(inputs=image, outputs=dmp_squeezed)
    return training_model, testing_model
