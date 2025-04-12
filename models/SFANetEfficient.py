import tensorflow as tf
from tensorflow.keras import Model, layers
from .EfficientNet import EfficientNet
from .VGG import BaseConv  # Reusing your BaseConv layer

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

class SFANetEfficient(Model):
    def __init__(self, half_res=True):
        super(SFANetEfficient, self).__init__()
        output_layer_names = ['block2a_activation', 'block3a_activation', 'block4a_activation', 'block6a_activation']
        self.efficientnet = EfficientNet(output_layer_names=output_layer_names)
        self.amp = BackEnd(half_res=half_res)
        self.dmp = BackEnd(half_res=half_res)
        self.conv_att = BaseConv(1, 1, 1, activation='sigmoid', use_bn=True)
        self.conv_out = BaseConv(1, 1, 1, activation=None, use_bn=False)
        self.channel_reduction = ChannelReduction()

    def call(self, inputs):
        x = self.channel_reduction(inputs)
        features = self.efficientnet(x)
        amp_out = self.amp(features)
        dmp_out = self.dmp(features)
        amp_out = self.conv_att(amp_out)
        dmp_out = amp_out * dmp_out
        dmp_out = self.conv_out(dmp_out)
        return dmp_out, amp_out

def build_model(input_shape, preprocess_fn=None, bce_loss_weight=0.1, half_res=True):
    image = layers.Input(input_shape)
    image_preprocessed = layers.Lambda(lambda x: preprocess_fn(x))(image)

    if image_preprocessed.shape[-1] != 3:
        x = layers.Conv2D(3, (1, 1), padding='same', name='channel_reduction')(image_preprocessed)
    else:
        x = image_preprocessed

    sfanet = SFANetEfficient(half_res=half_res)
    dmp, amp = sfanet(x)

    dmp_upsampled = layers.UpSampling2D(size=(4, 4), interpolation='bilinear', name='dmp_upsampling')(dmp)
    amp_upsampled = layers.UpSampling2D(size=(4, 4), interpolation='bilinear', name='amp_upsampling')(amp)

    training_model = tf.keras.Model(inputs=image, outputs=[dmp_upsampled, amp_upsampled])
    testing_model = tf.keras.Model(inputs=image, outputs=dmp_upsampled)
    return training_model, testing_model
