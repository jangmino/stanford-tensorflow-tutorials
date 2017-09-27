import tensorflow as tf

from layers import *

def encoder(input):
    # Create a conv network with 3 conv layers and 1 FC layer
    # Conv 1: filter: [3, 3, 1], stride: [2, 2], relu
    conv1 = conv(input, "conv1", filter_dims=[3,3,1], stride_dims=[2,2])
    
    # Conv 2: filter: [3, 3, 8], stride: [2, 2], relu
    conv2 = conv(conv1, "conv2", filter_dims=[3,3,8], stride_dims=[2,2])

    # Conv 3: filter: [3, 3, 8], stride: [2, 2], relu
    conv3 = conv(conv2, "conv3", filter_dims=[3,3,8], stride_dims=[2,2])

    # FC: output_dim: 100, no non-linearity
    enc_out = fc(conv3, "enc_fc", 100, non_linear_fn=None)

    return enc_out

def decoder(input):
    # Create a deconv network with 1 FC layer and 3 deconv layers
    # FC: output dim: 128, relu
    fc0 = fc(input, "dec_fc", 128)
    
    # Reshape to [batch_size, 4, 4, 8]
    feature = tf.reshape(fc0, shape=[-1,4,4,8])
    
    # Deconv 1: filter: [3, 3, 8], stride: [2, 2], relu
    deconv1 = deconv(feature, name="deconv1", filter_dims=[3,3,8], stride_dims=[2,2])
    
    # Deconv 2: filter: [8, 8, 1], stride: [2, 2], padding: valid, relu
    deconv2 = deconv(deconv1, name="deconv2", filter_dims=[8,8,1], stride_dims=[2,2], padding='VALID')

    # Deconv 3: filter: [7, 7, 1], stride: [1, 1], padding: valid, sigmoid
    dec_out = deconv(deconv2, name="deconv3", filter_dims=[7,7,1], stride_dims=[1,1], padding='VALID', non_linear_fn=tf.nn.sigmoid)
    return dec_out

def autoencoder(input_shape):
    # Define place holder with input shape
    input_image= tf.placeholder(tf.float32, input_shape, name='input_image')

    # Define variable scope for autoencoder
    with tf.variable_scope('autoencoder') as scope:
        # Pass input to encoder to obtain encoding
        enc_out = encoder(input_image)
        
        # Pass encoding into decoder to obtain reconstructed image
        reconstructed_image = decoder(enc_out)
        
        # Return input image (placeholder) and reconstructed image
        return input_image, reconstructed_image
