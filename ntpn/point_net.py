#!/usr/bin/env python3
"""
Point net model setup and evaluation.

Adapted from Qi et al, (2017) and examples at keras.io

This module implements the PointNet architecture for neural trajectory analysis,
including T-Net transformers, various ablation models, and critical set extraction.

@author: proxy_loken
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from keras import layers
from tensorflow import keras

from ntpn import ntpn_constants
from ntpn.logging_config import get_logger

logger = get_logger(__name__)


def _model_summary_str(model: keras.Model) -> str:
    """Capture model.summary() output as a string."""
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return '\n'.join(lines)


# MODEL BUILDING FUNCTIONS


def conv_bn(x: Any, filters: int) -> Any:
    """
    Convolution layer with batch normalization and ReLU activation.

    Args:
        x: Input tensor
        filters: Number of convolutional filters

    Returns:
        Activated tensor after Conv1D -> BatchNorm -> ReLU
    """
    x = layers.Conv1D(filters, kernel_size=1, padding='valid')(x)
    x = layers.BatchNormalization(momentum=ntpn_constants.BATCH_NORM_MOMENTUM)(x)

    return layers.Activation('relu')(x)


def dense_bn(x: Any, filters: int) -> Any:
    """
    Dense layer with batch normalization and ReLU activation.

    Args:
        x: Input tensor
        filters: Number of dense units

    Returns:
        Activated tensor after Dense -> BatchNorm -> ReLU
    """
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=ntpn_constants.BATCH_NORM_MOMENTUM)(x)

    return layers.Activation('relu')(x)


# Regularizer (this constrains the transformation to be near to orthogonal)
@keras.utils.register_keras_serializable()
class OrthogonalRegularizer(keras.regularizers.Regularizer):
    """
    Orthogonal regularizer for T-Net transformation matrices.

    Constrains learned transformations to be near-orthogonal by penalizing
    the difference between X*X^T and the identity matrix.

    Args:
        num_features: Dimension of the transformation matrix
        l2reg: L2 regularization weight (default: 0.001)
    """

    def __init__(self, num_features: int, l2reg: float = ntpn_constants.L2_REGULARIZATION) -> None:
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x: Any) -> Any:
        """
        Compute regularization loss.

        Args:
            x: Transformation matrix tensor

        Returns:
            Scalar regularization loss
        """
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

    def get_config(self) -> dict[str, Any]:
        # config=super().get_config()
        # config.update(
        #    {
        #        'num_features' : self.num_features(),
        #        'l2reg' : self.l2reg(),
        #        'eye' : self.eye()
        #    }
        # )

        return {'num_features': self.num_features, 'l2reg': self.l2reg, 'eye': self.eye}


def tnet(inputs: Any, num_features: int, units: int = 32) -> Any:
    """
    Transformer network (T-Net) for feature alignment.

    Learns a transformation matrix to align input features, making the network
    invariant to rotations and translations.

    Args:
        inputs: Input tensor
        num_features: Dimension of features to transform
        units: Base number of units for layers (default: 32)

    Returns:
        Transformed features after applying learned transformation
    """
    # initialize bias as the identity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, units)
    x = conv_bn(x, units * 2)
    x = conv_bn(x, units * 16)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, units * 8)
    x = dense_bn(x, units * 4)
    x = layers.Dense(
        num_features * num_features, kernel_initializer='zeros', bias_initializer=bias, activity_regularizer=reg
    )(x)

    feat_T = layers.Reshape((num_features, num_features))(x)
    # apply affine transform to the input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])


def point_net(num_points: int, num_classes: int, units: int = 32, dims: int = 1) -> keras.Model:
    """
    Main PointNet model for neural trajectory classification.

    Implements PointNet with two T-Net transformers for permutation and
    transformation invariance.

    Args:
        num_points: Number of points (neurons) in each trajectory
        num_classes: Number of output classes
        units: Base number of convolutional filters (default: 32)
        dims: Input feature dimension per point (default: 1)

    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=(num_points, dims))

    x = tnet(inputs, dims, units)
    x = conv_bn(x, units)
    x = conv_bn(x, units)
    x = tnet(x, units, units)
    x = conv_bn(x, units)
    x = conv_bn(x, units * 2)
    x = conv_bn(x, units * 16)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, units * 8)
    x = layers.Dropout(ntpn_constants.DROPOUT_RATE)(x)
    x = dense_bn(x, units * 4)
    x = layers.Dropout(ntpn_constants.DROPOUT_RATE)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='neural_trajectory_point_net')

    return model


# ABLATION versions of Point Net
def point_net_no_transform(num_points: int, num_classes: int, units: int = 32, dims: int = 1) -> keras.Model:
    """
    PointNet ablation: no T-Net transformers.

    Args:
        num_points: Number of points (neurons) in each trajectory
        num_classes: Number of output classes
        units: Base number of convolutional filters (default: 32)
        dims: Input feature dimension per point (default: 1)

    Returns:
        Keras model without transformation networks
    """

    inputs = keras.Input(shape=(num_points, dims))

    x = conv_bn(inputs, units)
    x = conv_bn(x, units)
    x = conv_bn(x, units)
    x = conv_bn(x, units)
    x = conv_bn(x, units * 2)
    x = conv_bn(x, units * 16)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, units * 8)
    x = layers.Dropout(ntpn_constants.DROPOUT_RATE)(x)
    x = dense_bn(x, units * 4)
    x = layers.Dropout(ntpn_constants.DROPOUT_RATE)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='pointnetnotransform')
    logger.debug('Model summary for %s:\n%s', model.name, _model_summary_str(model))

    return model


def point_net_no_pool(num_points: int, num_classes: int, units: int = 32, dims: int = 1) -> keras.Model:

    inputs = keras.Input(shape=(num_points, dims))

    x = tnet(inputs, dims, units)
    x = conv_bn(inputs, units)
    x = conv_bn(x, units)
    x = tnet(x, units, units)
    x = conv_bn(x, units)
    x = conv_bn(x, units)
    x = conv_bn(x, units * 2)
    x = conv_bn(x, units * 16)
    x = layers.Flatten()(x)
    x = dense_bn(x, units * 8)
    x = layers.Dropout(ntpn_constants.DROPOUT_RATE)(x)
    x = dense_bn(x, units * 4)
    x = layers.Dropout(ntpn_constants.DROPOUT_RATE)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='pointnetnopool')
    logger.debug('Model summary for %s:\n%s', model.name, _model_summary_str(model))

    return model


def point_net_no_pool_no_transform(num_points: int, num_classes: int, units: int = 32, dims: int = 1) -> keras.Model:

    inputs = keras.Input(shape=(num_points, dims))

    x = conv_bn(inputs, units)
    x = conv_bn(x, units)
    x = conv_bn(x, units)
    x = conv_bn(x, units)
    x = conv_bn(x, units * 2)
    x = conv_bn(x, units * 16)

    # take global signal either sequentially: RNN, or all at once: flatten
    x = layers.Flatten()(x)
    # cell = layers.SimpleRNNCell(units)
    # x = layers.RNN(cell)(x)

    x = dense_bn(x, units * 8)
    x = layers.Dropout(ntpn_constants.DROPOUT_RATE)(x)
    x = dense_bn(x, units * 4)
    x = layers.Dropout(ntpn_constants.DROPOUT_RATE)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='pointnetnopoolnotransform')
    logger.debug('Model summary for %s:\n%s', model.name, _model_summary_str(model))

    return model


# point net segmentation constructor
def point_net_segment(num_points: int, num_classes: int, units: int = 32, dims: int = 1) -> keras.Model:

    inputs = keras.Input(shape=(num_points, dims))

    transform_1 = tnet(inputs, dims, units)
    features_1 = conv_bn(transform_1, units)
    features_2 = conv_bn(features_1, units * 2)
    features_3 = conv_bn(features_2, units * 2)
    transform_2 = tnet(features_3, units * 2, units)
    features_4 = conv_bn(transform_2, units * 8)
    features_5 = conv_bn(features_4, units * 16)
    global_1 = layers.MaxPool1D(pool_size=num_points)(features_5)

    global_2 = layers.UpSampling1D(size=num_points)(global_1)

    segmentation_input = layers.Concatenate(name='segmentation_input')(
        [features_1, features_2, features_3, transform_2, features_4, global_2]
    )

    segment_1 = conv_bn(segmentation_input, units * 2)

    outputs = layers.Conv1D(num_classes, kernel_size=1, activation='softmax', name='segmentation_head')(segment_1)

    model = keras.Model(inputs=inputs, outputs=outputs, name='pointnetsegment')
    logger.debug('Model summary for %s:\n%s', model.name, _model_summary_str(model))

    return model


# Visualisation Functions


# run critical points
def predict_critical(
    model: keras.Model, samples: npt.NDArray[np.float32], layer_name: str = 'activation_31'
) -> npt.NDArray[np.float32]:

    # grab the last conv layer before the maxpooling
    # layer_name = 'conv1d_21'
    # layer_name = 'activation_31'
    # setup a new model that outputs the predictions at the target layer
    layer = model.get_layer(name=layer_name)
    crit_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)
    # feed in the sample inputs
    crit_preds = crit_extractor.predict(samples)

    return crit_preds


# run upper points
def predict_upper(
    model: keras.Model, samples: npt.NDArray[np.float32], layer_name: str = 'global_max_pooling1d_5'
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:

    # grab the output of the last max pooling layer
    # layer_name = 'global_max_pooling1d_5'
    # setup a new model that outputs the predictions at the target layer
    layer = model.get_layer(name=layer_name)
    upper_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)
    # feed in the sample inputs
    upper_preds = upper_extractor.predict(samples)

    return upper_preds


# generate critical points set, also returns mean dimensionality of the critical set
# INPUTS: crit_preds: predictions of last conv layer on input samples, num_samples: number of samples
# to generate critical points for, samples: samples inputs used to generate predictions
def generate_critical(
    crit_preds: npt.NDArray[np.float32], num_samples: int, samples: npt.NDArray[np.float32]
) -> list[list[npt.NDArray[np.float32]]]:

    # index points that contribute to maxpooling
    cs_index = np.argmax(crit_preds, axis=1)
    # list to hold critical points
    cs = []
    dims = []
    # extract corresponding critical points from samples used to generate predictions
    for i in range(num_samples):
        temp = []
        dims.append(np.size(np.unique(cs_index[i])))
        for index in cs_index[i]:
            temp.append(samples[i][index])
        cs.append(temp)
    # convert to numpy array
    cs = np.array(cs)
    # take mean of dims
    mean_dim = np.mean(dims)

    return cs, mean_dim


# generate upper points set
# INPUTS: max_preds: maxpool layer predictions on samples, max_unit_preds: conv layer predictions
# on unit sphere input, num_samples: number of samples to generate for, samples: samples used to
# generate predictions, unit_sphere: unit sphere points used to generate max output predictions
def generate_upper(
    max_preds: npt.NDArray[np.float32],
    max_unit_preds: npt.NDArray[np.int32],
    num_samples: int,
    samples: npt.NDArray[np.float32],
    unit_sphere: npt.NDArray[np.float32],
) -> list[list[npt.NDArray[np.float32]]]:
    # TODO: Check shapes, reshape if needed
    ups = []
    # extract points that do not change the max of the last conv layer
    for i in range(num_samples):
        temp = []
        # get difference between conv output on unit sphere(max_unit_preds) and maxpool output on samples (max_preds)
        x = max_preds[i] - max_unit_preds
        x = np.min(x, axis=1)
        for j in range(x.shape[0]):
            if x[j] >= 0:
                temp.append(unit_sphere[j])
        ups.append(temp)

    ups = np.array(ups)

    return ups
