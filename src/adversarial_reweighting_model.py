# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=dangerous-default-value
"""A custom estimator for adversarial reweighting model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ✅ Full version of adversarial_reweighting_model.py with ARL debugging support

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import metrics as contrib_metrics


class _AdversarialReweightingModel():
  def __init__(
      self,
      feature_columns,
      label_column_name,
      config,
      model_dir,
      primary_hidden_units=[64, 32],
      adversary_hidden_units=[32],
      batch_size=256,
      primary_learning_rate=0.01,
      adversary_learning_rate=0.01,
      optimizer='Adagrad',
      activation=tf.nn.relu,
      adversary_loss_type='ce_loss',
      adversary_include_label=True,
      upweight_positive_instance_only=False,
      pretrain_steps=5000):

    self._feature_columns = feature_columns
    self._primary_learning_rate = primary_learning_rate
    self._adversary_learning_rate = adversary_learning_rate
    self._optimizer = optimizer
    self._model_dir = model_dir
    self._primary_hidden_units = primary_hidden_units
    self._adversary_hidden_units = adversary_hidden_units
    self._config = config
    self._activation = activation
    self._batch_size = batch_size
    self._label_column_name = label_column_name
    self._adversary_include_label = adversary_include_label
    self._adversary_loss_type = adversary_loss_type
    self._pretrain_steps = pretrain_steps
    self._upweight_positive_instance_only = upweight_positive_instance_only

  def _primary_loss(self, labels, logits, example_weights):
    sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits)
    weighted_loss = example_weights * sigmoid_loss
    return tf.reduce_mean(weighted_loss)

  def _get_hinge_loss(self, labels, logits, pos_weights):
    if self._upweight_positive_instance_only:
      hinge_loss = tf.losses.hinge_loss(
          labels=labels, logits=logits, weights=pos_weights, reduction='none')
    else:
      hinge_loss = tf.losses.hinge_loss(labels=labels, logits=logits, reduction='none')
    hinge_loss = tf.maximum(hinge_loss, 0.1)
    return hinge_loss

  def _get_cross_entropy_loss(self, labels, logits):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

  def _adversary_loss(self, labels, logits, pos_weights, example_weights, loss_type):
    if loss_type == 'hinge_loss':
      loss = self._get_hinge_loss(labels, logits, pos_weights)
    else:
      loss = self._get_cross_entropy_loss(labels, logits)
    return -tf.reduce_mean(example_weights * loss)

  def _get_or_create_global_step_var(self):
    return tf.train.get_or_create_global_step()

  def _get_adversary_features_and_feature_columns(self, features, targets):
    adversarial_features = features.copy()
    adversary_feature_columns = self._feature_columns[:]
    if self._adversary_include_label:
      adversary_feature_columns.append(tf.feature_column.numeric_column(self._label_column_name))
      adversarial_features[self._label_column_name] = targets[self._label_column_name]
    return adversarial_features, adversary_feature_columns

  def _compute_example_weights(self, adv_output_layer):
    example_weights = tf.nn.sigmoid(adv_output_layer)
    mean_weight = tf.reduce_mean(example_weights)
    example_weights /= tf.maximum(mean_weight, 1e-4)
    example_weights = tf.ones_like(example_weights) + example_weights
    tf.print("[ARL DEBUG] Example Weights:", example_weights, summarize=10)
    return example_weights

  def _get_model_fn(self):
    def model_fn(features, labels, mode):
      pos_weights = tf.cast(tf.equal(labels[self._label_column_name], 1), tf.float32)
      class_labels = labels[self._label_column_name]
      current_step = self._get_or_create_global_step_var()

      if mode == tf_estimator.ModeKeys.TRAIN:
        tf.print("[ARL DEBUG] Training Step:", current_step)
        if "subgroup" in labels:
          tf.print("[ARL DEBUG] Subgroup IDs:", labels["subgroup"], summarize=10)

      with tf.name_scope('primary_NN'):
        with tf.variable_scope('primary'):
          input_layer = tf.feature_column.input_layer(features, self._feature_columns)
          h1 = tf.layers.Dense(self._primary_hidden_units[0], activation=self._activation)(input_layer)
          h2 = tf.layers.Dense(self._primary_hidden_units[1], activation=self._activation)(h1)
          logits = tf.layers.Dense(1)(h2)
          sigmoid_output = tf.nn.sigmoid(logits, name='sigmoid')
          class_predictions = tf.cast(tf.greater(sigmoid_output, 0.5), tf.float32)

      with tf.name_scope('adversary_NN'):
        with tf.variable_scope('adversary'):
          adv_features, adv_columns = self._get_adversary_features_and_feature_columns(features, labels)
          adv_input = tf.feature_column.input_layer(adv_features, adv_columns)
          adv_h1 = tf.layers.Dense(self._adversary_hidden_units[0])(adv_input)
          adv_output = tf.layers.Dense(1)(adv_h1)
          example_weights = tf.cond(
              tf.greater(current_step, self._pretrain_steps),
              true_fn=lambda: self._compute_example_weights(adv_output),
              false_fn=lambda: tf.ones_like(class_labels))

      tf.summary.histogram('example_weights', example_weights)
      tf.summary.histogram('labels', class_labels)

      primary_loss = self._primary_loss(class_labels, logits, example_weights)
      adversary_loss = self._adversary_loss(class_labels, logits, pos_weights,
                                            example_weights, self._adversary_loss_type)

      if mode == tf_estimator.ModeKeys.TRAIN:
        tf.print("[ARL DEBUG] Losses — Primary:", primary_loss, "Adversary:", adversary_loss)

      predictions = {
        (self._label_column_name, 'class_ids'): tf.reshape(class_predictions, [-1]),
        (self._label_column_name, 'logistic'): tf.reshape(sigmoid_output, [-1]),
        ('example_weights'): tf.reshape(example_weights, [-1])
      }

      metrics = {
        'labels': class_labels,
        'predictions': class_predictions
      }
      prob_metrics = {'labels': class_labels, 'predictions': sigmoid_output}

      if mode == tf_estimator.ModeKeys.EVAL:
        eval_metrics = {
          'accuracy': tf.metrics.accuracy(**metrics),
          'precision': tf.metrics.precision(**metrics),
          'recall': tf.metrics.recall(**metrics),
          'fp': tf.metrics.false_positives(**metrics),
          'fn': tf.metrics.false_negatives(**metrics),
          'tp': tf.metrics.true_positives(**metrics),
          'tn': tf.metrics.true_negatives(**metrics),
          'fpr': contrib_metrics.streaming_false_positive_rate(**metrics),
          'fnr': contrib_metrics.streaming_false_negative_rate(**metrics),
          'auc': tf.metrics.auc(curve='ROC', **prob_metrics),
          'aucpr': tf.metrics.auc(curve='PR', **prob_metrics)
        }
        return tf_estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=primary_loss,
            eval_metric_ops=eval_metrics)

      if mode == tf_estimator.ModeKeys.TRAIN:
        vars_all = tf.trainable_variables()
        vars_primary = [v for v in vars_all if 'primary' in v.name]
        vars_adv = [v for v in vars_all if 'adversary' in v.name]

        train_primary = contrib_layers.optimize_loss(
            loss=primary_loss,
            variables=vars_primary,
            global_step=contrib_framework.get_global_step(),
            learning_rate=self._primary_learning_rate,
            optimizer=self._optimizer)

        train_adv = contrib_layers.optimize_loss(
            loss=adversary_loss,
            variables=vars_adv,
            global_step=contrib_framework.get_global_step(),
            learning_rate=self._adversary_learning_rate,
            optimizer=self._optimizer)

        train_op = tf.cond(
            tf.greater(current_step, self._pretrain_steps),
            true_fn=lambda: tf.group([train_primary, train_adv]),
            false_fn=lambda: tf.group([train_primary]))

        return tf_estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=primary_loss + adversary_loss,
            train_op=train_op)

    return model_fn


class _AdversarialReweightingEstimator(tf_estimator.Estimator):
  def __init__(self, *args, **kwargs):
    self.model = _AdversarialReweightingModel(*args, **kwargs)
    super(_AdversarialReweightingEstimator, self).__init__(
        model_fn=self.model._get_model_fn(),
        model_dir=self.model._model_dir,
        config=self.model._config)

def get_estimator(*args, **kwargs):
  return _AdversarialReweightingEstimator(*args, **kwargs)
