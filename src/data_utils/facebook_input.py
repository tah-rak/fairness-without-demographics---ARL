# facebook_input.py

import json
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator
from tensorflow.contrib import lookup as contrib_lookup

IPS_WITH_LABEL_TARGET_COLUMN_NAME = "IPS_example_weights_with_label"
IPS_WITHOUT_LABEL_TARGET_COLUMN_NAME = "IPS_example_weights_without_label"
SUBGROUP_TARGET_COLUMN_NAME = "subgroup"

class FacebookInput():
  """Data reader for Facebook dataset."""

  def __init__(self, dataset_base_dir, train_file=None, test_file=None):
    self._dataset_base_dir = dataset_base_dir

    self._train_file = train_file or [f"{dataset_base_dir}/facebook_train.csv"]
    self._test_file = test_file or [f"{dataset_base_dir}/facebook_test.csv"]

    self._mean_std_file = f"{dataset_base_dir}/mean_std.json"
    self._vocabulary_file = f"{dataset_base_dir}/vocabulary.json"
    self._ips_with_label_file = f"{dataset_base_dir}/IPS_example_weights_with_label.json"
    self._ips_without_label_file = f"{dataset_base_dir}/IPS_example_weights_without_label.json"

    self.feature_names = [
        "userid", "age", "gender", "tenure", "friend_count",
        "friendships_initiated", "likes", "likes_received",
        "mobile_likes", "www_likes", "engagement_level"
    ]

    self.RECORD_DEFAULTS = [[0.0], [0.0], ["?"], [0.0], [0.0],
                            [0.0], [0.0], [0.0], [0.0], [0.0], ["?"]]

    self.target_column_name = "engagement_level"
    self.target_column_positive_value = "high"
    self.sensitive_column_names = ["gender"]
    self.sensitive_column_values = ["Female"]
    self.weight_column_name = "instance_weight"

  def get_input_fn(self, mode, batch_size=128):
    def _input_fn():
      if mode == tf_estimator.ModeKeys.TRAIN:
        filename_queue = tf.train.string_input_producer(self._train_file)
      else:
        filename_queue = tf.train.string_input_producer(self._test_file)

      features, targets = self.extract_features_and_targets(filename_queue, batch_size)
      targets = self._add_subgroups_to_targets(features, targets)
      targets = self._add_ips_example_weights_to_targets(targets)
      features[self.weight_column_name] = tf.ones_like(targets[self.target_column_name], dtype=tf.float32)

      return features, targets

    return _input_fn

  def extract_features_and_targets(self, filename_queue, batch_size):
    reader = tf.TextLineReader()
    _, value = reader.read(filename_queue)
    feature_list = tf.decode_csv(value, record_defaults=self.RECORD_DEFAULTS)
    features = dict(zip(self.feature_names, feature_list))

    features = self._binarize_protected_features(features)
    features = tf.train.batch(features, batch_size)

    targets = {}
    targets[self.target_column_name] = tf.reshape(
        tf.cast(tf.equal(features.pop(self.target_column_name),
                         self.target_column_positive_value), tf.float32), [-1, 1])
    return features, targets

  def _binarize_protected_features(self, features):
    for name, value in zip(self.sensitive_column_names, self.sensitive_column_values):
      features[name] = tf.cast(tf.equal(features.pop(name), value), tf.float32)
    return features

  def _add_subgroups_to_targets(self, features, targets):
    for name in self.sensitive_column_names:
      targets[name] = tf.reshape(tf.identity(features[name]), [-1, 1])

        # ✅ New: Define explicit subgroup ID for ARL logging
        # Subgroup = 2 * (label == high) + (gender == Female)
      label = tf.reshape(targets[self.target_column_name], [-1])
      gender = tf.reshape(targets["gender"], [-1])
      subgroup = 2 * label + gender
      targets["subgroup"] = tf.reshape(subgroup, [-1, 1])
    return targets

  def _load_json_dict_into_hashtable(self, filename):
    with tf.gfile.Open(filename, "r") as f:
      temp_dict = json.load(f, object_hook=lambda d:
                            {int(k) if k.isdigit() else k: v for k, v in d.items()})
    keys = list(temp_dict.keys())
    values = [temp_dict[k] for k in keys]
    return contrib_lookup.HashTable(
      contrib_lookup.KeyValueTensorInitializer(keys, values, tf.int64, tf.float32), -1)

  def _add_ips_example_weights_to_targets(self, targets):
    subgroups = (targets[self.target_column_name],
                 targets[self.sensitive_column_names[0]])

    targets[SUBGROUP_TARGET_COLUMN_NAME] = tf.map_fn(
        lambda x: (2 * x[1]), subgroups, dtype=tf.float32)

    ips_with_label_table = self._load_json_dict_into_hashtable(self._ips_with_label_file)
    ips_without_label_table = self._load_json_dict_into_hashtable(self._ips_without_label_file)

    targets[IPS_WITH_LABEL_TARGET_COLUMN_NAME] = tf.map_fn(
        lambda x: ips_with_label_table.lookup(
            tf.cast((4 * x[0]) + (2 * x[1]), tf.int64)), subgroups, tf.float32)

    targets[IPS_WITHOUT_LABEL_TARGET_COLUMN_NAME] = tf.map_fn(
        lambda x: ips_without_label_table.lookup(
            tf.cast((2 * x[1]), tf.int64)), subgroups, tf.float32)

    return targets

  def get_feature_columns(self, embedding_dimension=0, include_sensitive_columns=True):
    with tf.gfile.Open(self._mean_std_file, "r") as f:
      mean_std_dict = json.load(f)
    with tf.gfile.Open(self._vocabulary_file, "r") as f:
      vocab_dict = json.load(f)

    feature_columns = []
    for i in range(len(self.feature_names)):
      name = self.feature_names[i]
      if name in [self.weight_column_name, self.target_column_name]:
        continue
      elif name in self.sensitive_column_names:
        if include_sensitive_columns:
          feature_columns.append(tf.feature_column.numeric_column(name))
        continue
      elif self.RECORD_DEFAULTS[i][0] == "?":
        vocab = vocab_dict[name]
        cat_col = tf.feature_column.categorical_column_with_vocabulary_list(name, vocab)
        if embedding_dimension > 0:
          feature_columns.append(tf.feature_column.embedding_column(cat_col, embedding_dimension))
        else:
          feature_columns.append(tf.feature_column.indicator_column(cat_col))
      else:
        mean, std = mean_std_dict[name]
        feature_columns.append(tf.feature_column.numeric_column(name,
                                   normalizer_fn=(lambda x, m=mean, s=std: (x - m) / s)))

    return feature_columns, self.weight_column_name, self.sensitive_column_names, self.target_column_name
