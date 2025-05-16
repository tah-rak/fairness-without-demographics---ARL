from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

from tensorflow.contrib import lookup as contrib_lookup

IPS_WITH_LABEL_TARGET_COLUMN_NAME = "IPS_example_weights_with_label"
IPS_WITHOUT_LABEL_TARGET_COLUMN_NAME = "IPS_example_weights_without_label"
SUBGROUP_TARGET_COLUMN_NAME = "subgroup"

class PoliceInput():
    """Data reader for Police dataset."""

    def __init__(self, dataset_base_dir, train_file=None, test_file=None):
        """Data reader for Police dataset.

        Args:
          dataset_base_dir: (string) directory path.
          train_file: string list of training data paths.
          test_file: string list of evaluation data paths.
        """
        self._dataset_base_dir = dataset_base_dir

        if train_file:
            self._train_file = train_file
        else:
            self._train_file = ["{}/train.csv".format(self._dataset_base_dir)]

        if test_file:
            self._test_file = test_file
        else:
            self._test_file = ["{}/test.csv".format(self._dataset_base_dir)]
        
        self._mean_std_file = "{}/mean_std.json".format(self._dataset_base_dir)
        self._vocabulary_file = "{}/vocabulary.json".format(self._dataset_base_dir)
        self._ips_with_label_file = "{}/IPS_example_weights_with_label.json".format(
        self._dataset_base_dir)
        self._ips_without_label_file = "{}/IPS_example_weights_without_label.json".format(self._dataset_base_dir)

        # Define feature and target column names based on the dataset provided.
        self.feature_names = [
        "manner_of_death", "armed", "age", "gender", "race", "city", "state", 
        "signs_of_mental_illness", "threat_level", "flee", "body_camera"]
        self.target_column_name = "manner_of_death"  # Target is now 'manner_of_death'

        # Define the default values for each feature. 
        self.RECORD_DEFAULTS = [
        ["?"], ["?"], [0.0], ["?"], ["?"], ["?"], ["?"], ["?"], ["?"], ["?"], ["?"]]
        self.sensitive_column_names = ["gender", "race"]
        self.sensitive_column_values = ["Female", "Black"]
        self.weight_column_name = None
        # You can modify other attributes as needed, for example, if the target 
        # values are categorical or if you want specific column transformations.

    def get_input_fn(self, mode, batch_size=128):
        """Gets input_fn for the dataset.
        
        Args:
          mode: The execution mode, as defined in tf.estimator.ModeKeys.
          batch_size: Integer specifying batch size.

        Returns:
          An input_fn.
        """
        
        def _input_fn():
            """Input_fn for the dataset."""
            if mode == tf_estimator.ModeKeys.TRAIN:
                filename_queue = tf.train.string_input_producer(self._train_file)
            elif mode == tf_estimator.ModeKeys.EVAL:
                filename_queue = tf.train.string_input_producer(self._test_file)

            # Extract features and targets
            features, targets = self.extract_features_and_targets(
                filename_queue, batch_size)

            # Adds subgroup information to targets (optional)
            targets = self._add_subgroups_to_targets(features, targets)

            # Adds inverse propensity score (IPS) weights to targets (optional)
            targets = self._add_ips_example_weights_to_targets(targets)

            # Adding instance weight (optional, can be used for min-diff approaches)
            features[self.weight_column_name] = tf.ones_like(
                targets[self.target_column_name], dtype=tf.float32)

            return features, targets

        return _input_fn

    def extract_features_and_targets(self, filename_queue, batch_size):
        """Extract features and targets from filename queue."""

        reader = tf.TextLineReader()
        _, value = reader.read(filename_queue)
        feature_list = tf.decode_csv(value, record_defaults=self.RECORD_DEFAULTS)

        # Setting features dictionary.
        features = dict(zip(self.feature_names, feature_list))
        features = self._binarize_protected_features(features)
        features = tf.train.batch(features, batch_size)

        # Setting targets dictionary.
        targets = {}
        targets[self.target_column_name] = tf.reshape(
            tf.cast(
                tf.equal(
                    features.pop(self.target_column_name),
                    "Gunshot"  # Example: Check if manner_of_death is "Gunshot"
                ), tf.float32), [-1, 1])
        return features, targets

    def _binarize_protected_features(self, features):
        """Processes protected features and binarizes them."""
        for sensitive_column_name, sensitive_column_value in zip(
            self.sensitive_column_names, self.sensitive_column_values):
            features[sensitive_column_name] = tf.cast(
                tf.equal(
                    features.pop(sensitive_column_name), sensitive_column_value),
                tf.float32)
        return features

    def _add_subgroups_to_targets(self, features, targets):
        """Adds subgroup information to the targets dictionary."""
        for sensitive_column_name in self.sensitive_column_names:
            targets[sensitive_column_name] = tf.reshape(
                tf.identity(features[sensitive_column_name]), [-1, 1])
        return targets

    def _load_json_dict_into_hashtable(self, filename):
        """Load JSON dictionary into a HashTable."""
        with tf.gfile.Open(filename, "r") as filename:
            temp_dict = json.load(filename)
        temp_dict = {int(k): v for k, v in temp_dict.items()}
        keys = list(temp_dict.keys())
        values = [temp_dict[k] for k in keys]
        feature_names_to_values = contrib_lookup.HashTable(
            contrib_lookup.KeyValueTensorInitializer(
                keys, values, key_dtype=tf.int64, value_dtype=tf.float32), -1)
        return feature_names_to_values

    def _add_ips_example_weights_to_targets(self, targets):
        """Add inverse propensity scores (IPS) weights to the targets."""
        target_subgroups = (targets[self.target_column_name],
                            targets[self.sensitive_column_names[1]],
                            targets[self.sensitive_column_names[0]])

        targets[SUBGROUP_TARGET_COLUMN_NAME] = tf.map_fn(
            lambda x: (2 * x[1]) + (1 * x[2]), target_subgroups, dtype=tf.float32)

        ips_with_label_table = self._load_json_dict_into_hashtable(self._ips_with_label_file)
        ips_without_label_table = self._load_json_dict_into_hashtable(self._ips_without_label_file)

        targets[IPS_WITH_LABEL_TARGET_COLUMN_NAME] = tf.map_fn(
            lambda x: ips_with_label_table.lookup(
                tf.cast((4 * x[0]) + (2 * x[1]) + (1 * x[2]), dtype=tf.int64)),
            target_subgroups,
            dtype=tf.float32)

        targets[IPS_WITHOUT_LABEL_TARGET_COLUMN_NAME] = tf.map_fn(
            lambda x: ips_without_label_table.lookup(
                tf.cast((2 * x[1]) + (1 * x[2]), dtype=tf.int64)),
            target_subgroups,
            dtype=tf.float32)

        return targets

    def get_feature_columns(self, embedding_dimension=0, include_sensitive_columns=True):
        """Extract feature columns for the model."""
        # Load precomputed mean and standard deviation values for features.
        with tf.gfile.Open(self._mean_std_file, "r") as mean_std_file:
            mean_std_dict = json.load(mean_std_file)
        with tf.gfile.Open(self._vocabulary_file, "r") as vocabulary_file:
            vocab_dict = json.load(vocabulary_file)

        feature_columns = []
        for i in range(0, len(self.feature_names)):
            if (self.feature_names[i] in [
                self.weight_column_name, self.target_column_name]):
                continue
            elif self.feature_names[i] in self.sensitive_column_names:
                if include_sensitive_columns:
                    feature_columns.append(
                        tf.feature_column.numeric_column(self.feature_names[i]))
                else:
                    continue
            elif self.RECORD_DEFAULTS[i][0] == "?":
                sparse_column = tf.feature_column.categorical_column_with_vocabulary_list(
                    self.feature_names[i], vocab_dict[self.feature_names[i]])
                if embedding_dimension > 0:
                    feature_columns.append(
                        tf.feature_column.embedding_column(sparse_column, embedding_dimension))
                else:
                    feature_columns.append(
                        tf.feature_column.indicator_column(sparse_column))
            else:
                mean, std = mean_std_dict[self.feature_names[i]]
                feature_columns.append(
                    tf.feature_column.numeric_column(
                        self.feature_names[i],
                        normalizer_fn=(lambda x, m=mean, s=std: (x - m) / s)))
        return feature_columns, self.weight_column_name, self.sensitive_column_names, self.target_column_name
