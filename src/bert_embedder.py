from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import numpy as np
import tensorflow as tf
from bert import modeling
from bert import tokenization
import logging
logging.getLogger("tf").setLevel(logging.ERROR)

BERT_PRETRAINED_DIR = '../models/data/rubert_cased_L-12_H-768_A-12_v2'
TPU_ADDRESS = ''
LAYERS = [-1, -2]
NUM_TPU_CORES = 8
MAX_SEQ_LENGTH = 128
BERT_CONFIG = BERT_PRETRAINED_DIR + '/bert_config.json'
CHKPT_DIR = BERT_PRETRAINED_DIR + '/bert_model.ckpt.data-00000-of-00001'
VOCAB_FILE = BERT_PRETRAINED_DIR + '/vocab.txt'
INIT_CHECKPOINT = BERT_PRETRAINED_DIR + '/bert_model.ckpt.index'
BATCH_SIZE = 128


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b=None):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def input_fn_builder(features, seq_length):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_unique_ids = []
    all_input_ids = []
    all_input_mask = []
    all_input_type_ids = []

    for feature in features:
        all_unique_ids.append(feature.unique_id)
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_input_type_ids.append(feature.input_type_ids)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "unique_ids":
                tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_type_ids":
                tf.constant(
                    all_input_type_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
        })

        d = d.batch(batch_size=batch_size, drop_remainder=False)
        return d

    return input_fn


def model_fn_builder(bert_config, init_checkpoint, layer_indexes, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        input_type_ids = features["input_type_ids"]

        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=input_type_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        if mode != tf.estimator.ModeKeys.PREDICT:
            raise ValueError("Only PREDICT modes are supported: %s" % mode)

        tvars = tf.trainable_variables()
        scaffold_fn = None
        (assignment_map,
         initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
            tvars, init_checkpoint)
        if use_tpu:

            def tpu_scaffold():
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                return tf.train.Scaffold()

            scaffold_fn = tpu_scaffold
        else:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        all_layers = model.get_all_encoder_layers()

        predictions = {
            "unique_id": unique_ids,
        }

        for (i, layer_index) in enumerate(layer_indexes):
            predictions["layer_output_%d" % i] = all_layers[layer_index]

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("unique_id: %s" % example.unique_id)
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_sequence(input_sentences):
    examples = []
    unique_id = 0
    for sentence in input_sentences:
        line = tokenization.convert_to_unicode(sentence)
        examples.append(InputExample(unique_id=unique_id, text_a=line))
        unique_id += 1
    return examples


def get_features(input_text, dim=768):
    layer_indexes = LAYERS

    bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=VOCAB_FILE, do_lower_case=True)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        tpu_config=tf.contrib.tpu.TPUConfig(
            num_shards=NUM_TPU_CORES,
            per_host_input_for_training=is_per_host))

    examples = read_sequence(input_text)

    features = convert_examples_to_features(
        examples=examples, seq_length=MAX_SEQ_LENGTH, tokenizer=tokenizer)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=INIT_CHECKPOINT,
        layer_indexes=layer_indexes,
        use_tpu=False,
        use_one_hot_embeddings=True)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        predict_batch_size=BATCH_SIZE,
        train_batch_size=BATCH_SIZE)

    input_fn = input_fn_builder(
        features=features, seq_length=MAX_SEQ_LENGTH)

    # Get features
    # sentence = []
    for result in estimator.predict(input_fn, yield_single_examples=True):
        unique_id = int(result["unique_id"])
        feature = unique_id_to_feature[unique_id]
        output = collections.OrderedDict()
        for (i, token) in enumerate(feature.tokens):
            layers = []
            for (j, layer_index) in enumerate(layer_indexes):
                layer_output = result["layer_output_%d" % j]
                layer_output_flat = np.array([x for x in layer_output[i:(i + 1)].flat])
                layers.append(layer_output_flat)
            # sentence.append(layer_output_flat)
            output[token] = sum(layers)[:dim]
    # output['sentence'] = sum(sentence)[:dim]

    return output


# with tf.Session() as session:
#     embeddings = get_features(["В отношении городского благоустройства Великий Новгород оставлял позади себя многие "
#                                "средневековые города Западной Европы."], dim=50)
#     print(embeddings)
#     print(len(embeddings))
