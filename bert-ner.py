#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_BERT and zhoukaiyin's model(https://github.com/kyzhouhzau/BERT-NER).
@Author:JunGAO
@20190419
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
import tf_metrics
import pickle
import random
import numpy as np

MODE = 'classify'
DATA_DIR = 'Prodata/Renew_ner20190516.txt'
BERT_CONFIG_FILE = 'checkpoint/bert_config.json'
OUTPUT_DIR = 'output_0516/'+MODE+'/'
INIT_CHECKPOINT = 'checkpoint/bert_model.ckpt'
VOCAB_FILE = 'checkpoint/vocab.txt'


DO_TRAIN = True
DO_EVAL = True
DO_PREDICT = True

DO_LOWER_CASE = True
MAX_SEQ_LENGTH = 512
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 4
PREDICT_BATCH_SIZE = 4
LEARNING_RATE = 5e-5
NUM_TRAIN_EPOCHS = 20
WARMUP_PROPORTION = 0.1
SAVE_CHECKPOINTS_STEPS = 1000
ITERATIONS_PER_LOOP = 1000

USE_TPU = False
MASTER = None
NUM_TPU_CORES = 8
TPU_NAME = None
TPU_ZONE = None
GCP_PROJECT = None

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        # self.label_mask = label_mask


def read_txt_to_examples(input_file, mode=MODE):
    # mode = 'tag' : 标签识别模式
    # mode = 'classify' : 需求分类模式
    with open(input_file, 'r', encoding='utf-8') as f:
        examples = []
        lines = f.readlines()
        for i, line in enumerate(lines):
                if line.startswith('id:'):
                    id = int(line[3:8])
                    texts = []
                    tag_labels = []
                    classify_labels = []
                    if id == 10071:
                        break
                    continue
                if line.startswith('\t\t\n'):
                    if mode == 'tag':
                        example = InputExample(id, texts, tag_labels)
                        examples.append(example)
                    elif mode == 'classify':
                        example = InputExample(id, texts, classify_labels)
                        examples.append(example)
                    continue
                line_list = line.split('\t')
                texts.append(line_list[0])
                if line_list[1] == '':
                    classify_labels.append('O')
                else:
                    classify_labels.append(line_list[1])

                if line_list[2] == '\n':
                    tag_labels.append('O')
                else:
                    tag_labels.append(line_list[2].strip())

        return examples


def write_tokens(tokens, mode):
    if mode == "test":
        path = os.path.join(OUTPUT_DIR, "token_" + mode + ".txt")
        wf = open(path, 'a')
        for token in tokens:
            if token != "**NULL**":
                wf.write(token + '\n')
        wf.close()


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    label_map = {}
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    with open('./output/label2id.pkl', 'wb') as w:
        pickle.dump(label_map, w)
    textlist = example.text
    labellist = example.label
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")
    # tokens = tokenizer.tokenize(example.text)
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        # tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids
    )
    write_tokens(ntokens, mode)
    return feature


def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, mode=None
):
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64)
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d

    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )
    # output_layer : [batch_size, seq_length, hidden_size]
    output_layer = model.get_sequence_output()

    hidden_size = output_layer.shape[-1].value

    output_weight = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer()
    )
    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        output_layer = tf.reshape(output_layer, [-1, hidden_size])
        logits = tf.matmul(output_layer, output_weight, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        #logits : [batch_size*seq_length, num_labels]
        logits = tf.reshape(logits, [-1, MAX_SEQ_LENGTH, num_labels])

        ##########################################################################
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_sum(per_example_loss)
        probabilities = tf.nn.softmax(logits, axis=-1)
        predict = tf.argmax(probabilities, axis=-1)
        return (loss, per_example_loss, logits, predict)
        ##########################################################################


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, train_mode=MODE):

    def get_positive_index(train_mode=MODE):
        if train_mode == 'tag':
            return [2,3]
        elif train_mode == 'classify':
            return [i for i in range(2, 12)]

    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        # label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, predicts) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
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
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits):
                # def metric_fn(label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                precision = tf_metrics.precision(label_ids, predictions, num_labels, get_positive_index(MODE), average="macro")
                recall = tf_metrics.recall(label_ids, predictions, num_labels, get_positive_index(MODE), average="macro")
                f = tf_metrics.f1(label_ids, predictions, num_labels, get_positive_index(MODE), average="macro")
                #
                return {
                    "eval_precision": precision,
                    "eval_recall": recall,
                    "eval_f": f,
                    # "eval_loss": loss,
                }

            eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
            # eval_metrics = (metric_fn, [label_ids, logits])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predicts, scaffold_fn=scaffold_fn
            )
        return output_spec

    return model_fn

def get_label_list(examples):

    uniq_label_list = []
    for example in examples:
        for l in example.label:
            if l not in uniq_label_list:
                uniq_label_list.append(l)

    uniq_label_list.extend(['X', '[CLS]', '[SEP]'])

    return uniq_label_list


# Main Part
def train_eval():
    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG_FILE)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)
    tpu_cluster_resolver = None
    if USE_TPU and TPU_NAME:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            TPU_NAME, zone=TPU_ZONE, project=GCP_PROJECT)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=MASTER,
        model_dir=OUTPUT_DIR,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=ITERATIONS_PER_LOOP,
            num_shards=NUM_TPU_CORES,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    examples = read_txt_to_examples(DATA_DIR)
    label_list = get_label_list(examples)

    train_examples = random.sample(examples, int(0.9*len(examples)))
    eval_examples = list(set(examples) - set(train_examples))
    num_train_steps = int(
        len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list) + 1,
        init_checkpoint=INIT_CHECKPOINT,
        learning_rate=LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=USE_TPU,
        use_one_hot_embeddings=USE_TPU)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=USE_TPU,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        predict_batch_size=PREDICT_BATCH_SIZE)


    train_file = os.path.join(OUTPUT_DIR, "train0516.tf_record")
    filed_based_convert_examples_to_features(
        train_examples, label_list, MAX_SEQ_LENGTH, tokenizer, train_file)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", TRAIN_BATCH_SIZE)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)



    eval_file = os.path.join(OUTPUT_DIR, "eval0516.tf_record")
    filed_based_convert_examples_to_features(
        eval_examples, label_list, MAX_SEQ_LENGTH, tokenizer, eval_file)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d", len(eval_examples))
    tf.logging.info("  Batch size = %d", EVAL_BATCH_SIZE)
    eval_steps = None
    if USE_TPU:
        eval_steps = int(len(eval_examples) / EVAL_BATCH_SIZE)
    eval_drop_remainder = True if USE_TPU else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=eval_drop_remainder)
    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
    output_eval_file = os.path.join(OUTPUT_DIR, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        tf.logging.info("***** Eval results *****")
        for key in sorted(result.keys()):
            tf.logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
'''
    if DO_PREDICT:
        token_path = os.path.join(OUTPUT_DIR, "token_test.txt")
        with open('./output/label2id.pkl', 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}
        if os.path.exists(token_path):
            os.remove(token_path)
        predict_examples = processor.get_test_examples(DATA_DIR)

        predict_file = os.path.join(OUTPUT_DIR, "predict.tf_record")
        filed_based_convert_examples_to_features(predict_examples, label_list,
                                                 MAX_SEQ_LENGTH, tokenizer,
                                                 predict_file, mode="test")

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", PREDICT_BATCH_SIZE)
        if USE_TPU:
            # Warning: According to tpu_estimator.py Prediction on TPU is an
            # experimental feature and hence not supported here
            raise ValueError("Prediction in TPU not supported")
        predict_drop_remainder = True if USE_TPU else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=MAX_SEQ_LENGTH,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(OUTPUT_DIR, "label_test.txt")
        with open(output_predict_file, 'w') as writer:
            for prediction in result:
                output_line = "\n".join(id2label[id] for id in prediction if id != 0) + "\n"
                writer.write(output_line)
'''