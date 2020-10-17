# coding=utf-8
# @author: quewentian
# code reference from from BERT-train2deploy-master,https://github.com/xmxoxo/BERT-train2deploy
"""freeze ckpt file into pb file"""

import json
import os
from enum import Enum
from termcolor import colored
import sys
import modeling
import logging
import pickle
import tensorflow as tf
import argparse
import shutil
from crf_layer import CRF
from tensorflow.contrib.layers.python.layers import initializers

def set_logger(context, verbose=False):
    if os.name == 'nt':  # for Windows
        return NTLogger(context, verbose)

    logger = logging.getLogger(context)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        '%(levelname)-.1s:' + context + ':[%(filename).3s:%(funcName).3s:%(lineno)3d]:%(message)s', datefmt=
        '%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    return logger


class NTLogger:
    def __init__(self, context, verbose):
        self.context = context
        self.verbose = verbose

    def info(self, msg, **kwargs):
        print('I:%s:%s' % (self.context, msg), flush=True)

    def debug(self, msg, **kwargs):
        if self.verbose:
            print('D:%s:%s' % (self.context, msg), flush=True)

    def error(self, msg, **kwargs):
        print('E:%s:%s' % (self.context, msg), flush=True)

    def warning(self, msg, **kwargs):
        print('W:%s:%s' % (self.context, msg), flush=True)


def create_softmax_model(bert_config, is_training, input_ids, input_mask, segment_ids, labels, num_labels, use_one_hot_embeddings):
    """
           create model
           :param bert_config: bert config
           :param is_training:
           :param input_ids: idx of input data
           :param input_mask:
           :param segment_ids:
           :param labels: idx of labels
           :param num_labels: type of labels
           :param use_one_hot_embeddings:
           :return:
           """
    # representation
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    # embedding_layer = model.get_sequence_output()
    # output_layer = model.get_pooled_output()
    output_layer = model.get_sequence_output()
    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        output_layer = tf.reshape(output_layer, [-1, hidden_size])
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)###w*x
        logits = tf.nn.bias_add(logits, output_bias)###w*x+b
        logits = tf.reshape(logits, [-1, 256, 18])

        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        if labels is not None:
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
        else:
            loss, per_example_loss = None, None
            pred_ids = tf.argmax(probabilities, axis=-1)
        return (loss, per_example_loss, logits, pred_ids)

def create_crf_model(bert_config, is_training, input_ids, input_mask, segment_ids, labels, num_labels, use_one_hot_embeddings):
    """
        create model
        :param bert_config: bert config
        :param is_training:
        :param input_ids: idx of input data
        :param input_mask:
        :param segment_ids:
        :param labels: idx of labels
        :param num_labels: type of labels
        :param use_one_hot_embeddings:
        :return:
        """
    # representation
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    embedding = model.get_sequence_output()
    max_seq_length = embedding.shape[1].value
    used = tf.sign(tf.abs(input_ids))
    lengths = tf.reduce_sum(used, reduction_indices=1)

    crf = CRF(embedded_chars=embedding,droupout_rate=0.5,initializers=initializers, num_labels=num_labels,
                    seq_length=max_seq_length, labels=labels, lengths=lengths,
                    is_training=is_training)
    (total_loss, logits, trans, pred_ids) = crf.add_crf_layer()

    return (total_loss, logits, trans, pred_ids)

def init_predict_var(path):
    label2id_file = os.path.join(path, 'label2id.pkl')
    if os.path.exists(label2id_file):
        with open(label2id_file, 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}
            num_labels = len(label2id.items())
    return num_labels, label2id, id2label


# def optimize_class_model(args, logger=None):
def optimize_ner_model(args, logger=None):
    if not logger:
        logger = set_logger(colored('CLASSIFICATION_MODEL, Lodding...', 'cyan'), args.verbose)
        pass
    try:
        # if PB file exists, return the path of PB file; else, turn ckpt file into PB file and return PB file path
        if args.model_pb_dir is None:
            tmp_file = args.model_dir
        else:
            tmp_file = args.model_pb_dir

        pb_file = os.path.join(tmp_file, 'ner_model.pb')
        if os.path.exists(pb_file) and (args.pb_file_name == '' or args.pb_version == ''):
            print('pb_file exits', pb_file)
            return pb_file

        # read num_labels from label2id.pkl
        if not args.num_labels:
            num_labels, label2id, id2label = init_predict_var()
        else:
            num_labels = args.num_labels

        graph = tf.Graph()
        with graph.as_default():
            with tf.Session() as sess:
                input_ids = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_ids')
                input_mask = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_mask')

                bert_config = modeling.BertConfig.from_json_file(os.path.join(args.bert_model_dir, 'bert_config.json'))

                if args.model=='softmax':
                    loss, per_example_loss, logits, pred_ids = create_softmax_model(bert_config=bert_config,
                                                                                                is_training=False,
                                                                                                input_ids=input_ids,
                                                                                                input_mask=input_mask,
                                                                                                segment_ids=None,
                                                                                                labels=None,
                                                                                                num_labels=num_labels,
                                                                                                use_one_hot_embeddings=False)
                    #######NEED EXPERIMENT#################
                    pred_ids = tf.identity(pred_ids, 'pred_ids')
                    saver = tf.train.Saver()
                else:
                    if args.model=='CRF':
                        total_loss, logits, trans, pred_ids = create_crf_model(bert_config=bert_config,
                                                                                                    is_training=False,
                                                                                                    input_ids=input_ids,
                                                                                                    input_mask=input_mask,
                                                                                                    segment_ids=None,
                                                                                                    labels=None,
                                                                                                    num_labels=num_labels,
                                                                                                    use_one_hot_embeddings=False)
                        #######NEED EXPERIMENT#################
                       pred_ids = tf.identity(pred_ids, 'pred_ids')

                        saver = tf.train.Saver()

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                latest_checkpoint = tf.train.latest_checkpoint(args.model_dir)
                logger.info('loading... %s ' % latest_checkpoint)
                saver.restore(sess, latest_checkpoint)
                logger.info('freeze...')
                from tensorflow.python.framework import graph_util
                ##carry out pred_ids variable in ckpt file and save pred_ids as constant in pb file
                tmp_g = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['pred_ids'])
                logger.info('predict cut finished !!!')

        # Store binary model in file
        logger.info('write graph to a tmp file: %s' % pb_file)
        with tf.gfile.GFile(pb_file, 'wb') as f:
            f.write(tmp_g.SerializeToString())

        #if pb_file_name pb_version exists, generate tfserving format directly
        if args.pb_file_name == '' or args.pb_version == '':
            return pb_file

        pd_model_file = os.path.join(os.path.join(args.model_dir, args.pb_file_name), args.pb_version)
        if os.path.exists(pd_model_file):
            shutil.rmtree(pd_model_file)
            os.makedirs(pd_model_file)
        shutil.move(pb_file, pd_model_file)
        latest_checkpoint = tf.train.latest_checkpoint(args.model_dir)
        if latest_checkpoint == '':
            return pb_file
        data_path = os.path.join(os.path.join(args.model_dir, "{}.data-00000-of-00001".format(latest_checkpoint)))
        index_path = os.path.join(os.path.join(args.model_dir, "{}.index".format(latest_checkpoint)))
        pd_variables_path = os.path.join(pd_model_file, 'variables')
        os.makedirs(pd_variables_path)
        shutil.copy(data_path, os.path.join(pd_variables_path, "variables.data-00000-of-00001"))
        shutil.copy(index_path, os.path.join(pd_variables_path, "variables.index"))
        logger.info('pb dir success!!! {}'.format(pd_model_file))
        return pd_model_file
    except Exception as e:
        logger.error('fail to optimize the graph! %s' % e, exc_info=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trans ckpt file to .pb file')
    parser.add_argument('-bert_model_dir', type=str, required=True,
                        help='chinese google bert model path')
    parser.add_argument('-model_dir', type=str, required=True,
                        help='directory of a pretrained BERT model')
    parser.add_argument('-model_pb_dir', type=str, default=None,
                        help='directory of a pretrained BERT model,default = model_dir')
    parser.add_argument('-max_seq_len', type=int, default=256,
                        help='maximum length of a sequence,default:256')
    parser.add_argument('-num_labels', type=int, default=None,
                        help='length of all labels,default=2')
    parser.add_argument('-verbose', action='store_true', default=False,
                        help='turn on tensorflow logging for debug')
    parser.add_argument('-pb_file_name', type=str, default='',
                        help='please input pb file name')
    parser.add_argument('-pb_version', type=str, default='',
                        help='please input pb version')
    parser.add_argument('-model', type=str, default='',
                        help='please input which model you want to freeze(CRF or softmax)')

    args = parser.parse_args()

    optimize_ner_model(args, logger=None)
