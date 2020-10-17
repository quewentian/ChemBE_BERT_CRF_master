# -*- coding:utf-8 -*-
# BASED ON Google_BERT.
# code reference from :zhoukaiyin/,Macan/
# @Author:quewentian
"""BERT CRF NER model"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import json
import logging


import tensorflow as tf
import codecs
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.client import timeline
from tensorflow.contrib import estimator
from bert import modeling
from bert import optimization
from bert import tokenization
from crf_layer import CRF
import numpy as np

import tf_metrics
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

flags = tf.flags

FLAGS = flags.FLAGS
bert_path='cased_L-12_H-768_A-12_param'
root_path='../'

flags.DEFINE_string(
    "data_dir",  '/home/lzhpc/data/NERdata3/',
    "The input datadir.",
)

flags.DEFINE_string(
    "bert_config_file", os.path.join(bert_path, 'bert_config.json'),
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "task_name", 'ner', "The name of the task to train."
)

flags.DEFINE_string(
    "output_dir",  '2020entity3',
    "The output directory where the model checkpoints will be written."
)

flags.DEFINE_string(
    "best_output_dir",  '2020entity3/best',
    "The output directory where the model checkpoints will be written."
)


flags.DEFINE_string(
    "init_checkpoint", '/home/lzhpc/0BERT-NER-master/ckpt2020/model.ckpt-100000',
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_boolean('clean',False, 'remove the files which created by last training')

flags.DEFINE_bool("is_label", True, "Is the dataset labeled")

flags.DEFINE_bool("do_train",True, "Whether to run training."
)
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 128, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 6.0, "Total number of training epochs to perform.")
flags.DEFINE_float('droupout_rate', 0.5, 'Dropout rate')
flags.DEFINE_float('clip', 5, 'Gradient clip')
flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("vocab_file", os.path.join(bert_path, 'vocab.txt'),
                    "The vocabulary file that the BERT model was trained on.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


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


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file, batch_num = None):
        """Reads a BIO data."""
        label_list= ["<B-SVN>", "<I-SVN>", "<O>", "<B-MTH>", "<I-MTH>", "<B-RCT>", "<I-RCT>", "<B-ENG>", "<I-ENG>",
                "<B-EGVL>","<B-BON>","<I-BON>","<B-CMP>","<I-CMP>" ,"X", "[CLS]", "[SEP]"]

        with open(input_file) as f:
            words_num = 0  ###New indicators
            lines = []  ##2-dimensional vector，[[label of one sentence（type:string，space seperate)，words of one sentence（type:string，space seperate)]]
            words = []  ##words of one sentence
            labels = []  ##label of one sentence
            for line in f:
                contends = line.strip()
                word = line.strip().split(' ')[0]
                label = line.strip().split(' ')[-1]
                if label not in label_list:
                    label='<O>'
                words_num+=1
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
                try:
                    if len(contends) == 0 and (words[-1] == '.' or words[-1] == '。' or words[-1] == '!'):
                        l = ' '.join([label for label in labels if len(label) > 0])  ###
                        w = ' '.join([word for word in words if len(word) > 0])
                        lines.append([l, w])
                        words = []
                        labels = []
                        continue
                    words.append(word)
                    labels.append(label)
                except:
                    print(words,'   hhh')
            return (lines,words_num)

class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir, batch_num=None):
        train_examples = self._read_data(os.path.join(data_dir, "train.txt"),batch_num)
        return self._create_example(train_examples[0] , "train"), train_examples[1]


    def get_dev_examples(self, data_dir, batch_num=None):
        dev_examples = self._read_data(os.path.join(data_dir, "dev.txt"),batch_num)
        return self._create_example(dev_examples[0], "dev"), dev_examples[1]


    def get_test_examples(self, data_dir,batch_num=None):
        test_examples = self._read_data(os.path.join(data_dir, "test.txt"),batch_num)
        return self._create_example(test_examples[0], "test"), test_examples[1]


    def get_labels(self, data_dir):
        return self._get_labels(
            self._read_data(os.path.join(data_dir, "train.txt"))[0]
        )

    def get_predict_labels(self, data_dir):
        return self._get_labels(
            self._read_data(os.path.join(data_dir, "test.txt"))[0]
        )

    def _get_labels(self, lines):
        labels = set()
        for line in lines:
            for l in line[0].split():
                labels.add(l)
        labels = list(labels)
        labels.insert(0, "[CLS]")
        labels.insert(1, "[SEP]")
        labels.insert(2, "X")
        return labels

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            # print(line[1],'   text333')
            # print(line[0],'   label222')
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            if i == 0:
                print(label)
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples


def write_tokens(tokens, mode):
    """
    Write the sequence analysis result to the file
    only call this function when mode=test
    :param tokens:
    :param mode:
    :return:
    """
    if mode == "test":
        path = os.path.join(FLAGS.output_dir, "token_" + mode + ".txt")
        wf = codecs.open(path, 'a', encoding='utf-8')
        for token in tokens:
            if token != "**NULL**":
                wf.write(token + '\n')
        wf.close()


def convert_single_example(ex_index, example, label_list,
                           max_seq_length, tokenizer, mode,
                           predict_label_list=None):
    """
    analyse one example, change word to id and label to id, then Structure them into InputFeatures object
    :param ex_index: index
    :param example: one example
    :param label_list: label list
    :param max_seq_length:
    :param tokenizer:
    :param mode:
    :return:
    """
    label_map = {}
    if os.path.exists(os.path.join(FLAGS.output_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'rb') as rf:
            label_map = pickle.load(rf)
    else:
        # index label from 1
        for (i, label) in enumerate(label_list, 1):
            label_map[label] = i
        # save label->index ,type map
        # print("the label map is", label_map)
        with codecs.open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)
    eval_list_dir = os.path.join(FLAGS.output_dir, 'eval_ids_list.txt')
    if not os.path.exists(eval_list_dir):
        # try:
        # eval_list = [ label_map[i] for i in predict_label_list if i!='O' and i!='<O>' and i!='X' and i!="[CLS]" and i!="[SEP]"]
        eval_list = [label_map[i] for i in predict_label_list if i in  ["<B-SVN>", "<I-SVN>","<B-MTH>", "<I-MTH>", "<B-RCT>", "<I-RCT>", "<B-ENG>", "<I-ENG>",
                "<B-EGVL>", "<B-BON>", "<I-BON>", "<B-CMP>", "<I-CMP>"]]
        # except:
        #     print('key error: ',)
        file=open(eval_list_dir, 'w')
        for i in eval_list:
            file.write(str(i) + '\n')
        file.close()
        print("Get the eval list in eval_tag_list.txt")
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        # tokenization
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:  #
                labels.append("X")
    # tokens = tokenizer.tokenize(example.text)
    # cut sequence
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 The sequence needs to add a sentence beginning and ending mark
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # sentence beginning: CLS sign
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])  #
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")  #
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  #  change tokens to ID
    input_mask = [1] * len(input_ids)
    # label_mask = [1] * len(input_ids)
    # padding, 使用
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

    # print data info
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

    # Structured as a class
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    # mode='test'：call this function
    write_tokens(ntokens, mode)
    return feature


def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, mode=None, predict_label_list=None
):
    """
    change data to TF_Record structure as the input of model
    :param examples:  sample
    :param label_list:label list
    :param max_seq_length:
    :param tokenizer: tokenizer object
    :param output_file: tf.record output path
    :param mode:
    :return:
    """
    writer = tf.python_io.TFRecordWriter(output_file)
    # Iterate over training data
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        # for each training example
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode, predict_label_list)

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


def get_file_based_dataset(input_file, seq_length, is_training,  batch_size ,drop_remainder=False):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        # "label_ids":tf.VarLenFeature(tf.int64),
        # "label_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

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


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):
    """
    create model
    :param bert_config: bert cofig
    :param is_training:
    :param input_ids: idx of data
    :param input_mask:
    :param segment_ids:
    :param labels: idx of label
    :param num_labels: number of categories
    :param use_one_hot_embeddings:
    :return:
    """
    # load BertModel, and acuqire the corresponding embedding
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )
    # acuqire the corresponding embedding
    embedding = model.get_sequence_output()
    max_seq_length = embedding.shape[1].value

    used = tf.sign(tf.abs(input_ids))
    lengths = tf.reduce_sum(used, reduction_indices=1)  # [batch_size]

    crf = CRF(embedded_chars=embedding,droupout_rate=FLAGS.droupout_rate,initializers=initializers, num_labels=num_labels,
                    seq_length=max_seq_length, labels=labels, lengths=lengths,
                    is_training=is_training)
    rst = crf.add_crf_layer()
    return rst



def restore_model(init_checkpoint, bert_config, is_training, input_ids, input_mask,
                  segment_ids, label_ids, num_labels, use_one_hot_embeddings):
    tf.reset_default_graph()

    # use params to construct mode。 input_idx：idx of input data，label_ids：idx of input labels
    rst = create_model(bert_config, is_training, input_ids,
         input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    # load BERT model
    if init_checkpoint:
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                   init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    # print loaded params of model
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)

    return rst

def eval_phase(label_ids, pred_ids, num_labels):
    # Viterbi decode of result
    # crf decode
    eval_list = []
    assert True == os.path.exists(os.path.join(FLAGS.output_dir, "eval_ids_list.txt"))
    list_file = open(os.path.join(FLAGS.output_dir, "eval_ids_list.txt"), 'r')
    contents = list_file.readlines()
    for item in contents:
        eval_list.append(int(item.strip()))
    assert 0 < len(eval_list)
    print("eval_list:", eval_list)
    weight = tf.sequence_mask(FLAGS.max_seq_length)

    precision = tf_metrics.precision(label_ids, pred_ids, num_labels, eval_list, weight)
    tf.summary.scalar("precision", precision[1])
    recall = tf_metrics.recall(label_ids, pred_ids, num_labels, eval_list, weight)
    tf.summary.scalar("recall", recall[1])
    f = tf_metrics.f1(label_ids, pred_ids, num_labels, eval_list, weight)
    tf.summary.scalar("f1",f[1])

    cmp_precision = tf_metrics.precision(label_ids, pred_ids, num_labels, [13, 14], weight)
    tf.summary.scalar("cmp_precision",cmp_precision[1])
    cmp_recall = tf_metrics.recall(label_ids, pred_ids, num_labels, [13, 14], weight)
    tf.summary.scalar("cmp_recall", cmp_recall[1])
    cmp_f = tf_metrics.f1(label_ids, pred_ids, num_labels, [13, 14], weight)
    tf.summary.scalar("cmp_f", cmp_f[1])

    bon_precision = tf_metrics.precision(label_ids, pred_ids, num_labels, [11, 12],weight)
    tf.summary.scalar("bon_precision", bon_precision[1])
    bon_recall = tf_metrics.recall(label_ids, pred_ids, num_labels, [11, 12], weight)
    tf.summary.scalar("bon_recall", bon_recall[1])
    bon_f = tf_metrics.f1(label_ids, pred_ids, num_labels, [11, 12], weight)
    tf.summary.scalar("bon_f", bon_f[1])

    svn_precision = tf_metrics.precision(label_ids, pred_ids, num_labels, [1, 2], weight)
    tf.summary.scalar("svn_precision", svn_precision[1])
    svn_recall = tf_metrics.recall(label_ids, pred_ids, num_labels, [1, 2],weight)
    tf.summary.scalar("svn_recall", svn_recall[1])
    svn_f = tf_metrics.f1(label_ids, pred_ids, num_labels, [1, 2], weight)
    tf.summary.scalar("svn_f", svn_f[1])

    mth_precision = tf_metrics.precision(label_ids, pred_ids, num_labels, [4, 5], weight)
    tf.summary.scalar("mth_precision", mth_precision[1])
    mth_recall = tf_metrics.recall(label_ids, pred_ids, num_labels, [4, 5], weight)
    tf.summary.scalar("mth_recall", mth_recall[1])
    mth_f = tf_metrics.f1(label_ids, pred_ids, num_labels, [4, 5], weight)
    tf.summary.scalar("mth_f", mth_f[1])

    rct_precision = tf_metrics.precision(label_ids,pred_ids, num_labels, [6, 7], weight)
    tf.summary.scalar("rct_precision", rct_precision[1])
    rct_recall = tf_metrics.recall(label_ids, pred_ids, num_labels, [6, 7],weight)
    tf.summary.scalar("rct_recall", rct_recall[1])
    rct_f = tf_metrics.f1(label_ids, pred_ids, num_labels, [6, 7], weight)
    tf.summary.scalar("rct_f", rct_f[1])

    eng_precision = tf_metrics.precision(label_ids, pred_ids, num_labels, [8, 9], weight)
    tf.summary.scalar("eng_precision", eng_precision[1])
    eng_recall = tf_metrics.recall(label_ids, pred_ids, num_labels, [8, 9], weight)
    tf.summary.scalar("eng_recall", eng_recall[1])
    eng_f = tf_metrics.f1(label_ids, pred_ids, num_labels, [8, 9], weight)
    tf.summary.scalar("eng_f", eng_f[1])

    egvl_precision = tf_metrics.precision(label_ids, pred_ids, num_labels, [10], weight)
    tf.summary.scalar("egvl_precision", egvl_precision[1])
    egvl_recall = tf_metrics.recall(label_ids, pred_ids, num_labels, [10], weight)
    tf.summary.scalar("egvl_recall", egvl_recall[1])
    egvl_f = tf_metrics.f1(label_ids,pred_ids, num_labels, [10], weight)
    tf.summary.scalar("egvl_f", egvl_f[1])

    # return (precision, recall, f)
    return (precision, recall, f, cmp_precision,cmp_recall,cmp_f, bon_precision,bon_recall,bon_f,
            svn_precision,svn_recall,svn_f, mth_precision,mth_recall,mth_f,rct_precision,rct_recall,rct_f,
            eng_precision,eng_recall,eng_f,egvl_precision,egvl_recall,egvl_f)


def build_train_dev_data_pipeline(processor, tokenizer, label_list, predict_label_list):

    """
    code reference https://blog.csdn.net/briblue/article/details/80962728#commentBox，
    and https://isaacchanghau.github.io/post/tensorflow_dataset_api/
    :param processor:
    :param tokenizer:
    :param label_list:
    :param predict_label_list:
    :return:
    """
    """
    load training set as Dataset object
    """
    train_examples, train_words_num = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")

    filed_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer,
        train_file, predict_label_list=predict_label_list)

    tf.logging.info("***** Running trainning*****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)

    train_dataset = get_file_based_dataset(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        batch_size=FLAGS.train_batch_size,
        is_training=True)

    """
    load test set as Dataset object
    """
    dev_examples, dev_words_num = processor.get_dev_examples(FLAGS.data_dir, batch_num=20000)
    dev_file = os.path.join(FLAGS.output_dir, "dev.tf_record")

    filed_based_convert_examples_to_features(
        dev_examples, label_list, FLAGS.max_seq_length, tokenizer,
        dev_file, predict_label_list=predict_label_list)

    tf.logging.info("***** Running dev set*****")
    tf.logging.info("  Num examples = %d", len(dev_examples))
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
    dev_dataset = get_file_based_dataset(
        input_file=dev_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        batch_size=FLAGS.eval_batch_size)

    """
    construct iterator for trian and dev set
    """
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)
    train_init_op = train_dataset.make_one_shot_iterator()
    dev_init_op = dev_dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    tf.logging.info("*** Features ***")
    for name in sorted(next_element.keys()):
        tf.logging.info("  name = %s, shape = %s" % (name, next_element[name].shape))
    input_ids = next_element["input_ids"]
    input_mask = next_element["input_mask"]
    segment_ids = next_element["segment_ids"]
    label_ids = next_element["label_ids"]

    return (input_ids, input_mask, segment_ids, label_ids, train_init_op, dev_init_op,
            num_train_steps, handle)


def training_phase(bert_config, processor, tokenizer, label_list, predict_label_list):
    """
    :param processor:
    :param tokenizer:
    :param label_list:
    :param predict_label_list:
    :return:
    """
    """
     contruct training_data and dev_data joint pipeline，return train_init_op and dev_init_op to initialize
     """
    (input_ids, input_mask, segment_ids, label_ids, train_init_op, dev_init_op,
     num_train_steps, handle) = build_train_dev_data_pipeline(processor, tokenizer, label_list, predict_label_list)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    tf.logging.info("=========== Train and evaluate set are loaded ============")
    """
    construct model
    """
    (total_loss, logits, trans, pred_ids) = create_model(bert_config=bert_config, is_training=True,
                                                         input_ids=input_ids,
                                                         input_mask=input_mask, segment_ids=segment_ids,
                                                         labels=label_ids,
                                                         num_labels=len(label_list) + 1, use_one_hot_embeddings=False)
    tf.summary.scalar("total_loss", total_loss)
    tf.logging.info("================= Model is built ====================")
    """
    load BERT model
    """
    init_checkpoint = FLAGS.init_checkpoint
    tvars = tf.trainable_variables()

    if init_checkpoint:
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                  init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    train_op = optimization.create_optimizer(
        total_loss, FLAGS.learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)
    tf.logging.info("================BERT are loaded to initiate and train_op is built===========")
    """
    evaluate
    """
    (precision, recall, f, cmp_precision, cmp_recall, cmp_f, bon_precision, bon_recall, bon_f,
     svn_precision, svn_recall, svn_f, mth_precision, mth_recall, mth_f, rct_precision, rct_recall, rct_f,
     eng_precision, eng_recall, eng_f, egvl_precision, egvl_recall, egvl_f) = eval_phase(label_ids, pred_ids,len(label_list) + 1)
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
    running_vars_initializer = tf.variables_initializer(var_list=running_vars) #初始化precision、recall、f1这些计算节点

    prec_scalar, prec_op = precision
    recall_scalar, recall_op = recall
    f1_scalar, f1_op = f

    cmp_prec_scalar, cmp_prec_op = cmp_precision
    cmp_recall_scalar, cmp_recall_op = cmp_recall
    cmp_f1_scalar, cmp_f1_op = cmp_f

    bon_prec_scalar, bon_prec_op = bon_precision
    bon_recall_scalar, bon_recall_op = bon_recall
    bon_f1_scalar, bon_f1_op = bon_f

    svn_prec_scalar, svn_prec_op = svn_precision
    svn_recall_scalar, svn_recall_op = svn_recall
    svn_f1_scalar, svn_f1_op = svn_f

    mth_prec_scalar, mth_prec_op = mth_precision
    mth_recall_scalar, mth_recall_op = mth_recall
    mth_f1_scalar, mth_f1_op = mth_f

    rct_prec_scalar, rct_prec_op =rct_precision
    rct_recall_scalar, rct_recall_op =rct_recall
    rct_f1_scalar, rct_f1_op = rct_f

    eng_prec_scalar, eng_prec_op = eng_precision
    eng_recall_scalar, eng_recall_op = eng_recall
    eng_f1_scalar, eng_f1_op =eng_f

    egvl_prec_scalar, egvl_prec_op = egvl_precision
    egvl_recall_scalar, egvl_recall_op = egvl_recall
    egvl_f1_scalar, egvl_f1_op = egvl_f

    tf.logging.info("=================eval metrics are loaded=========================")
    """
    Save models 
    """
    saver = tf.train.Saver(max_to_keep=5)

    merged = tf.summary.merge_all()

    tf.logging.info("==================Entering Session Running=========================")
    with tf.Session() as sess:
        best_precision=0.4
        sess.run(tf.global_variables_initializer()) #除了CRF层，其它层都被initialized了
        train_writer = tf.summary.FileWriter(FLAGS.output_dir , sess.graph)
        dev_writer = tf.summary.FileWriter(FLAGS.output_dir + '/eval')
        train_iterator_handle = sess.run(train_init_op.string_handle())
        dev_iterator_handle = sess.run(dev_init_op.string_handle())
        for step in range(num_train_steps):
            if step % 100 == 0:
                tf.logging.info("===============evaluate at %d step=============="%step)
                sess.run(running_vars_initializer)
                sess.run(dev_init_op.initializer)
                # while True:
                while True:
                    try:
                        # print(sess.run([label_ids, pred_ids], feed_dict={handle: dev_iterator_handle}))
                        summary,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = sess.run([merged, prec_op, recall_op, f1_op,cmp_prec_op,cmp_recall_op,cmp_f1_op,svn_prec_op,svn_recall_op,svn_f1_op,bon_prec_op,bon_recall_op,bon_f1_op,mth_prec_op,mth_recall_op,mth_f1_op,rct_prec_op,rct_recall_op,rct_f1_op,eng_prec_op,eng_recall_op,eng_f1_op,egvl_prec_op,egvl_recall_op,egvl_f1_op], feed_dict={handle: dev_iterator_handle})
                    except tf.errors.OutOfRangeError:
                        break
                dev_writer.add_summary(summary, step)
                _precision, _recall, _f1, _cmp_prec, _cmp_recall, _cmp_f1,_bon_prec,_bon_recall,_bon_f1,_svn_prec,_svn_recall,_svn_f1,_mth_prec,_mth_recall,_mth_f1,_rct_prec,_rct_recall,_rct_f1, _eng_prec,_eng_recall,_eng_f1,_egvl_prec,_egvl_recall,_egvl_f1= sess.run([prec_scalar, recall_scalar, f1_scalar,cmp_prec_scalar,cmp_recall_scalar,cmp_f1_scalar,
                                                                                                            bon_prec_scalar,bon_recall_scalar,bon_f1_scalar,svn_prec_scalar,svn_recall_scalar,svn_f1_scalar,mth_prec_scalar,mth_recall_scalar,mth_f1_scalar,rct_prec_scalar,rct_recall_scalar,rct_f1_scalar,eng_prec_scalar,eng_recall_scalar,eng_f1_scalar,egvl_prec_scalar,egvl_recall_scalar,egvl_f1_scalar ])
                print("At step {}, the precision is {:.2f}%,the recall is {:.2f}%,the f1 is {:.2f}%".format(step, _precision*100, _recall*100, _f1*100))
                print("At step {}, the compound precision is {:.2f}%,the recall is {:.2f}%,the f1 is {:.2f}%".format(step,_cmp_prec * 100,_cmp_recall * 100,_cmp_f1 * 100))
                print("At step {}, the bond precision is {:.2f}%,the recall is {:.2f}%,the f1 is {:.2f}%".format(step,_bon_prec * 100,_bon_recall * 100, _bon_f1 * 100))
                print("At step {}, the solvent precision is {:.2f}%,the recall is {:.2f}%,the f1 is {:.2f}%".format(step,_svn_prec * 100,_svn_recall * 100,_svn_f1 * 100))
                print("At step {}, the method precision is {:.2f}%,the recall is {:.2f}%,the f1 is {:.2f}%".format(step, _mth_prec * 100, _mth_recall * 100, _mth_f1 * 100))
                print("At step {}, the reaction precision is {:.2f}%,the recall is {:.2f}%,the f1 is {:.2f}%".format(step,  _rct_prec * 100,_rct_recall * 100,_rct_f1 * 100))
                print("At step {}, the pKa precision is {:.2f}%,the recall is {:.2f}%,the f1 is {:.2f}%".format(step, _eng_prec * 100, _eng_recall * 100,_eng_f1 * 100))
                print("At step {}, the pKa value precision is {:.2f}%,the recall is {:.2f}%,the f1 is {:.2f}%".format(step,_egvl_prec * 100, _egvl_recall * 100, _egvl_f1 * 100))
                if _precision>best_precision and _recall>0.4:
                    # print("Best one! At step {}, the precision is {:.2f}%,the recall is {:.2f}%,the f1 is {:.2f}%".format(step, _recall * 100,_f1 * 100))
                    best_precision=_precision
                    tf.logging.info("===============save model at %d step==============" % step)
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _, _total_loss = sess.run([merged, train_op, total_loss],
                                                       feed_dict={handle: train_iterator_handle},
                                                       options=run_options,
                                                       run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                    train_writer.add_summary(summary, step)
                    tf.logging.info("========== the total loss is %.5f ===============" % (_total_loss))
                    print('Adding run metadata for', step)
                    save_path = saver.save(sess, os.path.join(FLAGS.best_output_dir, "model.ckpt"), global_step=step)
                    print("Model saved in path: %s" % save_path)
            else:
                if step % 200 == 0:
                # if step % 1000 == 999:
                    tf.logging.info("===============save model at %d step==============" % step)
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ , _total_loss= sess.run([merged, train_op, total_loss],
                                          feed_dict={handle: train_iterator_handle},
                                          options=run_options,
                                          run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                    train_writer.add_summary(summary, step)
                    tf.logging.info("========== the total loss is %.5f ===============" %(_total_loss))
                    print('Adding run metadata for', step)
                    save_path = saver.save(sess, os.path.join(FLAGS.output_dir, "model.ckpt"), global_step=step)
                    print("Model saved in path: %s" % save_path)
                else:
                    # print(sess.run([pred_ids, label_ids], feed_dict={handle: train_iterator_handle}))
                    summary, _ = sess.run([merged, train_op], feed_dict={handle: train_iterator_handle})
                    train_writer.add_summary(summary, step)
        train_writer.close()
        dev_writer.close()

def predict_phase(bert_config, processor, tokenizer, label_list, predict_label_list):
    test_batch_size = 32

    predict_examples, predict_examples_number = processor.get_test_examples(FLAGS.data_dir, test_batch_size)
    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    filed_based_convert_examples_to_features(predict_examples, label_list,
                                             FLAGS.max_seq_length, tokenizer,
                                             predict_file, mode="test", predict_label_list=predict_label_list)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d", len(predict_examples))##367
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
    tf.logging.info(" Character Number = %d", predict_examples_number)

    predict_dataset = get_file_based_dataset(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        batch_size=FLAGS.predict_batch_size)

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, predict_dataset.output_types, predict_dataset.output_shapes)
    predict_init_op = predict_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    # predict_iterator = predict_dataset.make_one_shot_iterator()
    # next_element = predict_iterator.get_next()

    predict_result = []
    tf.logging.info("*** Features ***")
    global_step = 0
    for name in sorted(next_element.keys()):
        tf.logging.info("  name = %s, shape = %s" % (name, next_element[name].shape))
    input_ids = next_element["input_ids"]
    input_mask = next_element["input_mask"]
    segment_ids = next_element["segment_ids"]
    label_ids = next_element["label_ids"]
    # print(len(input_ids),'   input id')
    tf.logging.info("=========== Predict set are loaded ============")

    (total_loss, logits, trans, pred_ids) = create_model(bert_config=bert_config, is_training=False, input_ids=input_ids,
                          input_mask=input_mask, segment_ids=segment_ids, labels=label_ids,
                          num_labels=len(label_list) + 1, use_one_hot_embeddings=False)

    tf.logging.info("================= Model is built ====================")

    saver = tf.train.Saver()

    merged = tf.summary.merge_all()


    with tf.Session() as sess:
        #
        saver.restore(sess, 'output2/best/model.ckpt-2500')
        # sess.run(tf.global_variables_initializer())
        # predict_writer = tf.summary.FileWriter(FLAGS.output_dir+'/predict' , sess.graph)
        predict_init_handle = sess.run(predict_init_op.string_handle())
        # while True:
            # try:
                # tf.logging.info("======================= the %d step starts ==================="%global_step)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
                # result = sess.run( pred_ids)
        label, result = sess.run([label_ids, pred_ids], feed_dict={handle: predict_init_handle}
                                          , options=run_options, run_metadata=run_metadata)
                # summary, label, result= sess.run([merged, label_ids, pred_ids], feed_dict={handle:predict_init_handle}
                #                                  ,options=run_options, run_metadata=run_metadata)
                # predict_writer.add_summary(summary, global_step=global_step)
        print("the label is", label)
        print("the result is", result)
        print(len(result))######[ 16 , 128 ]
        result_to_pair(predict_examples, result)

        global_step+=0



def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    processors = {
        "ner": NerProcessor
    }

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    (label_list, predict_label_list) = load_soft_labels(processor)

    if FLAGS.do_train:
        clean_before_train()
        training_phase(bert_config, processor, tokenizer, label_list, predict_label_list)
    if FLAGS.do_predict:
        predict_phase(bert_config, processor, tokenizer, label_list, predict_label_list)

def clean_before_train():
    # 在train 的时候，才删除上一轮产出的文件，在predicted 的时候不做clean
    if FLAGS.clean and FLAGS.do_train:
        if os.path.exists(FLAGS.output_dir):
            def del_file(path):
                ls = os.listdir(path)
                for i in ls:
                    c_path = os.path.join(path, i)
                    if os.path.isdir(c_path):
                        del_file(c_path)
                    else:
                        os.remove(c_path)

            try:
                del_file(FLAGS.output_dir)
            except Exception as e:
                print(e)
                print('pleace remove the files of output dir and data.conf')
                exit(-1)

def load_soft_labels(processor):
    label_list=["<B-SVN>", "<I-SVN>", "<O>", "<B-MTH>", "<I-MTH>", "<B-RCT>", "<I-RCT>", "<B-ENG>", "<I-ENG>",
         "<B-EGVL>", "<B-BON>", "<I-BON>", "<B-CMP>", "<I-CMP>", "X", "[CLS]", "[SEP]"]

    with codecs.open(os.path.join(FLAGS.data_dir, 'label_list.txt'), 'w', encoding='utf-8') as f:
        for label in label_list:
            f.write(label + '\n')
        f.close()
    print("the label list is:", label_list)
    predict_label_list = processor.get_predict_labels(FLAGS.data_dir)
    print("predict_label_list", predict_label_list)
    return (label_list, predict_label_list)


def result_to_pair(predict_examples, result):
    with codecs.open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'rb') as rf:
        label2id = pickle.load(rf)
        id2label = {value: key for key, value in label2id.items()}
    # print("label2id is", label2id)
    # print("id2label is", id2label)
    print(id2label,'   hhh')

    output_predict_file = os.path.join(FLAGS.output_dir, "predict_file.txt")
    line = ''
    with codecs.open(output_predict_file, 'w', encoding='utf-8') as writer:

        # print(result,'   hhh')
        for predict_line, prediction in zip(predict_examples, result):
            idx = 0

            line_token = str(predict_line.text).split(' ')
            label_token = str(predict_line.label).split(' ')
            print(line_token,'   ###')
            print(label_token,'   666')
            print(prediction,' mmm')


            for id in range(len(line_token)):

                pred = id2label[prediction[id]]

                print(line_token[id], '  ', label_token[id], '  ', pred)
                # if pred in ['[CLS]', '[SEP]']:
                #     continue
                # else:
                line += line_token[id]
                line+=' '
                line+=label_token[id]
                line += ' '
                line += pred
                line += ' '
                line+='\n'
            line+='\n'
            idx += 1

        writer.write(line + '\n')


def load_data():
    processer = NerProcessor()
    processer.get_labels(FLAGS.data_dir)
    example = processer.get_train_examples(FLAGS.data_dir)
    print()


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.FLAGS.set_default('do_train', True)
    flags.FLAGS.set_default('do_eval', True)
    flags.FLAGS.set_default('do_predict',False)
    tf.app.run()
    # load_data()
