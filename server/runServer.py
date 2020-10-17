"""
运行 BERT NER Server
# @Author  : MaCan
# @File    : run.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def start_server():
    # from bert_base.server.myserver import BertServer
    from myServer import BertServer
    from myHelper import get_args_parser

    args = get_args_parser().parse_args(['-model_dir', '/home/lzhpc/0BERT-NER-master/new_no_relation3',
                                         '-model_pb_dir', '/home/lzhpc/0BERT-NER-master/new_no_relation3',
                                         '-bert_model_dir', '/home/lzhpc/0BERT-NER-master/cased_L-12_H-768_A-12_param/',
                                         '-http_port', '8080',
                                         '-port', '5575',
                                         '-port_out', '5576',
                                         '-max_seq_len', '256',
                                         '-num_worker', '2'])
    server = BertServer(args)
    print(args)
    server = BertServer(args)
    server.start()
    server.join()



if __name__ == '__main__':

    start_server()