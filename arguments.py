import argparse
import time
import os
import sys

class Arguments:
    def __init__(self):

        #
        self.lstm = True
        self.cpu = False
        self.eval = False
        self.model_dir = './kaggle/working'
        self.load_model = './checkpoints/model_0/ckpt-00002000'
        self.num_LSTM_layers = 2
        self.num_MLP_layers = 1
        self.LSTM_hidden_size = 512
        self.MLP_hidden_size = 512
        self.embedding_size = 512

        self.n_heads = 4
        self.n_layers = 3
        self.dim_forward = 256

        self.keep_last_n = None
        self.eval_every_n = 10
        self.log_interval = 10
        self.log_dir = './kaggle/working'
        self.log_name = 'model_0.csv'

        self.max_eval_size = 32

        #data
        self.train_dataset = "./data/" + 'train_plot.json'
        self.dev_dataset = "./data/" + 'test_plot.json'
        self.test_dataset = "./data/" + 'test_plot.json'
        self.code_vocab = "./data/" + 'code_vocab.json'
        self.word_vocab = "./data/" + 'nl_vocab.json'
        self.cpu = False
        self.cuda = False
        self.num_plot_types = 6
        self.joint_plot_types = False
        self.data_order_invariant = False
        self.nl = True
        self.use_comments = True
        self.code_context = True
        self.local_df_only = False
        self.target_code_transform = True
        self.max_num_code_cells = 2
        self.max_word_len = 512 
        self.max_code_context_len = 512
        self.max_decode_len = 200


        #model
        self.hierarchy = True
        self.copy_mechanism = True
        self.nl_code_linking = True

        #train
        self.optimizer ='adam'
        self.lr = 1e-3
        self.lr_decay_steps = 6000
        self.lr_decay_rate = 0.9
        self.dropout_rate = 0.2
        self.gradient_clip = 5.0
        self.num_epochs = 10
        self.batch_size = 32
        self.param_init = 0.1
        self.seed = 1312002

def get_arg_parser(title):
	parser = argparse.ArgumentParser(description=title)
	parser.add_argument('--cpu', action='store_true', default=False)
	parser.add_argument('--inference', action='store_true', default=False)
	parser.add_argument('--eval', action='store_true', default=True)
	parser.add_argument('--model_dir', type=str, default='./checkpoints/model_0')
	parser.add_argument('--load_model', type=str, default='./checkpoints/model_0/ckpt-00001500')
	parser.add_argument('--num_LSTM_layers', type=int, default=2)
	parser.add_argument('--num_MLP_layers', type=int, default=1)
	parser.add_argument('--LSTM_hidden_size', type=int, default=512)
	parser.add_argument('--MLP_hidden_size', type=int, default=512)
	parser.add_argument('--embedding_size', type=int, default=512)

	parser.add_argument('--keep_last_n', type=int, default=None)
	parser.add_argument('--eval_every_n', type=int, default=1500)
	parser.add_argument('--log_interval', type=int, default=1500)
	parser.add_argument('--log_dir', type=str, default='../logs')
	parser.add_argument('--log_name', type=str, default='model_0.csv')

	parser.add_argument('--max_eval_size', type=int, default=1000)

	data_group = parser.add_argument_group('data')
	data_group.add_argument('--train_dataset', type=str, default='../data/train_plot.json')
	data_group.add_argument('--dev_dataset', type=str, default='../data/dev_plot_hard.json')
	data_group.add_argument('--test_dataset', type=str, default='./data/test_plot.json')
	data_group.add_argument('--code_vocab', type=str, default='./data/code_vocab.json')
	data_group.add_argument('--word_vocab', type=str, default='./data/nl_vocab.json')
	data_group.add_argument('--word_vocab_size', type=int, default=None)
	data_group.add_argument('--code_vocab_size', type=int, default=None)
	data_group.add_argument('--num_plot_types', type=int, default=6)
	data_group.add_argument('--joint_plot_types', action='store_true', default=False)
	data_group.add_argument('--data_order_invariant', action='store_true', default=False)
	data_group.add_argument('--nl', action='store_true', default=True)
	data_group.add_argument('--use_comments', action='store_true', default=True)
	data_group.add_argument('--code_context', action='store_true', default=True)
	data_group.add_argument('--local_df_only', action='store_true', default=False)
	data_group.add_argument('--target_code_transform', action='store_true', default=True)
	data_group.add_argument('--max_num_code_cells', type=int, default=2)
	data_group.add_argument('--max_word_len', type=int, default=512)
	data_group.add_argument('--max_code_context_len', type=int, default=512)
	data_group.add_argument('--max_decode_len', type=int, default=200)

	model_group = parser.add_argument_group('model')
	model_group.add_argument('--hierarchy', action='store_true', default=False)
	model_group.add_argument('--copy_mechanism', action='store_true', default=True)
	model_group.add_argument('--nl_code_linking', action='store_true', default=True)

	train_group = parser.add_argument_group('train')
	train_group.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'])
	train_group.add_argument('--lr', type=float, default=1e-3)
	train_group.add_argument('--lr_decay_steps', type=int, default=6000)
	train_group.add_argument('--lr_decay_rate', type=float, default=0.9)
	train_group.add_argument('--dropout_rate', type=float, default=0.2)
	train_group.add_argument('--gradient_clip', type=float, default=5.0)
	train_group.add_argument('--num_epochs', type=int, default=50)
	train_group.add_argument('--batch_size', type=int, default=32)
	train_group.add_argument('--param_init', type=float, default=0.1)
	train_group.add_argument('--seed', type=int, default=None)

	return parser
