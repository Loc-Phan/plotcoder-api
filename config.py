cpu = False
inference = True
eval = False
model_dir = './checkpoints/model_0'
load_model = './checkpoints/model_0/ckpt-00001500'
num_LSTM_layers = 2
num_MLP_layers = 1
LSTM_hidden_size = 512
MLP_hidden_size = 512
embedding_size = 512

keep_last_n = None
eval_every_n = 1500
log_interval = 1500
log_dir = '../logs'
log_name = 'model_0.csv'

max_eval_size = 1000

dev_dataset = '../data/dev_plot_hard.json'
test_dataset = './data/test.json'
code_vocab = './data/code_vocab.json'
word_vocab = './data/nl_vocab.json'
word_vocab_size = None
code_vocab_size = None
num_plot_types = 6
joint_plot_types = False
data_order_invariant = False
nl = True
use_comments = True
code_context = True
local_df_only = False
target_code_transform = True
max_num_code_cells = 2
max_word_len = 512
max_code_context_len = 512
max_decode_len = 200

hierarchy = False
copy_mechanism = True
nl_code_linking = True

optimizer = 'adam'
# choices=['adam', 'sgd', 'rmsprop']
lr = 1e-3
lr_decay_steps = 6000
lr_decay_rate = 0.9
dropout_rate = 0.2
gradient_clip = 5.0
num_epochs = 50
batch_size = 32
param_init = 0.1
seed = None
