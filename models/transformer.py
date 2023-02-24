import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import cuda, Tensor
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
import torch.nn.functional as F
from typing import Optional, Any, Union, Callable
import copy
import math

import numpy as np

# Special vocabulary symbols
_PAD = b"_PAD"
_EOS = b"_EOS"
_GO = b"_GO"
_UNK = b"_UNK"
_DF = b"_DF"
_VAR = b"_VAR"
_STR = b"_STR"
_FUNC = b"_FUNC"
_VALUE = b"_VALUE"
_START_VOCAB = [_PAD, _EOS, _GO, _UNK, _DF, _VAR, _STR, _FUNC, _VALUE]

PAD_ID = 0
EOS_ID = 1
GO_ID = 2
UNK_ID = 3
DF_ID = 4
VAR_ID = 5
STR_ID = 6
FUNC_ID = 7
VALUE_ID = 8

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.2, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]


        return self.dropout(x)
    
def np_to_tensor(inp, output_type, cuda_flag):
	if output_type == 'float':
		inp_tensor = Variable(torch.FloatTensor(inp))
	elif output_type == 'int':
		inp_tensor = Variable(torch.LongTensor(inp))
	else:
		print('undefined tensor type')
	if cuda_flag:
		inp_tensor = inp_tensor
	return inp_tensor

class TransformerDecoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.code_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        self.nl_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm4 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)
    
    def forward(self, tgt: Tensor, code_memory: Tensor, nl_memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._codemha_block(self.norm2(x), code_memory, memory_mask, memory_key_padding_mask)
            x = x + self._nlmha_block(self.norm3(x), nl_memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm4(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._codemha_block(x, code_memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._nlmha_block(x, nl_memory, memory_mask, memory_key_padding_mask))
            x = self.norm4(x + self._ff_block(x))

        return x


    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _nlmha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.nl_multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)
    
    def _codemha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.code_multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout3(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout4(x)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class TransformerDecoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, code_memory: Tensor, nl_memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        
        output = tgt

        for mod in self.layers:
            output = mod(output, code_memory, nl_memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
    
class PlotCodeGeneratorTrans(nn.Module):
	def __init__(self, args, word_vocab, code_vocab):
		super(PlotCodeGeneratorTrans, self).__init__()
		self.cuda_flag = args.cuda
		self.word_vocab_size = args.word_vocab_size
		self.code_vocab_size = args.code_vocab_size
		self.num_plot_types = args.num_plot_types
		self.word_vocab = word_vocab
		self.code_vocab = code_vocab
		self.batch_size = args.batch_size
		self.embedding_size = args.embedding_size
		self.gradient_clip = args.gradient_clip
		self.lr = args.lr
		self.dropout_rate = args.dropout_rate
		self.nl = args.nl
		self.use_comments = args.use_comments
		self.code_context = args.code_context
		self.hierarchy = args.hierarchy
		self.copy_mechanism = args.copy_mechanism
		self.nl_code_linking = args.nl_code_linking
		self.max_word_len = args.max_word_len
		self.max_code_context_len = args.max_code_context_len
		self.max_decode_len = args.max_decode_len
		self.dropout = nn.Dropout(p=self.dropout_rate)
		self.n_heads = args.n_heads
		self.n_layers = args.n_layers
		self.dim_forward = args.dim_forward
        
		self.decoder_code_linear = nn.Linear(self.embedding_size * 2, self.embedding_size)
		self.decoder_copy_linear = nn.Linear(self.embedding_size * 2, self.embedding_size)
		if not self.nl_code_linking:
			self.code_ctx_linear = nn.Linear(self.embedding_size * 2 + self.embedding_size, self.embedding_size)
		else:
			self.code_ctx_word_linear = nn.Linear(self.embedding_size * 3, self.embedding_size)
			self.code_word_linear = nn.Linear(self.embedding_size * 2, self.embedding_size)
		self.word_embedding = nn.Embedding(self.word_vocab_size, self.embedding_size)
		if self.copy_mechanism:
			self.code_embedding = nn.Embedding(self.code_vocab_size, self.embedding_size)
		else:
			self.code_embedding = nn.Embedding(self.code_vocab_size + self.max_code_context_len, self.embedding_size)
			self.code_predictor = nn.Linear(self.embedding_size, self.code_vocab_size + self.max_code_context_len)
			self.copy_predictor = nn.Linear(self.embedding_size, self.code_vocab_size + self.max_code_context_len)
		self.norm_layer = nn.LayerNorm(self.embedding_size)
		self.norm_layer1 = nn.LayerNorm(self.embedding_size)
		encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=self.n_heads, dim_feedforward=self.dim_forward, activation="relu", batch_first=True, dropout=self.dropout_rate, norm_first=True)
		self.input_nl_encoder = nn.TransformerEncoder(encoder_layer = encoder_layer, norm = self.norm_layer ,num_layers=self.n_layers)
		encoder_layer1 = nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=self.n_heads, dim_feedforward=self.dim_forward,  activation="relu", batch_first=True, dropout=self.dropout_rate, norm_first= True)
		self.input_code_encoder = nn.TransformerEncoder(encoder_layer = encoder_layer1, norm = self.norm_layer1, num_layers=self.n_layers)
		self.code_word_linear = nn.Linear(self.embedding_size * 2, self.embedding_size)
		if self.hierarchy:
			self.norm_layer2 = nn.LayerNorm(self.embedding_size * 2)
			decoder_layer = TransformerDecoderLayer(d_model=self.embedding_size * 2, nhead=self.n_heads, dim_feedforward=self.dim_forward,  activation="relu", batch_first=True, dropout=self.dropout_rate, norm_first= True)
			self.decoder = TransformerDecoder(decoder_layer = decoder_layer, norm = self.norm_layer2,num_layers=self.n_layers)
# 		self.code_encoder_output_linear = nn.Linear(self.embedding_size, self.embedding_size * 2)
# 		self.nl_encoder_output_linear = nn.Linear(self.embedding_size, self.embedding_size * 2)
		self.position_encoding = PositionalEncoding(d_model = self.embedding_size, max_len = self.max_code_context_len)
		self.loss = nn.CrossEntropyLoss()
		if args.optimizer == 'adam':
			self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
		elif args.optimizer == 'sgd':
			self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
		elif args.optimizer == 'rmsprop':
			self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr)
		else:
			raise ValueError('optimizer undefined: ', args.optimizer)
		self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, total_iters = args.lr_decay_steps)

	def init_weights(self, param_init):
		for param in self.parameters():
			nn.init.uniform_(param, -param_init, param_init)

	def lr_decay(self, lr_decay_rate):
		self.lr *= lr_decay_rate
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = self.lr

	def train_step(self):
		if self.gradient_clip > 0:
			clip_grad_norm(self.parameters(), self.gradient_clip)
		self.optimizer.step()
		self.scheduler.step()


	def forward(self, batch_input, batch_labels, eval_flag=False):
		batch_size = batch_labels.size()[0]
		batch_init_data = batch_input['init_data']
		batch_nl_input = batch_input['nl']
		batch_nl_embedding = self.word_embedding(batch_nl_input) 
		batch_nl_pos = self.position_encoding(batch_nl_embedding * math.sqrt(self.embedding_size))
		encoder_word_mask = (batch_nl_input == PAD_ID) 
		encoder_word_mask = torch.max(encoder_word_mask, (batch_nl_input == UNK_ID))
# 		encoder_word_mask = torch.max(encoder_word_mask, (batch_nl_input == EOS_ID))
		if self.cuda_flag:
			encoder_word_mask = encoder_word_mask
		nl_encoder_output = self.input_nl_encoder(batch_nl_pos, src_key_padding_mask = encoder_word_mask)
		batch_code_context_input = batch_input['code_context']
		encoder_code_mask = (batch_code_context_input == PAD_ID)
		encoder_code_mask = torch.max(encoder_code_mask, (batch_code_context_input == UNK_ID))
# 		encoder_code_mask = torch.max(encoder_code_mask, (batch_code_context_input == EOS_ID))
		batch_code_context_embedding = self.code_embedding(batch_code_context_input)
		batch_code_nl_embedding = []
		batch_input_code_nl_indices = batch_input['input_code_nl_indices']
		max_code_len = batch_code_context_input.size()[1]
		max_word_len = batch_nl_input.size()[1]
		for batch_idx in range(batch_size):
			input_code_nl_indices = batch_input_code_nl_indices[batch_idx, :, :]
			cur_code_nl_embedding_0 = nl_encoder_output[batch_idx, input_code_nl_indices[:, 0]]
			cur_code_nl_embedding_1 = nl_encoder_output[batch_idx, input_code_nl_indices[:, 1]]
			cur_code_nl_embedding = cur_code_nl_embedding_0 + cur_code_nl_embedding_1
			batch_code_nl_embedding.append(cur_code_nl_embedding)
		batch_code_nl_embedding = torch.stack(batch_code_nl_embedding, dim=0)
		code_encoder_input = torch.cat([batch_code_context_embedding, batch_code_nl_embedding], dim=-1)
		code_encoder_input = self.code_word_linear(code_encoder_input)
		code_encoder_input = self.position_encoding(code_encoder_input * math.sqrt(self.embedding_size))
		if self.cuda_flag:
			encoder_code_mask = encoder_code_mask
		code_encoder_output = self.input_code_encoder(code_encoder_input, src_key_padding_mask = encoder_code_mask)

		gt_output = batch_input['gt']
		target_code_output = batch_input['code_output']
		target_df_output = batch_input['df_output']
		target_var_output = batch_input['var_output']
		target_str_output = batch_input['str_output']
		code_output_mask = batch_input['code_output_mask']
		output_df_mask = batch_input['output_df_mask']
		output_var_mask = batch_input['output_var_mask']
		output_str_mask = batch_input['output_str_mask']

		
		gt_decode_length = target_code_output.size()[1]
		if not eval_flag:
			decode_length = gt_decode_length
		else:
			decode_length = self.max_decode_len

		decoder_input_sketch = torch.ones(batch_size, 1, dtype=torch.int64) * GO_ID
		if self.cuda_flag:
			decoder_input_sketch = decoder_input_sketch
		decoder_input_sketch_embedding = self.code_embedding(decoder_input_sketch)  * math.sqrt(self.embedding_size)
		decoder_input = torch.ones(batch_size, 1, dtype=torch.int64) * GO_ID
		if self.cuda_flag:
			decoder_input = decoder_input
		decoder_input_embedding = self.code_embedding(decoder_input)  * math.sqrt(self.embedding_size)
		finished = torch.zeros(batch_size, 1, dtype=torch.int64)

		max_code_mask_len = code_output_mask.size()[1]

		pad_mask = torch.zeros(max_code_mask_len)
		pad_mask[PAD_ID] = 1e9
		pad_mask = torch.stack([pad_mask] * batch_size, dim=0)
		if self.cuda_flag:
			finished = finished
			pad_mask = pad_mask

		batch_code_output_indices = np_to_tensor(np.array(list(range(self.code_vocab_size))), 'int', self.cuda_flag)
		batch_code_output_embedding = self.code_embedding(batch_code_output_indices)
		batch_code_output_embedding = torch.stack([batch_code_output_embedding] * batch_size, dim=0)

		batch_output_code_ctx_embedding = []
		batch_output_code_ctx_indices = batch_input['output_code_ctx_indices']
		for batch_idx in range(batch_size):
			output_code_ctx_indices = batch_output_code_ctx_indices[batch_idx]
			cur_output_code_ctx_embedding = code_encoder_output[batch_idx, output_code_ctx_indices]
			batch_output_code_ctx_embedding.append(cur_output_code_ctx_embedding)
		batch_output_code_ctx_embedding = torch.stack(batch_output_code_ctx_embedding, dim=0)

		if self.nl_code_linking:
			batch_output_code_nl_embedding = []
			batch_output_code_nl_indices = batch_input['output_code_nl_indices']
			for batch_idx in range(batch_size):
				output_code_nl_indices = batch_output_code_nl_indices[batch_idx, :, :]
				cur_output_code_nl_embedding_0 = nl_encoder_output[batch_idx, output_code_nl_indices[:, 0]]
				cur_output_code_nl_embedding_1 = nl_encoder_output[batch_idx, output_code_nl_indices[:, 1]]
				cur_output_code_nl_embedding = cur_output_code_nl_embedding_0 + cur_output_code_nl_embedding_1
				batch_output_code_nl_embedding.append(cur_output_code_nl_embedding)
			batch_output_code_nl_embedding = torch.stack(batch_output_code_nl_embedding, dim=0)
			batch_code_output_embedding = torch.cat([batch_code_output_embedding, batch_output_code_ctx_embedding, batch_output_code_nl_embedding], dim=-1)
			batch_code_output_embedding = self.code_ctx_word_linear(batch_code_output_embedding)
		else:
			batch_code_output_embedding = torch.cat([batch_code_output_embedding, batch_output_code_ctx_embedding], dim=-1)
			batch_code_output_embedding = self.code_ctx_linear(batch_code_output_embedding)				
		if self.code_context:
			batch_code_output_context_embedding = []

			for batch_idx in range(batch_size):
				output_code_indices = batch_init_data[batch_idx]['output_code_indices']
				cur_code_output_context_embedding = []
				for code_idx in output_code_indices:
					cur_code_output_context_embedding.append(code_encoder_output[batch_idx, code_idx, :])
				if len(cur_code_output_context_embedding) < max_code_mask_len - self.code_vocab_size:
					cur_code_output_context_embedding += [np_to_tensor(np.zeros(self.embedding_size), 'float', self.cuda_flag)] * (max_code_mask_len - self.code_vocab_size - len(cur_code_output_context_embedding))
				cur_code_output_context_embedding = torch.stack(cur_code_output_context_embedding, dim=0)
				batch_code_output_context_embedding.append(cur_code_output_context_embedding)
			batch_code_output_context_embedding = torch.stack(batch_code_output_context_embedding, dim=0)
			batch_code_output_context_embedding = batch_code_output_context_embedding
			batch_code_output_embedding = torch.cat([batch_code_output_embedding, batch_code_output_context_embedding], dim=1)
		code_pred_logits = []
		code_predictions = []
		df_pred_logits = []
		df_predictions = []
		var_pred_logits = []
		var_predictions = []
		str_pred_logits = []
		str_predictions = []
		predictions = []
		decoder_position = self.position_encoding(torch.zeros(batch_size, decode_length + 1, self.embedding_size))       
		cur_decoder_position = decoder_position[:,0,:].unsqueeze(1)
		decoder_input_step = torch.cat([decoder_input_sketch_embedding + cur_decoder_position
					, decoder_input_embedding + cur_decoder_position], dim = -1)
# 		code_encoder_output = self.code_encoder_output_linear(code_encoder_output)
# 		nl_encoder_output = self.nl_encoder_output_linear(nl_encoder_output)
		code_encoder_output =  torch.cat([code_encoder_output, code_encoder_output], dim=-1)
		nl_encoder_output =  torch.cat([nl_encoder_output, nl_encoder_output], dim=-1)
		if eval_flag:
			for step in range(decode_length):
				if self.hierarchy:
					decoder_output = self.decoder(decoder_input_step, code_encoder_output, nl_encoder_output)
				else:
					decoder_output = self.decoder(decoder_input_embedding, code_encoder_output, nl_encoder_output)
				decoder_output = decoder_output[:,-1,:]

				decoder_code_output = self.decoder_code_linear(decoder_output)
				if self.hierarchy:
					decoder_copy_output = self.decoder_copy_linear(decoder_output)

				if self.copy_mechanism:
					cur_code_pred_logits = torch.bmm(batch_code_output_embedding, decoder_code_output.unsqueeze(2))
					cur_code_pred_logits = cur_code_pred_logits.squeeze(-1)
				else:
					cur_code_pred_logits = self.code_predictor(decoder_code_output)

				cur_code_pred_logits = cur_code_pred_logits + finished.float() * pad_mask
				cur_code_pred_logits = cur_code_pred_logits - (1.0 - code_output_mask) * 1e9
				cur_code_predictions = cur_code_pred_logits.max(1)[1]

				if eval_flag:
					sketch_predictions = cur_code_predictions
				else:
					sketch_predictions = target_code_output[:, step]

				if self.hierarchy:
					if self.copy_mechanism:
						cur_copy_pred_logits = torch.bmm(batch_code_output_embedding, decoder_copy_output.unsqueeze(2))
						cur_copy_pred_logits = cur_copy_pred_logits.squeeze(-1)
					else:
						cur_copy_pred_logits = self.copy_predictor(decoder_copy_output)
					cur_df_pred_logits = cur_copy_pred_logits - (1.0 - output_df_mask) * 1e9
					cur_df_predictions = cur_df_pred_logits.max(1)[1] * ((sketch_predictions == DF_ID).long())

					cur_var_pred_logits = cur_copy_pred_logits - (1.0 - output_var_mask) * 1e9
					cur_var_predictions = cur_var_pred_logits.max(1)[1] * ((sketch_predictions == VAR_ID).long())

					cur_str_pred_logits = cur_copy_pred_logits - (1.0 - output_str_mask) * 1e9
					cur_str_predictions = cur_str_pred_logits.max(1)[1] * ((sketch_predictions == STR_ID).long())

				if eval_flag:
					decoder_input_sketch = cur_code_predictions
					decoder_input = cur_code_predictions
					if self.hierarchy:
						decoder_input = torch.max(decoder_input, cur_df_predictions)
						decoder_input = torch.max(decoder_input, cur_var_predictions)
						decoder_input = torch.max(decoder_input, cur_str_predictions)
				else:
					decoder_input_sketch = target_code_output[:, step]
					decoder_input = gt_output[:, step]
				if self.copy_mechanism:
					decoder_input_sketch_embedding = []
					for batch_idx in range(batch_size):
						decoder_input_sketch_embedding.append(batch_code_output_embedding[batch_idx, decoder_input_sketch[batch_idx], :])
					decoder_input_sketch_embedding = torch.stack(decoder_input_sketch_embedding, dim=0) * math.sqrt(self.embedding_size)

					decoder_input_embedding = []
					for batch_idx in range(batch_size):
						decoder_input_embedding.append(batch_code_output_embedding[batch_idx, decoder_input[batch_idx], :])
					decoder_input_embedding = torch.stack(decoder_input_embedding, dim=0) * math.sqrt(self.embedding_size)
				else:
					decoder_input_sketch_embedding = self.code_embedding(decoder_input_sketch) * math.sqrt(self.embedding_size)
					decoder_input_embedding = self.code_embedding(decoder_input) * math.sqrt(self.embedding_size)
				cur_decoder_position = decoder_position[:,step+1,:]
				decoder_input_embedding = decoder_input_embedding + cur_decoder_position
				decoder_input_sketch_embedding = decoder_input_sketch_embedding + cur_decoder_position
				decoder_input_sketch_embedding = decoder_input_sketch_embedding.unsqueeze(1)
				decoder_input_embedding = decoder_input_embedding.unsqueeze(1)
				cur_decoder_input_step = torch.cat([decoder_input_sketch_embedding, decoder_input_embedding], dim = -1)
				decoder_input_step = torch.cat([decoder_input_step,cur_decoder_input_step], dim = 1)
				if step < gt_decode_length:
					code_pred_logits.append(cur_code_pred_logits)
				code_predictions.append(cur_code_predictions)
				cur_predictions = cur_code_predictions
				if self.hierarchy:
					if step < gt_decode_length:
						df_pred_logits.append(cur_df_pred_logits)
						var_pred_logits.append(cur_var_pred_logits)
						str_pred_logits.append(cur_str_pred_logits)
					df_predictions.append(cur_df_predictions)
					var_predictions.append(cur_var_predictions)
					str_predictions.append(cur_str_predictions)
					cur_predictions = torch.max(cur_predictions, cur_df_predictions)
					cur_predictions = torch.max(cur_predictions, cur_var_predictions)
					cur_predictions = torch.max(cur_predictions, cur_str_predictions)
				predictions.append(cur_predictions)
				cur_finished = (decoder_input == EOS_ID).long().unsqueeze(1)
				finished = torch.max(finished, cur_finished)
				if torch.sum(finished) == batch_size and step >= gt_decode_length - 1:
					break
                    
			total_loss = 0.0
			code_pred_logits = torch.stack(code_pred_logits, dim=0)
			code_pred_logits = code_pred_logits.permute(1, 2, 0)
			code_predictions = torch.stack(code_predictions, dim=0)
			code_predictions = code_predictions.permute(1, 0)
			total_loss += F.cross_entropy(code_pred_logits, target_code_output, ignore_index=PAD_ID)

			if self.hierarchy:
				df_pred_logits = torch.stack(df_pred_logits, dim=0)
				df_pred_logits = df_pred_logits.permute(1, 2, 0)
				df_predictions = torch.stack(df_predictions, dim=0)
				df_predictions = df_predictions.permute(1, 0)
				df_loss = F.cross_entropy(df_pred_logits, target_df_output, ignore_index=-1)

				var_pred_logits = torch.stack(var_pred_logits, dim=0)
				var_pred_logits = var_pred_logits.permute(1, 2, 0)
				var_predictions = torch.stack(var_predictions, dim=0)
				var_predictions = var_predictions.permute(1, 0)
				var_loss = F.cross_entropy(var_pred_logits, target_var_output, ignore_index=-1)

				str_pred_logits = torch.stack(str_pred_logits, dim=0)
				str_pred_logits = str_pred_logits.permute(1, 2, 0)
				str_predictions = torch.stack(str_predictions, dim=0)
				str_predictions = str_predictions.permute(1, 0)
				str_loss = F.cross_entropy(str_pred_logits, target_str_output, ignore_index=-1)
				total_loss += (df_loss + var_loss + str_loss) / 3.0

			predictions = torch.stack(predictions, dim=0)
			predictions = predictions.permute(1, 0)
		else:
			code_output_mask = code_output_mask.unsqueeze(1)
			output_df_mask = output_df_mask.unsqueeze(1)
			output_var_mask = output_var_mask.unsqueeze(1)
			output_str_mask = output_str_mask.unsqueeze(1)
			pad_mask = pad_mask.unsqueeze(1)
			start_token = torch.ones(batch_size, 1, dtype=torch.int64) * GO_ID
			encoder_word_mask = encoder_word_mask.unsqueeze(1).repeat(1,decode_length,1)
			finished = (gt_output == EOS_ID)
			indices = finished.nonzero(as_tuple=True)[1].repeat_interleave(decode_length)
			mask_output_padding = torch.arange(0, decode_length).repeat(batch_size,1).flatten()
			if self.cuda_flag:
				mask_output_padding = mask_output_padding
				indices = indices
				start_token = start_token
			finished = (mask_output_padding > indices).view(batch_size, decode_length)
			finished = finished.unsqueeze(-1)
			decoder_mask = generate_square_subsequent_mask(decode_length)
			target_code_input = torch.cat([start_token, target_code_output[:,:-1]], dim = -1)
			gt_input = torch.cat([start_token, gt_output[:,:-1]], dim = -1)
			if self.copy_mechanism:
				decoder_input_sketch_embedding = []
				for batch_idx in range(batch_size):
					decoder_input_sketch_embedding.append(batch_code_output_embedding[batch_idx, target_code_input[batch_idx], :])
				decoder_input_sketch_embedding = torch.stack(decoder_input_sketch_embedding, dim=0) * math.sqrt(self.embedding_size)
				decoder_input_embedding = []
				for batch_idx in range(batch_size):
					decoder_input_embedding.append(batch_code_output_embedding[batch_idx, gt_input[batch_idx], :])
				decoder_input_embedding = torch.stack(decoder_input_embedding, dim=0) * math.sqrt(self.embedding_size)
			else:
				decoder_input_sketch_embedding = self.code_embedding(target_code_input) * math.sqrt(self.embedding_size)
				decoder_input_embedding = self.code_embedding(gt_input) * math.sqrt(self.embedding_size)
			decoder_input_sketch_embedding = self.position_encoding(decoder_input_sketch_embedding)
			decoder_input_embedding = self.position_encoding(decoder_input_embedding) 
			if self.cuda_flag:
				decoder_mask = decoder_mask
				decoder_input_sketch_embedding = decoder_input_sketch_embedding
				decoder_input_embedding = decoder_input_embedding
			if self.hierarchy:
				decoder_output = self.decoder(torch.cat([decoder_input_sketch_embedding , decoder_input_embedding], dim = -1), 
				code_memory = code_encoder_output, nl_memory = nl_encoder_output, tgt_mask = decoder_mask)
			else:
				decoder_output = self.decoder(decoder_input_embedding, code_encoder_output, nl_encoder_output, tgt_mask = decoder_mask)

			decoder_code_output = self.decoder_code_linear(decoder_output)
			if self.hierarchy:
				decoder_copy_output = self.decoder_copy_linear(decoder_output)

			if self.copy_mechanism:
				batch_code_output_embedding = batch_code_output_embedding.permute(0,2,1)
				code_pred_logits = torch.bmm(decoder_code_output, batch_code_output_embedding)
			else:
				code_pred_logits = self.code_predictor(decoder_code_output)
			code_pred_logits = code_pred_logits + finished.float() * pad_mask
			code_pred_logits = code_pred_logits - (1.0 - code_output_mask) * 1e9
			code_predictions = code_pred_logits.max(dim = 2)[1]
			sketch_predictions = target_code_output 
			if self.hierarchy:
				if self.copy_mechanism:
					copy_pred_logits = torch.bmm(decoder_copy_output, batch_code_output_embedding)
				else:
					copy_pred_logits = self.copy_predictor(decoder_copy_output)
				df_pred_logits = copy_pred_logits - (1.0 - output_df_mask) * 1e9
				df_predictions = df_pred_logits.max(dim = 2)[1] * ((sketch_predictions == DF_ID).long())

				var_pred_logits = copy_pred_logits - (1.0 - output_var_mask) * 1e9
				var_predictions = var_pred_logits.max(dim = 2)[1] * ((sketch_predictions == VAR_ID).long())

				str_pred_logits = copy_pred_logits - (1.0 - output_str_mask) * 1e9
				str_predictions = str_pred_logits.max(dim = 2)[1] * ((sketch_predictions == STR_ID).long())
			predictions = code_predictions
			if self.hierarchy:
				predictions = torch.maximum(predictions, df_predictions)
				predictions = torch.maximum(predictions, var_predictions)
				predictions = torch.maximum(predictions, str_predictions)
			total_loss = 0.0
			code_pred_logits = code_pred_logits.permute(0, 2, 1)
			total_loss += F.cross_entropy(code_pred_logits, target_code_output, ignore_index=PAD_ID)

			if self.hierarchy:
				df_pred_logits = df_pred_logits.permute(0, 2, 1)
				df_loss = F.cross_entropy(df_pred_logits, target_df_output, ignore_index=-1)

				var_pred_logits = var_pred_logits.permute(0, 2, 1)
				var_loss = F.cross_entropy(var_pred_logits, target_var_output, ignore_index=-1)

				str_pred_logits = str_pred_logits.permute(0, 2, 1)
				str_loss = F.cross_entropy(str_pred_logits, target_str_output, ignore_index=-1)
				total_loss += (df_loss + var_loss + str_loss) / 3.0
		return total_loss, code_pred_logits, predictions

    

def generate_square_subsequent_mask(sz):
	return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)