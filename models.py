import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import random
from torch.autograd import Variable as Var
from utils import *
from data import *
from lf_evaluator import *
import numpy as np
from typing import List

def add_models_args(parser):
    """
    Command-line arguments to the system related to your model.  Feel free to extend here.  
    """
    # Some common arguments for your convenience
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=10, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--hidden_size', type=int, default=50, help='hidden size of LSTM')

    # 65 is all you need for GeoQuery
    parser.add_argument('--decoder_len_limit', type=int, default=65, help='output length limit of the decoder')

    # Feel free to add other hyperparameters for your input dimension, etc. to control your network
    # 50-200 might be a good range to start with for embedding and LSTM sizes


class NearestNeighborSemanticParser(object):
    """
    Semantic parser that uses Jaccard similarity to find the most similar input example to a particular question and
    returns the associated logical form.
    """
    def __init__(self, training_data: List[Example]):
        self.training_data = training_data

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        """
        :param test_data: List[Example] to decode
        :return: A list of k-best lists of Derivations. A Derivation consists of the underlying Example, a probability,
        and a tokenized input string. If you're just doing one-best decoding of example ex and you
        produce output y_tok, you can just return the k-best list [Derivation(ex, 1.0, y_tok)]
        """
        test_derivs = []
        for test_ex in test_data:
            test_words = test_ex.x_tok
            best_jaccard = -1
            best_train_ex = None
            # Find the highest word overlap with the train data
            for train_ex in self.training_data:
                # Compute word overlap
                train_words = train_ex.x_tok
                overlap = len(frozenset(train_words) & frozenset(test_words))
                jaccard = overlap/float(len(frozenset(train_words) | frozenset(test_words)))
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_train_ex = train_ex
            # N.B. a list!
            test_derivs.append([Derivation(test_ex, 1.0, best_train_ex.y_tok)])
        return test_derivs


###################################################################################################################
# You do not have to use any of the classes in this file, but they're meant to give you a starting implementation.
# for your network.
###################################################################################################################

class Seq2SeqSemanticParser(nn.Module):
    def __init__(self, input_indexer, output_indexer,
                 in_emb_dim, hidden_size, out_max_len,
                 embedding_dropout=0.2, tf_ratio=0.5, bidirect=True):

        # We've include some args for setting up the input embedding and encoder
        # You'll need to add code for output embedding and decoder
        super(Seq2SeqSemanticParser, self).__init__()
        self.tf_ratio = tf_ratio
        self.input_indexer = input_indexer
        self.output_indexer = output_indexer
        self.out_max_len = out_max_len
        self.pad_idx = self.output_indexer.index_of(PAD_SYMBOL)

        self.input_emb = EmbeddingLayer(in_emb_dim, len(input_indexer), embedding_dropout)
        self.encoder = RNNEncoder(in_emb_dim, hidden_size, bidirect)

        self.output_emb = EmbeddingLayer(hidden_size, len(output_indexer), 0.5)
        self.decoder = RNNDecoder(hidden_size, len(output_indexer))


    def forward(self, x_tensor, inp_lens_tensor, y_tensor):
        """
        :param x_tensor/y_tensor: either a non-batched input/output [sent len x voc size] or a batched input/output
        [batch size x sent len x voc size]
        :param inp_lens_tensor/out_lens_tensor: either a vector of input/output length [batch size] or a single integer.
        lengths aren't needed if you don't batchify the training.
        :return: loss of the batch
        """

        enc_word, enc_mask, enc_h = self._encode_input(x_tensor, inp_lens_tensor)
        init_h = enc_h

        avg_loss, loss = self._decode_output(y_tensor, init_h)

        return avg_loss, loss


    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:


        EOS = self.output_indexer.index_of(EOS_SYMBOL)

        input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in test_data]))
        all_input_test_data = make_padded_input_tensor(test_data, self.input_indexer, input_max_len, reverse_input=False)

        derivations = list()

        # get derivation for each example
        for i in range(all_input_test_data.shape[0]):

            deriv = list()
            x_tensor = torch.tensor(all_input_test_data[i]).reshape(1, -1)
            x_lens = torch.tensor(np.count_nonzero(x_tensor, axis=1))

            # encode each example
            enc_word, enc_mask, enc_h = self._encode_input(x_tensor, x_lens)

            dec_h = enc_h
            y_step = torch.tensor([self.output_indexer.index_of(SOS_SYMBOL)]).reshape(1, -1)

            for t in range(self.out_max_len):

                # embed target token
                out_emb = self.output_emb(y_step)

                # call decoder and get most likely token at each step
                probs, dec_h = self.decoder(out_emb, dec_h)

                # new input is predicted token
                y_step_t = torch.argmax(probs, dim=1).detach()

                if y_step_t.item() == EOS:
                    break

                y_tok = self.output_indexer.get_object(y_step_t.item())

                deriv.append(y_tok)

                y_step = y_step_t.reshape(1, -1)

            deriv = Derivation(test_data[i], 1, deriv)
            derivations.append([deriv])

        return derivations


    def _encode_input(self, x_tensor, inp_lens_tensor):
        """
        Runs the encoder (input embedding layer and encoder as two separate modules) on a tensor of inputs x_tensor with
        inp_lens_tensor lengths.
        YOU DO NOT NEED TO USE THIS FUNCTION. It's merely meant to illustrate the usage of EmbeddingLayer and RNNEncoder
        as they're given to you, as well as show what kinds of inputs/outputs you need from your encoding phase.
        :param x_tensor: [batch size, sent len] tensor of input token indices
        :param inp_lens_tensor: [batch size] vector containing the length of each sentence in the batch
        :param model_input_emb: EmbeddingLayer
        :param model_enc: RNNEncoder
        :return: the encoder outputs (per word), the encoder context mask (matrix of 1s and 0s reflecting which words
        are real and which ones are pad tokens), and the encoder final states (h and c tuple)
        E.g., calling this with x_tensor (0 is pad token):
        [[12, 25, 0, 0],
        [1, 2, 3, 0],
        [2, 0, 0, 0]]
        inp_lens = [2, 3, 1]
        will return outputs with the following shape:
        enc_output_each_word = 3 x 4 x dim, enc_context_mask = [[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]],
        enc_final_states = 3 x dim
        """

        # batch_sz = x_tensor.shape[0]

        # append EOS token to input
        # eos_tensor = torch.tensor([self.input_indexer.index_of(EOS_SYMBOL)] * batch_sz).reshape(batch_sz, 1)
        # x_tensor = torch.cat((x_tensor, eos_tensor), 1)

        input_emb = self.input_emb.forward(x_tensor)
        (enc_output_each_word, enc_context_mask, enc_final_states) = self.encoder.forward(input_emb, inp_lens_tensor)
        enc_final_states_reshaped = (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))
        return (enc_output_each_word, enc_context_mask, enc_final_states_reshaped)

    def _decode_output(self, y_tensor:torch.Tensor, init_h:tuple) -> (torch.Tensor, torch.Tensor):
        """
        passes target through decoder one time step at a time using teacher forcing
        :arg y_tensor: tensor with target indices for decoder
        :arg out_lens_tensor: length of each target sentence
        :arg init_h: initial hidden state of decoder (context vector for encoder)
        :returns: average loss across inputs and total loss across timesteps
        """

        # initial hidden state is context vector of encoder
        dec_hidden = init_h

        loss = 0.0
        batch_sz = y_tensor.shape[0]
        target_len = y_tensor.shape[1]

        # SOS token, initial input
        y_step = torch.tensor([[self.output_indexer.index_of(SOS_SYMBOL)] * batch_sz]).reshape(batch_sz, -1)

        teacher_forcing = True if random.random() < self.tf_ratio else False

        # update ratio
        if teacher_forcing:
            self.tf_ratio -= .05

        # teacher forcing, pass gold target as input for current timestep
        for t in range(target_len):

            out_emb = self.output_emb(y_step)

            probs, dec_hidden = self.decoder(out_emb, dec_hidden)

            target = y_tensor[:, t]
            loss += F.nll_loss(probs, target, ignore_index=self.pad_idx)

            if teacher_forcing:
                # pass target as input
                y_step = target.reshape(batch_sz, 1)

            else:
                # pass prediction as next input
                y_step = torch.argmax(probs, dim=1).detach().reshape(batch_sz, -1)

        # compute mean loss across timesteps
        loss = loss / target_len

        # compute mean loss across the batch
        avg_loss = loss / batch_sz

        return avg_loss, loss


class EmbeddingLayer(nn.Module):
    """
    Embedding layer that has a lookup table of symbols that is [full_dict_size x input_dim]. Includes dropout.
    Works for both non-batched and batched inputs
    """
    def __init__(self, input_dim: int, full_dict_size: int, embedding_dropout_rate: float):
        """
        :param input_dim: dimensionality of the word vectors
        :param full_dict_size: number of words in the vocabulary
        :param embedding_dropout_rate: dropout rate to apply
        """
        super(EmbeddingLayer, self).__init__()
        self.dropout = nn.Dropout(embedding_dropout_rate)
        self.word_embedding = nn.Embedding(full_dict_size, input_dim)

    def forward(self, input):
        """
        :param input: either a non-batched input [sent len x voc size] or a batched input
        [batch size x sent len x voc size]
        :return: embedded form of the input words (last coordinate replaced by input_dim)
        """
        embedded_words = self.word_embedding(input)
        final_embeddings = self.dropout(embedded_words)
        return final_embeddings


class RNNEncoder(nn.Module):
    """
    One-layer RNN encoder for batched inputs -- handles multiple sentences at once. To use in non-batched mode, call it
    with a leading dimension of 1 (i.e., use batch size 1)
    """
    def __init__(self, input_size: int, hidden_size: int, bidirect: bool):
        """
        :param input_size: size of word embeddings output by embedding layer
        :param hidden_size: hidden size for the LSTM
        :param bidirect: True if bidirectional, false otherwise
        """
        super(RNNEncoder, self).__init__()
        self.bidirect = bidirect
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reduce_h_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.reduce_c_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True,
                               dropout=0., bidirectional=self.bidirect)
        self.init_weight()

    def init_weight(self):
        """
        Initializes weight matrices using Xavier initialization
        :return:
        """
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        if self.bidirect:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0_reverse, gain=1)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0_reverse, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
        if self.bidirect:
            nn.init.constant_(self.rnn.bias_hh_l0_reverse, 0)
            nn.init.constant_(self.rnn.bias_ih_l0_reverse, 0)

    def get_output_size(self):
        return self.hidden_size * 2 if self.bidirect else self.hidden_size

    def sent_lens_to_mask(self, lens, max_length):
        return torch.from_numpy(np.asarray([[1 if j < lens.data[i].item() else 0 for j in range(0, max_length)] for i in range(0, lens.shape[0])]))

    def forward(self, embedded_words, input_lens):
        """
        Runs the forward pass of the LSTM
        :param embedded_words: [batch size x sent len x input dim] tensor
        :param input_lens: [batch size]-length vector containing the length of each input sentence
        :return: output (each word's representation), context_mask (a mask of 0s and 1s
        reflecting where the model's output should be considered), and h_t, a *tuple* containing
        the final states h and c from the encoder for each sentence.
        """
        # Takes the embedded sentences, "packs" them into an efficient Pytorch-internal representation
        packed_embedding = nn.utils.rnn.pack_padded_sequence(embedded_words, input_lens, batch_first=True, enforce_sorted=False)
        # Runs the RNN over each sequence. Returns output at each position as well as the last vectors of the RNN
        # state for each sentence (first/last vectors for bidirectional)
        output, hn = self.rnn(packed_embedding)
        # Unpacks the Pytorch representation into normal tensors
        output, sent_lens = nn.utils.rnn.pad_packed_sequence(output)
        max_length = input_lens.data[0].item()
        context_mask = self.sent_lens_to_mask(sent_lens, max_length)

        # Grabs the encoded representations out of hn, which is a weird tuple thing.
        # Note: if you want multiple LSTM layers, you'll need to change this to consult the penultimate layer
        # or gather representations from all layers.
        if self.bidirect:
            h, c = hn[0], hn[1]
            # Grab the representations from forward and backward LSTMs
            h_, c_ = torch.cat((h[0], h[1]), dim=1), torch.cat((c[0], c[1]), dim=1)
            # Reduce them by multiplying by a weight matrix so that the hidden size sent to the decoder is the same
            # as the hidden size in the encoder
            new_h = self.reduce_h_W(h_)
            new_c = self.reduce_c_W(c_)
            h_t = (new_h, new_c)
        else:
            h, c = hn[0][0], hn[1][0]
            h_t = (h, c)
        return (output, context_mask, h_t)
###################################################################################################################
# End optional classes
###################################################################################################################


class RNNDecoder(nn.Module):

    def __init__(self, hidden_sz:int, output_sz:int):
        super(RNNDecoder, self).__init__()

        self.hidden_sz = hidden_sz
        self.output_sz = output_sz
        self.rnn = nn.LSTM(hidden_sz, hidden_sz, batch_first=True)
        self.out = nn.Linear(hidden_sz, output_sz)
        self.softmax = nn.LogSoftmax(dim=1)
        self.init_weight()

    def init_weight(self):
        """
        Initializes weight matrices using Xavier initialization
        :return:
        """

        nn.init.xavier_uniform_(self.out.weight)
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)

        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)


    def forward(self, input, hidden):
        """
        :arg input: embeddings for target language
        :arg input_lens: length of each sentence in input
        :arg hidden: initial hidden state of Decoder (final hidden state of encoder)
        :arg enc_mask: used to accumulate valid loss terms
        :returns: probability distribution over words
        """
        input = F.relu(input)
        output, (h, c) = self.rnn(input, hidden)
        x = self.out(output).reshape(-1, self.output_sz)
        probs = self.softmax(x)
        s = probs.sum(dim=1)
        return probs, (h, c)



def make_padded_input_tensor(exs: List[Example], input_indexer: Indexer, max_len: int, reverse_input=False) -> np.ndarray:
    """
    Takes the given Examples and their input indexer and turns them into a numpy array by padding them out to max_len.
    Optionally reverses them.
    :param exs: examples to tensor-ify
    :param input_indexer: Indexer over input symbols; needed to get the index of the pad symbol
    :param max_len: max input len to use (pad/truncate to this length)
    :param reverse_input: True if we should reverse the inputs (useful if doing a unidirectional LSTM encoder)
    :return: A [num example, max_len]-size array of indices of the input tokens
    """
    if reverse_input:
        return np.array(
            [[ex.x_indexed[len(ex.x_indexed) - 1 - i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
              for i in range(0, max_len)]
             for ex in exs])
    else:
        return np.array([[ex.x_indexed[i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
                          for i in range(0, max_len)]
                         for ex in exs])

def make_padded_output_tensor(exs, output_indexer, max_len):
    """
    Similar to make_padded_input_tensor, but does it on the outputs without the option to reverse input
    :param exs:
    :param output_indexer:
    :param max_len:
    :return: A [num example, max_len]-size array of indices of the output tokens
    """
    return np.array([[ex.y_indexed[i] if i < len(ex.y_indexed) else output_indexer.index_of(PAD_SYMBOL) for i in range(0, max_len)] for ex in exs])


def train_model_encdec(train_data: List[Example], dev_data: List[Example], input_indexer, output_indexer, args) -> Seq2SeqSemanticParser:
    """
    Function to train the encoder-decoder model on the given data.
    :param train_data:
    :param dev_data: Development set in case you wish to evaluate during training
    :param input_indexer: Indexer of input symbols
    :param output_indexer: Indexer of output symbols
    :param args:
    :return:
    """
    # Create indexed input
    input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in train_data]))
    all_train_input_data = make_padded_input_tensor(train_data, input_indexer, input_max_len, reverse_input=False)
    # all_test_input_data = make_padded_input_tensor(dev_data, input_indexer, input_max_len, reverse_input=False)

    output_max_len = np.max(np.asarray([len(ex.y_indexed) for ex in train_data]))
    all_train_output_data = make_padded_output_tensor(train_data, output_indexer, output_max_len)
    # all_test_output_data = make_padded_output_tensor(dev_data, output_indexer, output_max_len)

    if args.print_dataset:
        print("Train length: %i" % input_max_len)
        print("Train output length: %i" % np.max(np.asarray([len(ex.y_indexed) for ex in train_data])))
        print("Train matrix: %s; shape = %s" % (all_train_input_data, all_train_input_data.shape))

    # First create a model. Then loop over epochs, loop over examples, and given some indexed words
    # call your seq-to-seq model, accumulate losses, update parameters

    lr = args.lr
    epochs = args.epochs
    batch_sz = args.batch_size
    hidden_sz = 300

    # instantiate model
    seq2seq = Seq2SeqSemanticParser(input_indexer, output_indexer, input_max_len, hidden_sz, output_max_len)

    encoder_optim = torch.optim.Adam(seq2seq.encoder.parameters(), lr=lr)
    decoder_optim = torch.optim.Adam(seq2seq.decoder.parameters(), lr=lr)

    def _train(seq2seq:Seq2SeqSemanticParser,
               all_train_input_data:np.ndarray,
               all_train_output_data:np.ndarray,
               batch_sz:int) -> float:
        """
        Trains the Seq2Seq model on a batch of input pairs for a single epoch
        :returns: average loss over batched input pairs for a single epoch
        """

        seq2seq.train()

        # shuffle dataset
        idxs = np.arange(all_train_input_data.shape[0])
        np.random.shuffle(idxs)

        avg_loss = 0.0
        n_batches = all_train_input_data.shape[0] // batch_sz

        # train model on batch of input
        for start in range(0, idxs.size, batch_sz):
            end = start + batch_sz
            batch_idxs = idxs[start:end]

            x = torch.tensor(all_train_input_data[batch_idxs])
            y = torch.tensor(all_train_output_data[batch_idxs])

            x_lens = torch.tensor(np.count_nonzero(x, axis=1))

            # zero out previous gradients
            encoder_optim.zero_grad()
            decoder_optim.zero_grad()

            batch_loss, loss = seq2seq(x, x_lens, y)
            avg_loss += batch_loss

            loss.backward()

            encoder_optim.step()
            decoder_optim.step()

        avg_loss /= n_batches

        return avg_loss


    def _test(seq2seq:Seq2SeqSemanticParser,
               all_test_input_data:np.ndarray,
               all_test_output_data:np.ndarray) -> float:
        """
        Evaluates the seq2seq model on test data
        """


    for e in range(epochs):

        avg_loss = _train(seq2seq, all_train_input_data, all_train_output_data, batch_sz)
        # metrics = _test(seq2seq, all_test_input_data, all_test_output_data)

        print(f"===> Epoch: {e}, Avg. Loss: {avg_loss}")

    return seq2seq