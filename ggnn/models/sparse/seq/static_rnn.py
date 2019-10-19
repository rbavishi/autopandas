import collections
import functools
import itertools
import multiprocessing
import operator
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
import yaml
from typing import List

from autopandas.utils.io import IndexedFileReader
from ggnn.models.sparse.seq.base import BaseGGNNSeq


class LSTMDecoder(object):
    def __init__(self, in_size, out_size, hid_size, depth, dropout_keep_prob):
        self.in_size = in_size
        self.out_size = out_size
        self.hid_size = hid_size
        self.depth = depth
        self.dropout_keep_prob = dropout_keep_prob

        self.inp_proj = self.create_inp_proj()
        self.decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hid_size,
                                                    num_proj=self.out_size,
                                                    name='LSTM_cell',
                                                    dtype=tf.float32)

    def create_inp_proj(self):
        inp_proj_params = None
        if self.in_size != self.hid_size:
            weights = tf.Variable(self.init_weights([self.in_size, self.hid_size]),
                                  name='decoder_proj_W')
            biases = tf.Variable(np.zeros(self.hid_size), name='decoder_proj_b')
            inp_proj_params = {
                "weights": weights,
                "biases": biases
            }

        return inp_proj_params

    @staticmethod
    def init_weights(shape):
        return np.sqrt(6.0 / (shape[-2] + shape[-1])) * (2 * np.random.rand(*shape).astype(np.float32) - 1)

    def __call__(self, inputs, lengths):
        #  Shape of inputs is assumed to be [batch_size, max_timesteps, hidden_node_size]
        acts = inputs
        if self.inp_proj is not None:
            W = self.inp_proj["weights"]
            b = self.inp_proj["biases"]
            hid = tf.matmul(acts, tf.nn.dropout(W, self.dropout_keep_prob)) + b
            acts = tf.nn.relu(hid)

        rnn_inp = [acts] + [tf.zeros(tf.shape(acts)) for _ in range(self.depth - 1)]
        rnn_out = tf.nn.static_rnn(self.decoder_cell, rnn_inp, sequence_length=lengths, dtype=tf.float32)[0]

        if self.depth == 1:
            rnn_out = rnn_out[0]
        else:
            rnn_out = tf.stack(rnn_out)

        return rnn_out


class GGNNSeqStaticRNN(BaseGGNNSeq):
    def process_raw_graph(self, graph):
        (adjacency_lists, num_incoming_edge_per_type) = self.graph_to_adjacency_lists(graph['edges'])
        label = graph.get('label_seq', [graph.get('label', 0)] * self.params['max_depth'])
        #  Expand the label to max_depth, and add a terminal token
        terminal_token = self.params['num_classes']  # labels are 0-indexed
        req_depth = self.params['max_depth'] + 1  # including the terminal token
        label += [terminal_token]
        depth = len(label)
        label = np.pad(label, (0, req_depth - depth), mode='constant', constant_values=0)

        return {"adjacency_lists": adjacency_lists,
                "num_incoming_edge_per_type": num_incoming_edge_per_type,
                "init": self.to_one_hot(graph["node_features"], self.params['annotation_size']),
                "label": label,
                "label_len": depth}

    def define_placeholders(self):
        super().define_placeholders()
        self.placeholders['target_values'] = tf.placeholder(tf.int64, [None, None], name='target_values')
        self.placeholders['target_lengths'] = tf.placeholder(tf.int64, [None], name="target_lengths")
        self.placeholders['target_masks'] = tf.placeholder(tf.float32, [None, None], name="target_lengths")

    def prepare_final_layer(self):
        #  By default, pools up the node-embeddings (sum by default),
        #  and applies a simple MLP
        pooled = self.perform_pooling(self.ops['final_node_representations'])
        self.ops['final_predictor'] = self.final_predictor()
        return self.ops['final_predictor'](pooled, self.placeholders['target_lengths'])

    def final_predictor(self):
        return LSTMDecoder(self.params['hidden_size_node'], self.params['num_classes'] + 1,
                           self.params['hidden_size_final_mlp'], self.params['max_depth'] + 1,
                           self.placeholders['out_layer_dropout_keep_prob'])

    def make_model(self, mode):
        self.define_placeholders()

        #  First, compute the node-level representations, after the message-passing algorithm
        with tf.variable_scope("graph_model"):
            self.prepare_specific_graph_model()
            self.ops['final_node_representations'] = self.compute_final_node_representations()

        with tf.variable_scope("out_layer"):
            # Should return logits with dimension equal to the number of output classes
            logits = self.prepare_final_layer()  # shape is [max_depth, batch_size, num_classes]
            labels = self.placeholders['target_values']  # shape is [max_depth, batch_size]
            mask = self.placeholders['target_masks']  # shape is [max_depth, batch_size]

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss *= mask

            self.ops['loss'] = tf.reduce_mean(loss)
            probabilities = tf.nn.softmax(logits)  # shape is [max_depth, batch_size, num_classes]
            predictions = tf.argmax(probabilities, -1)  # [max_depth, batch_size]
            predictions *= tf.cast(mask, "int64")  # works because target is padded with zeroes

            correct_prediction = tf.equal(predictions, labels)  # [max_depth, batch_size]
            correct_prediction = tf.reduce_all(correct_prediction, axis=0)  # shape is [batch_size]
            self.ops['accuracy_task'] = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            top_k = tf.nn.top_k(probabilities, k=self.params['num_classes'] + 1)  # including the terminal token
            #  Shape is [batch_size, num_classes, time-steps] after transposing
            self.ops['preds'] = tf.transpose(top_k.indices, [1, 2, 0])
            self.ops['probs'] = tf.transpose(top_k.values, [1, 2, 0])
            self.ops['targets'] = tf.transpose(self.placeholders['target_values'], [1, 0])  # [batch_size, time-steps]

    def make_minibatch_iterator(self, data, is_training: bool):
        """Create minibatches by flattening adjacency matrices into a single adjacency matrix with
        multiple disconnected components."""
        if is_training:
            if isinstance(data, IndexedFileReader):
                data.shuffle()
            else:
                np.random.shuffle(data)

        # Pack until we cannot fit more graphs in the batch
        state_dropout_keep_prob = self.params['graph_state_dropout_keep_prob'] if is_training else 1.
        edge_weights_dropout_keep_prob = self.params['edge_weight_dropout_keep_prob'] if is_training else 1.
        num_graphs = 0

        while num_graphs < len(data):
            num_graphs_in_batch = 0
            batch_node_features = []
            batch_target_task_values = []
            batch_target_lengths = []
            batch_target_masks = []
            batch_adjacency_lists = [[] for _ in range(self.params['num_edge_types'])]
            batch_num_incoming_edges_per_type = []
            batch_graph_nodes_list = []
            node_offset = 0

            while num_graphs < len(data) and node_offset + len(data[num_graphs]['init']) < self.params['batch_size']:
                cur_graph = data[num_graphs]
                num_nodes_in_graph = len(cur_graph['init'])
                padded_features = np.pad(cur_graph['init'],
                                         (
                                             (0, 0),
                                             (0, self.params['hidden_size_node'] - self.params['annotation_size'])),
                                         'constant')
                batch_node_features.extend(padded_features)
                batch_graph_nodes_list.append(
                    np.full(shape=[num_nodes_in_graph], fill_value=num_graphs_in_batch, dtype=np.int32))
                for i in range(self.params['num_edge_types']):
                    if i in cur_graph['adjacency_lists']:
                        batch_adjacency_lists[i].append(cur_graph['adjacency_lists'][i] + node_offset)

                # Turn counters for incoming edges into np array:
                num_incoming_edges_per_type = np.zeros((num_nodes_in_graph, self.params['num_edge_types']))
                for (e_type, num_incoming_edges_per_type_dict) in cur_graph['num_incoming_edge_per_type'].items():
                    for (node_id, edge_count) in num_incoming_edges_per_type_dict.items():
                        num_incoming_edges_per_type[node_id, e_type] = edge_count
                batch_num_incoming_edges_per_type.append(num_incoming_edges_per_type)

                batch_target_task_values.append(cur_graph['label'])
                batch_target_lengths.append(cur_graph['label_len'])
                mask = [1] * cur_graph['label_len'] + [0] * (len(cur_graph['label']) - cur_graph['label_len'])
                batch_target_masks.append(mask)
                num_graphs += 1
                num_graphs_in_batch += 1
                node_offset += num_nodes_in_graph

            #  Shape is now [max_depth, batch_size]
            batch_target_task_values = np.transpose(batch_target_task_values, [1, 0])
            batch_target_masks = np.transpose(batch_target_masks, [1, 0])

            batch_feed_dict = {
                self.placeholders['initial_node_representation']: np.array(batch_node_features),
                self.placeholders['num_incoming_edges_per_type']: np.concatenate(batch_num_incoming_edges_per_type,
                                                                                 axis=0),
                self.placeholders['graph_nodes_list']: np.concatenate(batch_graph_nodes_list),
                self.placeholders['target_values']: batch_target_task_values,
                self.placeholders['target_lengths']: batch_target_lengths,
                self.placeholders['target_masks']: batch_target_masks,
                self.placeholders['num_graphs']: num_graphs_in_batch,
                self.placeholders['graph_state_keep_prob']: state_dropout_keep_prob,
                self.placeholders['edge_weight_dropout_keep_prob']: edge_weights_dropout_keep_prob
            }

            # Merge adjacency lists and information about incoming nodes:
            for i in range(self.params['num_edge_types']):
                if len(batch_adjacency_lists[i]) > 0:
                    adj_list = np.concatenate(batch_adjacency_lists[i])
                else:
                    adj_list = np.zeros((0, 2), dtype=np.int32)
                batch_feed_dict[self.placeholders['adjacency_lists'][i]] = adj_list

            yield batch_feed_dict

    @staticmethod
    def top_k_prod(vals, k):
        #  vals has to be of shape [depth, num_vals, 2]
        #  complexity of this procedure is O(depth * k^2)
        depth = len(vals)
        worklist = [[[i[0]], i[1], [i[1]]] for i in vals[0]]
        for d in range(1, depth):
            layer = vals[d]
            new_vals = []
            for i in layer:
                for j in worklist:
                    new_vals.append([j[0] + [i[0]], j[1] * i[1], j[2] + [i[1]]])

            worklist = list(reversed(sorted(new_vals, key=lambda x: x[1])))[:k]

        return worklist

    def get_top_k(self, pred, k=10):
        #  first_k is the top-k preds for each time-step (shape is [depth, top_k, 2])
        #  and 2 is because of the fact that entries are of the form (function_name, probability)
        first_k = list(map(lambda x: list(zip(*x)), pred[:k]))
        first_k = np.transpose(first_k, [1, 0, 2])

        #  Now pick top-k sequences with the highest joint probability (probabilities across timesteps multiplied)
        return self.top_k_prod(first_k, k)
        # products = list(itertools.product(*first_k))
        #
        # #  key is just the product of probabilities
        # def sort_key(x):
        #     prod = 1
        #     for entry in x:
        #         prod *= float(entry[1])
        #
        #     return prod
        #
        # sorted_products = sorted(products, key=lambda x: -sort_key(x))[:k]  # shape is [top_k, depth, 2]
        # sorted_products = np.transpose(sorted_products, [0, 2, 1])
        # #  Compute the product of probabilities
        # return [[i[0].tolist(),
        #          i[1], functools.reduce(operator.mul, map(float, i[1]))] for i in sorted_products]

    def interpret_pred(self, pred: List[int]):
        terminal_token = self.params['num_classes']
        try:
            terminal_pos = pred.index(terminal_token)
        except ValueError:
            terminal_pos = -1

        return ":".join(list(map(lambda x: self.label_mapping[x], pred[:terminal_pos])))

    def infer_top_k(self, pred, k=10):
        # pred is of shape [_, 2, depth]
        pred_top_k = self.get_top_k(pred, k)
        terminal_token = self.params['num_classes']
        for idx, pred in enumerate(pred_top_k):
            pred[0] = self.interpret_pred(pred[0])

        return pred_top_k

    def mprocess(self, arg):
        pred, k = arg
        return self.infer_top_k(pred, k)

    def perform_analysis(self, preds, targets):
        #  Apply the label mapping
        label_map = {}
        with open(self.params.args.label_mapping, 'r') as f:
            seq_to_int = yaml.load(f)
            for k, v in seq_to_int.items():
                label_map[v] = k

        self.label_mapping = label_map

        preds_top_k = []
        for i in tqdm.tqdm(preds):
            preds_top_k.append(self.infer_top_k(i, self.params.args.top_k))

        # preds_top_k = [self.infer_top_k(i, self.params.args.top_k) for i in preds]
        targets = list(map(lambda x: self.interpret_pred(list(x)), targets))

        top_k = []
        #  Class-specific accuracy
        class_top_k = collections.defaultdict(list)

        for target, pred in zip(targets, preds_top_k):
            # if any([i in blacklist for i in target.split(':')]):
            #     continue
            top_k.append([int(i[0] == target) for i in pred])
            #  There should only be one 1
            try:
                idx = top_k[-1].index(1)
                for j in range(idx+1, len(top_k[-1])):
                    top_k[-1][j] = 0
            except ValueError:
                pass

            class_top_k[target].append(top_k[-1])
            if sum(top_k[-1]) > 1:
                print("WTF : ", pred, target)

        print(len(preds), len(top_k))
        top_k = np.array(top_k)
        total = len(top_k)
        top_k_acc = np.cumsum(np.sum(top_k, axis=0)) / total
        for i, acc in enumerate(top_k_acc, 1):
            print("Top-{} Accuracy : {:.4f}".format(i, acc))

        per_class_top_k = {}
        for k, v in class_top_k.items():
            per_class_top_k[str(k)] = list(np.cumsum(np.sum(v, axis=0)) / len(v))

        top_k_df = pd.DataFrame(list(per_class_top_k.values()),
                                index=list(per_class_top_k.keys()),
                                columns=['Top-{}'.format(i) for i in range(1, self.params.args.top_k + 1)])
        top_k_df.sort_values(['Top-1'], ascending=False, inplace=True)
        top_k_df.loc['total'] = top_k_acc
        with open('{}/top-{}.csv'.format(self.wdir, self.params.args.top_k), 'w') as f:
            print(top_k_df.to_csv(), file=f)

    def save_interface(self, path: str):
        from ggnn.inference.interfaces import GGNNSeqStaticRNNInterface
        interface = GGNNSeqStaticRNNInterface()
        with open(path, 'wb') as f:
            pickle.dump(interface, f)
