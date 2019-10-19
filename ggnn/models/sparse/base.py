import collections
import pickle

import tensorflow as tf
import numpy as np
from typing import Tuple, Dict, List

import tqdm

from autopandas.utils.io import IndexedFileReader
from ggnn.models.base import BaseGGNN
from ggnn.models import utils


class SparseGGNN(BaseGGNN):
    @classmethod
    def default_params(cls):
        params = dict(super().default_params())
        params.update({
            'batch_size': 100000,
            'use_edge_bias': False,

            #  Message-passing attention
            'use_propagation_attention': True,
            'attention_mechanism': 'scalar-edge-dot',
            'attention_normalization': 'softmax',

            #  Attention mechanism at the time of pooling node embeddings
            'use_node_pooling_attention': False,
            'node_pool_attention_mechanism': 'default',
            'node_pool_attention_normalization': 'softmax',

            'use_edge_msg_avg_aggregation': True,
            'residual_connections': {  # For layer i, specify list of layers whose output is added as an input
                # "2": [0],
                # "4": [0, 2]
            },

            # 'layer_timesteps': [2, 2, 1, 2, 1],  # number of layers & propagation steps per layer
            'layer_timesteps': [1, 1],

            'graph_rnn_cell': 'GRU',  # GRU, CudnnCompatibleGRUCell, or RNN
            'graph_rnn_activation': 'tanh',  # tanh, ReLU
            'graph_state_dropout_keep_prob': 1.,
            'edge_weight_dropout_keep_prob': .8
        })

        return params

    # --------------------------------------------------------------- #
    #  Data Processing
    # --------------------------------------------------------------- #

    def preprocess_data(self, path, is_training_data=False):
        reader = IndexedFileReader(path)

        num_fwd_edge_types = 0
        annotation_size = 0
        num_classes = 0
        depth = 1
        for g in tqdm.tqdm(reader, desc='Preliminary Data Pass'):
            num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['edges']] + [-1]) + 1)
            annotation_size = max(annotation_size, max(g['node_features']) + 1)
            num_classes = max(num_classes, g['label'] + 1)

        self.params['num_edge_types'] = num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2)
        self.params['annotation_size'] = annotation_size
        self.params['num_classes'] = num_classes
        reader.close()

    def process_raw_graph(self, graph):
        (adjacency_lists, num_incoming_edge_per_type) = self.graph_to_adjacency_lists(graph['edges'])
        return {"adjacency_lists": adjacency_lists,
                "num_incoming_edge_per_type": num_incoming_edge_per_type,
                "init": self.to_one_hot(graph["node_features"], self.params['annotation_size']),
                "label": graph.get("label", 0)}

    def graph_to_adjacency_lists(self, graph) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[int, int]]]:
        adj_lists = collections.defaultdict(list)
        num_incoming_edges_dicts_per_type = {}
        for src, e, dest in graph:
            fwd_edge_type = e
            adj_lists[fwd_edge_type].append((src, dest))
            if fwd_edge_type not in num_incoming_edges_dicts_per_type:
                num_incoming_edges_dicts_per_type[fwd_edge_type] = collections.defaultdict(int)

            num_incoming_edges_dicts_per_type[fwd_edge_type][dest] += 1
            if self.params['tie_fwd_bkwd']:
                adj_lists[fwd_edge_type].append((dest, src))
                num_incoming_edges_dicts_per_type[fwd_edge_type][src] += 1

        final_adj_lists = {e: np.array(sorted(lm), dtype=np.int32)
                           for e, lm in adj_lists.items()}

        # Add backward edges as an additional edge type that goes backwards:
        if not (self.params['tie_fwd_bkwd']):
            for (edge_type, edges) in adj_lists.items():
                bwd_edge_type = self.params['num_edge_types'] + edge_type
                final_adj_lists[bwd_edge_type] = np.array(sorted((y, x) for (x, y) in edges), dtype=np.int32)
                if bwd_edge_type not in num_incoming_edges_dicts_per_type:
                    num_incoming_edges_dicts_per_type[bwd_edge_type] = collections.defaultdict(int)

                for (x, y) in edges:
                    num_incoming_edges_dicts_per_type[bwd_edge_type][y] += 1

        return final_adj_lists, num_incoming_edges_dicts_per_type

    @staticmethod
    def to_one_hot(vals: List[int], depth: int):
        res = []
        for val in vals:
            v = [0] * depth
            v[val] = 1
            res.append(v)

        return res

    def perform_pooling(self, last_h):
        #  By default, it simply sums up the node embeddings
        #  We do not assume sorted segment_ids
        graph_node_sums = tf.unsorted_segment_sum(data=last_h,  # [v x h]
                                                  segment_ids=self.placeholders['graph_nodes_list'],
                                                  num_segments=self.placeholders['num_graphs'])  # [g x h]

        if not self.params.get('use_node_pooling_attention', False):
            return graph_node_sums

        mechanism = self.params.get('node_pooling_attention_mechanism', 'default')

        if mechanism == 'default':  # TODO : Get a better name
            node_sum_copies = tf.gather(params=graph_node_sums,
                                        indices=self.placeholders['graph_nodes_list'])

            nodes_with_node_sums = tf.concat([last_h, node_sum_copies], -1)
            intermediate_alignment = tf.layers.dense(nodes_with_node_sums, units=100, activation=tf.nn.relu,
                                                     name='interm_node_pool_attn')
            alignment_scores = tf.layers.dense(intermediate_alignment, units=1,
                                               name='final_node_pool_attn')

            attention = self.compute_attention_normalization(alignment_scores, self.placeholders['graph_nodes_list'],
                                                             self.placeholders['num_graphs'])
            normalized_last_h = attention * last_h
            graph_attention_sums = tf.unsorted_segment_sum(data=normalized_last_h,
                                                           segment_ids=self.placeholders['graph_nodes_list'],
                                                           num_segments=self.placeholders['num_graphs'])

            return graph_attention_sums

        else:
            raise NotImplementedError("Node-Pooling attention mechanism {} not implemented".format(mechanism))

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
                num_graphs += 1
                num_graphs_in_batch += 1
                node_offset += num_nodes_in_graph

            batch_feed_dict = {
                self.placeholders['initial_node_representation']: np.array(batch_node_features),
                self.placeholders['num_incoming_edges_per_type']: np.concatenate(batch_num_incoming_edges_per_type,
                                                                                 axis=0),
                self.placeholders['graph_nodes_list']: np.concatenate(batch_graph_nodes_list),
                self.placeholders['target_values']: batch_target_task_values,
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

    def save_interface(self, path: str):
        from ggnn.inference.interfaces import SparseGGNNInterface
        interface = SparseGGNNInterface()
        with open(path, 'wb') as f:
            pickle.dump(interface, f)

    # --------------------------------------------------------------- #
    #  Model Definition
    # --------------------------------------------------------------- #

    def define_placeholders(self):
        super().define_placeholders()
        h_dim = self.params['hidden_size_node']
        self.placeholders['initial_node_representation'] = tf.placeholder(tf.float32, [None, h_dim],
                                                                          name='node_features')
        self.placeholders['adjacency_lists'] = [tf.placeholder(tf.int32, [None, 2], name='adjacency_e%s' % e)
                                                for e in range(self.params['num_edge_types'])]
        self.placeholders['num_incoming_edges_per_type'] = tf.placeholder(tf.float32,
                                                                          [None, self.params['num_edge_types']],
                                                                          name='num_incoming_edges_per_type')
        self.placeholders['graph_nodes_list'] = tf.placeholder(tf.int32, [None], name='graph_nodes_list')
        self.placeholders['graph_state_keep_prob'] = tf.placeholder(tf.float32, None, name='graph_state_keep_prob')
        self.placeholders['edge_weight_dropout_keep_prob'] = tf.placeholder(tf.float32, None,
                                                                            name='edge_weight_dropout_keep_prob')

    def prepare_specific_graph_model(self) -> None:
        h_dim = self.params['hidden_size_node']

        activation_name = self.params['graph_rnn_activation'].lower()
        if activation_name == 'tanh':
            activation_fun = tf.nn.tanh
        elif activation_name == 'relu':
            activation_fun = tf.nn.relu
        else:
            raise Exception("Unknown activation function type '%s'." % activation_name)

        self.GGNNWeights = collections.namedtuple('GGNNWeights', ['edge_weights',
                                                                  'edge_biases',
                                                                  'edge_type_attention_weights',
                                                                  'rnn_cells', ])

        # Generate per-layer values for edge weights, biases and gated units:
        self.weights = {}  # Used by super-class to place generic things
        self.gnn_weights = {
            'edge_weights': [],
            'edge_biases': [],
            'rnn_cells': []
        }

        for layer_idx in range(len(self.params['layer_timesteps'])):
            with tf.variable_scope('gnn_layer_%i' % layer_idx):
                edge_weights = tf.Variable(utils.glorot_init([self.params['num_edge_types'] * h_dim, h_dim]),
                                           name='gnn_edge_weights_%i' % layer_idx)
                edge_weights = tf.reshape(edge_weights, [self.params['num_edge_types'], h_dim, h_dim])
                edge_weights = tf.nn.dropout(edge_weights, keep_prob=self.placeholders['edge_weight_dropout_keep_prob'])
                self.gnn_weights['edge_weights'].append(edge_weights)

                if self.params['use_propagation_attention']:
                    self.setup_attention_weights(layer_idx)

                if self.params['use_edge_bias']:
                    self.gnn_weights['edge_biases'].append(
                        tf.Variable(np.zeros([self.params['num_edge_types'], h_dim], dtype=np.float32),
                                    name='gnn_edge_biases_%i' % layer_idx))

                cell_type = self.params['graph_rnn_cell'].lower()
                if cell_type == 'gru':
                    cell = tf.nn.rnn_cell.GRUCell(h_dim, activation=activation_fun)
                elif cell_type == 'cudnncompatiblegrucell':
                    assert (activation_name == 'tanh')
                    import tensorflow.contrib.cudnn_rnn as cudnn_rnn
                    cell = cudnn_rnn.CudnnCompatibleGRUCell(h_dim)
                elif cell_type == 'rnn':
                    cell = tf.nn.rnn_cell.BasicRNNCell(h_dim, activation=activation_fun)
                else:
                    raise Exception("Unknown RNN cell type '%s'." % cell_type)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                     state_keep_prob=self.placeholders['graph_state_keep_prob'])
                self.gnn_weights['rnn_cells'].append(cell)

    def setup_attention_weights(self, layer_idx: int):
        if self.params['attention_mechanism'] == 'scalar-edge-dot':
            if 'edge_type_attention_weights' not in self.gnn_weights:
                self.gnn_weights['edge_type_attention_weights'] = []

            self.gnn_weights['edge_type_attention_weights'].append(
                tf.Variable(np.ones([self.params['num_edge_types']], dtype=np.float32),
                            name='edge_type_attention_weights_%i' % layer_idx))

    def compute_attention_scores(self, layer_idx,
                                 messages,
                                 message_source_states, message_target_states,
                                 message_edge_types):

        #  Return unnormalized attention scores for each message (Shape [M])
        mechanism = self.params['attention_mechanism']
        if mechanism == 'scalar-edge-dot':
            message_edge_type_factors = tf.nn.embedding_lookup(
                params=self.gnn_weights['edge_type_attention_weights'][layer_idx],
                ids=message_edge_types)  # Shape [M]

            #  Basically take the dot product of the source state and target state
            message_attention_scores = tf.einsum('mi,mi->m', message_source_states, message_target_states)

            #  And now multiply it with a single weight associated with every edge-type
            message_attention_scores = message_attention_scores * message_edge_type_factors
            return message_attention_scores

        elif mechanism == 'scalar-edge-dense':
            message_edge_type_factors = tf.nn.embedding_lookup(
                params=self.gnn_weights['edge_type_attention_weights'][layer_idx],
                ids=message_edge_types)  # Shape [M]

            concat_source_target = tf.concat([message_source_states, message_target_states], -1)
            intermediate_attention = tf.layers.dense(concat_source_target, units=100, activation=tf.nn.relu,
                                                     name='interm_msg_pass_attn_{0}'.format(layer_idx))
            message_attention_scores = tf.squeeze(tf.layers.dense(
                intermediate_attention, units=1, name='final_msg_pass_attn_{0}'.format(layer_idx)), [1])

            message_attention_scores = message_attention_scores * message_edge_type_factors
            return message_attention_scores

        else:
            raise NotImplementedError("Message-Passing attention mechanism {} not implemented".format(mechanism))

    def compute_attention_normalization(self, message_attention_scores, message_targets, num_nodes):
        if self.params['attention_normalization'] == 'softmax':
            # The following is softmax-ing over the incoming messages per node.
            # As the number of incoming varies, we can't just use tf.softmax.
            # Reimplement with logsumexp trick:

            # Step (1): Obtain shift constant as max of messages going into a node
            message_attention_score_max_per_target = tf.unsorted_segment_max(
                data=message_attention_scores,
                segment_ids=message_targets,
                num_segments=num_nodes)  # Shape [V]

            # Step (2): Distribute max out to the corresponding messages again, and shift scores:
            message_attention_score_max_per_message = tf.gather(
                params=message_attention_score_max_per_target,
                indices=message_targets)  # Shape [M]

            message_attention_scores -= message_attention_score_max_per_message

            # Step (3): Exp, sum up per target, compute exp(score) / exp(sum) as attention prob:
            message_attention_scores_exped = tf.exp(message_attention_scores)  # Shape [M]
            message_attention_score_sum_per_target = tf.unsorted_segment_sum(
                data=message_attention_scores_exped,
                segment_ids=message_targets,
                num_segments=num_nodes)  # Shape [V]

            message_attention_normalisation_sum_per_message = tf.gather(
                params=message_attention_score_sum_per_target,
                indices=message_targets)  # Shape [M]

            message_attention = message_attention_scores_exped / (
                    message_attention_normalisation_sum_per_message + utils.SMALL_NUMBER)  # Shape [M]

            return message_attention

    def compute_final_node_representations(self) -> tf.Tensor:
        # Used shape abbreviations:
        #   V ~ number of nodes
        #   D ~ state dimension
        #   E ~ number of edges of current type
        #   M ~ number of messages (sum of all E)

        #  One entry per layer (final state of that layer), shape: number of nodes in batch v x D
        node_states_per_layer = [self.placeholders['initial_node_representation']]
        num_nodes = tf.shape(self.placeholders['initial_node_representation'], out_type=tf.int32)[0]

        message_sources = []  # list of tensors of message sources of shape [E]
        message_targets = []  # list of tensors of message targets of shape [E]
        message_edge_types = []  # list of tensors of edge type of shape [E]
        for edge_type_idx, adjacency_list_for_edge_type in enumerate(self.placeholders['adjacency_lists']):
            edge_sources = adjacency_list_for_edge_type[:, 0]
            edge_targets = adjacency_list_for_edge_type[:, 1]
            message_sources.append(edge_sources)
            message_targets.append(edge_targets)
            message_edge_types.append(tf.ones_like(edge_targets, dtype=tf.int32) * edge_type_idx)
        message_sources = tf.concat(message_sources, axis=0)  # Shape [M]
        message_targets = tf.concat(message_targets, axis=0)  # Shape [M]
        message_edge_types = tf.concat(message_edge_types, axis=0)  # Shape [M]

        for (layer_idx, num_timesteps) in enumerate(self.params['layer_timesteps']):
            with tf.variable_scope('gnn_layer_%i' % layer_idx):
                # Extract residual messages, if any:
                layer_residual_connections = self.params['residual_connections'].get(str(layer_idx))
                if layer_residual_connections is None:
                    layer_residual_states = []
                else:
                    layer_residual_states = [node_states_per_layer[residual_layer_idx]
                                             for residual_layer_idx in layer_residual_connections]

                if self.params['use_propagation_attention']:
                    message_edge_type_factors = tf.nn.embedding_lookup(
                        params=self.gnn_weights['edge_type_attention_weights'][layer_idx],
                        ids=message_edge_types)  # Shape [M]

                # Record new states for this layer. Initialised to last state, but will be updated below:
                node_states_per_layer.append(node_states_per_layer[-1])
                for step in range(num_timesteps):
                    with tf.variable_scope('timestep_%i' % step):
                        messages = []  # list of tensors of messages of shape [E, D]
                        message_source_states = []  # list of tensors of edge source states of shape [E, D]

                        # Collect incoming messages per edge type
                        for edge_type_idx, adjacency_list_for_edge_type in enumerate(
                                self.placeholders['adjacency_lists']):
                            edge_sources = adjacency_list_for_edge_type[:, 0]
                            edge_source_states = tf.nn.embedding_lookup(params=node_states_per_layer[-1],
                                                                        ids=edge_sources)  # Shape [E, D]
                            all_messages_for_edge_type = tf.matmul(edge_source_states,
                                                                   self.gnn_weights['edge_weights'][layer_idx][
                                                                       edge_type_idx])  # Shape [E, D]
                            messages.append(all_messages_for_edge_type)
                            message_source_states.append(edge_source_states)

                        messages = tf.concat(messages, axis=0)  # Shape [M, D]

                        if self.params['use_propagation_attention']:
                            message_source_states = tf.concat(message_source_states, axis=0)  # Shape [M, D]
                            message_target_states = tf.nn.embedding_lookup(params=node_states_per_layer[-1],
                                                                           ids=message_targets)  # Shape [M, D]

                            message_attention_scores = self.compute_attention_scores(layer_idx, messages,
                                                                                     message_source_states,
                                                                                     message_target_states,
                                                                                     message_edge_types)

                            message_attention = self.compute_attention_normalization(message_attention_scores,
                                                                                     message_targets,
                                                                                     num_nodes)

                            self.ops['message_sources_%d_%d' % (layer_idx, step)] = message_sources
                            self.ops['message_targets_%d_%d' % (layer_idx, step)] = message_targets
                            self.ops['message_attentions_%d_%d' % (layer_idx, step)] = message_attention

                            # Step (4): Weigh messages using the attention prob:
                            messages = messages * tf.expand_dims(message_attention, -1)

                        incoming_messages = tf.unsorted_segment_sum(data=messages,
                                                                    segment_ids=message_targets,
                                                                    num_segments=num_nodes)  # Shape [V, D]

                        if self.params['use_edge_bias']:
                            incoming_messages += tf.matmul(self.placeholders['num_incoming_edges_per_type'],
                                                           self.gnn_weights['edge_biases'][layer_idx])  # Shape [V, D]

                        if self.params['use_edge_msg_avg_aggregation']:
                            num_incoming_edges = tf.reduce_sum(self.placeholders['num_incoming_edges_per_type'],
                                                               keep_dims=True, axis=-1)  # Shape [V, 1]
                            incoming_messages /= num_incoming_edges + utils.SMALL_NUMBER

                        incoming_information = tf.concat(layer_residual_states + [incoming_messages],
                                                         axis=-1)  # Shape [V, D*(1 + num of residual connections)]

                        # pass updated vertex features into RNN cell
                        node_states_per_layer[-1] = self.gnn_weights['rnn_cells'][layer_idx](
                            incoming_information, node_states_per_layer[-1])[1]  # Shape [V, D]

        return node_states_per_layer[-1]

        # --------------------------------------------------------------- #
        #  Debugging
        # --------------------------------------------------------------- #

    def debug_attention(self, graph):
        #  Graph is assumed to be processed
        graph = self.process_raw_graph(graph)

        batch_iterator = utils.ThreadedIterator(self.make_minibatch_iterator([graph], is_training=False),
                                                max_queue_size=50)

        fetch_list = []
        for (layer_idx, num_timesteps) in enumerate(self.params['layer_timesteps']):
            for tstep in range(num_timesteps):
                fetch_list += [self.ops['message_sources_%d_%d' % (layer_idx, tstep)],
                               self.ops['message_targets_%d_%d' % (layer_idx, tstep)],
                               self.ops['message_attentions_%d_%d' % (layer_idx, tstep)]]

        attention_dict = {}
        for step, batch_data in enumerate(batch_iterator):
            batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0

            result = self.sess.run(fetch_list, feed_dict=batch_data)

            idx = 0
            for (layer_idx, num_timesteps) in enumerate(self.params['layer_timesteps']):
                for tstep in range(num_timesteps):
                    message_sources, message_targets, message_attentions = result[idx], result[idx + 1], result[idx + 2]
                    attention_dict[(layer_idx, tstep)] = collections.defaultdict(list)

                    for source, target, attention in zip(message_sources, message_targets, message_attentions):
                        attention_dict[(layer_idx, tstep)][target].append((source, attention))

        return attention_dict
