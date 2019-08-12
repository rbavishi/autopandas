import collections
import pickle
from typing import Dict, Tuple, List
import numpy as np
import tensorflow as tf
import tqdm

from autopandas_v2.generators.ml.networks.ggnn import utils
from autopandas_v2.generators.ml.networks.ggnn.base import BaseSmartGenGGNN
from autopandas_v2.ml.networks.ggnn.models.sparse.seq.static_rnn import LSTMDecoder
from autopandas_v2.ml.networks.ggnn.utils import ParamsNamespace
from autopandas_v2.utils.ioutils import IndexedFileReader


class ModelOrderedSubsets(BaseSmartGenGGNN):
    def preprocess_data(self, path, is_training_data=False):
        reader = IndexedFileReader(path)

        num_fwd_edge_types = 0
        annotation_size = 0
        max_length = 0
        for g in tqdm.tqdm(reader, desc='Preliminary Data Pass', dynamic_ncols=True):
            num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['edges']] + [-1]) + 1)
            annotation_size = max(annotation_size, max(g['node_features']) + 1)
            max_length = min(max(max_length, len(g['selected'])), 8)

        self.params['num_edge_types'] = num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2)
        self.params['annotation_size'] = annotation_size
        self.params['max_length'] = max_length
        reader.close()

    def process_raw_graph(self, graph):
        if 'selected' in graph and len(graph['selected']) > 8:
            return None

        processed = super().process_raw_graph(graph)
        processed['elements'] = graph['elements']
        processed['unroll_length'] = len(graph['selected']) if 'selected' in graph else self.params['max_length']
        processed['selected_one_hot'] = [[int(e == t) for e in processed['elements']]
                                         for t in graph.get('selected', [])]
        processed['terminal'] = graph['terminal']
        return processed

    def define_placeholders(self):
        super().define_placeholders()
        self.placeholders['elements'] = tf.placeholder(tf.int32, [None], name='elements')
        self.placeholders['elements_true'] = tf.placeholder(tf.int32, [None], name='elements_true')
        self.placeholders['elem_graph_nodes_list'] = tf.placeholder(tf.int32, [None], name='elem_graph_nodes_list')
        self.placeholders['stacked_elem_graph_nodes_list'] = tf.placeholder(tf.int32, [None],
                                                                            name='stacked_elem_graph_nodes_list')
        self.placeholders['graph_ids_timestep'] = tf.placeholder(tf.int32, [None], name='graph_ids_timestep')
        self.placeholders['graph_id_ranges'] = tf.placeholder(tf.int32, [None], name='graph_id_ranges')
        self.placeholders['subset_targets'] = tf.placeholder(tf.int64, [None], name='subset_targets')
        self.placeholders['subset_lengths'] = tf.placeholder(tf.int64, [None], name="subset_lengths")
        self.placeholders['subset_masks'] = tf.placeholder(tf.float32, [None], name='subset_masks')
        self.placeholders['sum_seq_lengths'] = tf.placeholder(tf.float32, None, name='sum_seq_in_lengths')

    def per_graph_custom_minibatch_iterator(self, graph_num: int, graph: Dict, node_offset: int) -> Dict:
        return {
            'elements': np.array(graph['elements']) + node_offset,
            'elements_true': np.array(graph['elements']),
            'subset_lengths': np.array([graph['unroll_length']]),
            'elem_graph_nodes_list': np.full([len(graph['elements'])], fill_value=graph_num),
        }

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
            sum_seq_lengths = 0
            batch_node_features = []
            batch_adjacency_lists = [[] for _ in range(self.params['num_edge_types'])]
            batch_num_incoming_edges_per_type = []
            batch_graph_nodes_list = []
            batch_subset_targets = [[] for i in range(self.params['max_length'])]
            batch_subset_masks = [[] for i in range(self.params['max_length'])]
            batch_stacked_elem_graph_nodes_list = [[] for i in range(self.params['max_length'])]
            batch_graph_ids_timestep = [[] for i in range(self.params['max_length'])]
            batch_graph_id_ranges = [[] for i in range(self.params['max_length'])]
            batch_per_graph_custom = collections.defaultdict(list)
            node_offset = 0
            batch_size = self.params['batch_size'] / self.params['max_length']

            while num_graphs < len(data) and (num_graphs_in_batch == 0 or
                                              node_offset + len(data[num_graphs]['init']) < batch_size):
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
                sum_seq_lengths += len(cur_graph['selected_one_hot'])

                for d, i in enumerate(cur_graph['selected_one_hot']):
                    batch_subset_targets[d].extend(i)
                    batch_subset_masks[d].extend([1] * len(i))
                    batch_stacked_elem_graph_nodes_list[d].extend(np.full([len(cur_graph['elements'])],
                                                                          fill_value=num_graphs_in_batch))
                    batch_graph_ids_timestep[d].extend(np.full([len(cur_graph['elements'])],
                                                               fill_value=num_graphs_in_batch))
                    batch_graph_id_ranges[d].append(num_graphs_in_batch)

                for d in range(len(cur_graph['selected_one_hot']), self.params['max_length']):
                    batch_subset_targets[d].extend([0] * len(cur_graph['elements']))
                    batch_subset_masks[d].extend([0] * len(cur_graph['elements']))
                    batch_stacked_elem_graph_nodes_list[d].extend(np.full([len(cur_graph['elements'])],
                                                                          fill_value=num_graphs_in_batch))
                    batch_graph_ids_timestep[d].extend(np.full([len(cur_graph['elements'])],
                                                               fill_value=num_graphs_in_batch))
                    batch_graph_id_ranges[d].append(num_graphs_in_batch)

                for k, v in self.per_graph_custom_minibatch_iterator(num_graphs_in_batch, data[num_graphs],
                                                                     node_offset).items():
                    batch_per_graph_custom[k].append(v)

                num_graphs += 1
                num_graphs_in_batch += 1
                node_offset += num_nodes_in_graph

            for d, l in enumerate(batch_stacked_elem_graph_nodes_list):
                batch_stacked_elem_graph_nodes_list[d] = np.array(l) + d * num_graphs_in_batch

            batch_feed_dict = {
                self.placeholders['initial_node_representation']: np.array(batch_node_features),
                self.placeholders['num_incoming_edges_per_type']: np.concatenate(batch_num_incoming_edges_per_type,
                                                                                 axis=0),
                self.placeholders['graph_nodes_list']: np.concatenate(batch_graph_nodes_list),
                self.placeholders['num_graphs']: num_graphs_in_batch,
                self.placeholders['graph_state_keep_prob']: state_dropout_keep_prob,
                self.placeholders['edge_weight_dropout_keep_prob']: edge_weights_dropout_keep_prob,
                self.placeholders['num_graphs_in_batch']: num_graphs_in_batch,
                self.placeholders['sum_seq_lengths']: sum_seq_lengths,
                #  First-dim is max_length
                self.placeholders['subset_targets']: np.concatenate(batch_subset_targets, axis=0),
                self.placeholders['subset_masks']: np.concatenate(batch_subset_masks, axis=0),
                self.placeholders['stacked_elem_graph_nodes_list']: np.concatenate(batch_stacked_elem_graph_nodes_list,
                                                                                   axis=0),
                self.placeholders['graph_ids_timestep']: np.concatenate(batch_graph_ids_timestep, axis=0),
                self.placeholders['graph_id_ranges']: np.concatenate(batch_graph_id_ranges, axis=0)
            }

            for k, v in batch_per_graph_custom.items():
                #  The subclass's responsibility to make sure self.placeholders[k] exists
                batch_feed_dict[self.placeholders[k]] = np.concatenate(v)

            # Merge adjacency lists and information about incoming nodes:
            for i in range(self.params['num_edge_types']):
                if len(batch_adjacency_lists[i]) > 0:
                    adj_list = np.concatenate(batch_adjacency_lists[i])
                else:
                    adj_list = np.zeros((0, 2), dtype=np.int32)
                batch_feed_dict[self.placeholders['adjacency_lists'][i]] = adj_list

            yield batch_feed_dict

    def final_classifier(self):
        #  By default, a simple MLP with one hidden layer
        return utils.MLP(self.params['hidden_size_node'] * 2, 1,
                         [self.params['hidden_size_final_mlp']],
                         self.placeholders['out_layer_dropout_keep_prob'])

    def final_predictor(self):
        return LSTMDecoder(self.params['hidden_size_node'], self.params['hidden_size_node'],
                           self.params['hidden_size_final_mlp'], self.params['max_length'],
                           self.placeholders['out_layer_dropout_keep_prob'])

    def compute_candidate_softmax_and_loss(self, cand_logits):
        # The following is softmax-ing over the candidates per graph.
        # As the number of candidates varies, we can't just use tf.softmax.
        # We implement it with the logsumexp trick:

        stacked_elem_graph_nodes_list = self.placeholders['stacked_elem_graph_nodes_list']
        total_num_graphs_in_batch = self.placeholders['num_graphs_in_batch'] * self.params['max_length']

        # Step (1): Obtain shift constant as max of the logits
        max_per_graph = tf.unsorted_segment_max(
            data=cand_logits,
            segment_ids=stacked_elem_graph_nodes_list,
            num_segments=total_num_graphs_in_batch
        )  # Shape [max_length x G]

        # # Step (2): Distribute max out to the corresponding logits again, and shift scores:
        max_per_cand = tf.gather(params=max_per_graph,
                                 indices=stacked_elem_graph_nodes_list)

        cand_logits_shifted = cand_logits - max_per_cand

        # # Step (3): Exp, sum up per target, compute exp(score) / exp(sum) as softmax:
        scores_exped = tf.exp(cand_logits_shifted)
        scores_sum_per_graph = tf.unsorted_segment_sum(
            data=scores_exped,
            segment_ids=stacked_elem_graph_nodes_list,
            num_segments=total_num_graphs_in_batch
        )  # Shape [max_length x G]

        scores_sum_per_cand = tf.gather(
            params=scores_sum_per_graph,
            indices=stacked_elem_graph_nodes_list
        )

        self.ops['softmax_values'] = scores_exped / (scores_sum_per_cand + utils.SMALL_NUMBER)
        self.ops['log_softmax_values'] = cand_logits_shifted - tf.log(scores_sum_per_cand + utils.SMALL_NUMBER)
        labels = self.placeholders['subset_targets']
        flat_loss_values = -tf.cast(labels, "float32") * self.ops['log_softmax_values']
        flat_loss_values *= self.placeholders['subset_masks']
        losses = tf.unsorted_segment_sum(
            data=flat_loss_values,
            segment_ids=stacked_elem_graph_nodes_list,
            num_segments=total_num_graphs_in_batch
        )

        self.ops['loss'] = tf.reduce_sum(losses) / self.placeholders['sum_seq_lengths']

        flat_correct_prediction = tf.cast(tf.equal(cand_logits, max_per_cand), "int64") * labels

        correct_prediction_per_timestep = tf.unsorted_segment_max(
            data=flat_correct_prediction,
            segment_ids=stacked_elem_graph_nodes_list,
            num_segments=total_num_graphs_in_batch
        )  # Shape [max_length x G]

        correct_prediction_timestep_sum = tf.unsorted_segment_sum(
            data=correct_prediction_per_timestep,
            segment_ids=self.placeholders['graph_id_ranges'],
            num_segments=self.placeholders['num_graphs_in_batch']
        )

        correct_prediction = tf.cast(tf.equal(correct_prediction_timestep_sum, self.placeholders['subset_lengths']),
                                     "float")

        self.ops['accuracy_task'] = tf.reduce_mean(correct_prediction)

    def make_model(self, mode):
        super().make_model(mode)
        with tf.variable_scope("out_layer"):
            self.ops['final_classifier'] = self.final_classifier()
            self.ops['final_predictor'] = self.final_predictor()

            elements_repr = tf.gather(params=self.ops['final_node_representations'],
                                      indices=self.placeholders['elements'])
            elements_pooled = tf.unsorted_segment_sum(
                data=elements_repr,
                segment_ids=self.placeholders['elem_graph_nodes_list'],
                num_segments=self.placeholders['num_graphs_in_batch']
            )
            pooled_representations = self.prepare_pooled_node_representations()
            graph_pooled = pooled_representations - elements_pooled

            graph_cand_concat = tf.concat([graph_pooled, elements_pooled], -1)
            # Shape of graph_cand_rnn is [max_length, batch_size, H]
            graph_cand_rnn = self.ops['final_predictor'](graph_cand_concat, self.placeholders['subset_lengths'])
            #  Distribute to all the elements
            #  Shape is now [max_length, sum of number of elements over the batch, H]
            graph_cand_rnn_elem_copies = tf.gather(params=graph_cand_rnn,
                                                   indices=self.placeholders['elem_graph_nodes_list'],
                                                   axis=1)
            #  Shape is [max_length, sum of number of elements over the batch, H]
            stacked_elem_repr = tf.stack([elements_repr for i in range(self.params['max_length'])])
            #  Shape is [max_length, sum of number of elements over the batch, 2H]
            graph_cand_classifier_inp = tf.concat([graph_cand_rnn_elem_copies, stacked_elem_repr], -1)
            #  Ready it up for matrix-multiply
            graph_cand_classifier_inp = tf.reshape(graph_cand_classifier_inp, [-1, self.params['hidden_size_node'] * 2])
            #  Shape is [max_length x sum of number of elements over the batch]
            graph_cand_logits = tf.reshape(self.ops['final_classifier'](graph_cand_classifier_inp), [-1])

            #  Now do softmax black-magic
            self.compute_candidate_softmax_and_loss(graph_cand_logits)

    def save_interface(self, path: str):
        from autopandas_v2.generators.ml.networks.ggnn.interfaces import ModelOrderedSubsetsInterface
        interface = ModelOrderedSubsetsInterface()
        with open(path, 'wb') as f:
            pickle.dump(interface, f)

    def infer(self, raw_graph_data, **kwargs):
        graphs = [self.process_raw_graph(g) for g in raw_graph_data]

        batch_iterator = utils.ThreadedIterator(self.make_minibatch_iterator(graphs, is_training=False),
                                                max_queue_size=50)

        preds = []
        max_length = self.params['max_length']

        for step, batch_data in enumerate(batch_iterator):
            batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
            fetch_list = [self.ops['softmax_values'],
                          self.placeholders['graph_ids_timestep'],
                          self.placeholders['elements_true']]

            result = self.sess.run(fetch_list, feed_dict=batch_data)
            softmax_values, graph_ids, elements = result
            elements = np.concatenate([elements] * max_length, axis=0)
            groupings = collections.defaultdict(list)
            for probs, graph_id, element in zip(softmax_values, graph_ids, elements):
                groupings[graph_id].append((probs, element))

            for graph_id, flat_preds in groupings.items():
                num_elems = len(flat_preds) // max_length
                graph_preds = []
                for i in range(0, len(flat_preds), num_elems):
                    graph_preds.append(sorted(flat_preds[i: i + num_elems], key=lambda x: -x[0]))

                preds.append(graph_preds)

        return preds

    def init_analysis(self):
        super().init_analysis()
        self.correct_indices = []

    def finish_analysis(self):
        result = super().finish_analysis()
        top_k = self.params.args.top_k
        for k in range(1, top_k + 1):
            acc = sum(1 for i in self.correct_indices if i < k) / self.num_graphs
            print("Top-{} Accuracy : {}".format(k, acc))
            result['Top-{}'.format(k)] = acc

        return result

    def get_fetch_list(self):
        return super().get_fetch_list() + [self.ops['softmax_values'],
                                           self.placeholders['graph_ids_timestep'],
                                           self.placeholders['subset_targets']]

    def analyze_result(self, num_graphs, result):
        super().analyze_result(num_graphs, result)
        groupings = collections.defaultdict(list)
        softmax_values, graph_ids, targets = result[1:]
        max_length = self.params['max_length']
        for probs, graph_id, selected in zip(softmax_values, graph_ids, targets):
            groupings[graph_id].append((probs, selected))

        for graph_id, flat_preds in groupings.items():
            num_elems = len(flat_preds) // max_length
            graph_preds = []
            flat_preds = [(i[0], i[1], idx % num_elems) for idx, i in enumerate(flat_preds)]
            for i in range(0, len(flat_preds), num_elems):
                graph_preds.append(sorted(flat_preds[i: i + num_elems], key=lambda x: -x[0]))

            self.correct_indices.append(self.get_beam_search_idx(graph_preds, self.params.args.top_k, num_elems-1))

    def get_beam_search_idx(self, items: List[List[Tuple[float, int, int]]], width: int, num_elems: int):
        results: List[Tuple[float, List[int], List[int]]] = []
        beam: List[Tuple[float, List[int], List[int]]] = [(1.0, [], [])]
        for depth, preds in enumerate(items):
            new_beam: List[Tuple[float, List[int], List[int]]] = []
            for prob, selected, idx in preds:
                if idx == num_elems:
                    results.extend([(cum_prob * prob, elems[:], sels[:]) for cum_prob, elems, sels in beam])
                else:
                    new_beam.extend([(cum_prob * prob, elems + [idx], sels + [selected])
                                     for cum_prob, elems, sels in beam
                                     if idx not in elems])

            beam = list(reversed(sorted(new_beam)))[:width]

        results = list(reversed(sorted(results)))

        for idx, (prob, seq, sels) in enumerate(results):
            if all(i == 1 for i in sels):
                return idx

        return 1000000
