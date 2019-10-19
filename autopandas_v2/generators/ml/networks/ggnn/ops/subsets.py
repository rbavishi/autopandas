import collections
import pickle
from typing import Dict, Tuple, List
import numpy as np
import tensorflow as tf
from autopandas_v2.generators.ml.networks.ggnn import utils
from autopandas_v2.generators.ml.networks.ggnn.base import BaseSmartGenGGNN


class ModelSubsets(BaseSmartGenGGNN):
    def process_raw_graph(self, graph):
        processed = super().process_raw_graph(graph)
        processed['elements'] = graph['elements']
        processed['selected_one_hot'] = [int(i in graph.get('selected', [])) for i in graph['elements']]
        return processed

    def define_placeholders(self):
        super().define_placeholders()
        self.placeholders['elements'] = tf.placeholder(tf.int32, [None], name='elements')
        self.placeholders['elements_true'] = tf.placeholder(tf.int32, [None], name='elements_true')
        self.placeholders['elem_graph_nodes_list'] = tf.placeholder(tf.int32, [None], name='elements')
        self.placeholders['select_targets'] = tf.placeholder(tf.int64, [None], name='select_targets')

    def per_graph_custom_minibatch_iterator(self, graph_num: int, graph: Dict, node_offset: int) -> Dict:
        return {
            'elements': np.array(graph['elements']) + node_offset,
            'elements_true': np.array(graph['elements']),
            'select_targets': np.array(graph['selected_one_hot']),
            'elem_graph_nodes_list': np.full([len(graph['elements'])], fill_value=graph_num)
        }

    def final_classifier(self):
        #  By default, a simple MLP with one hidden layer
        return utils.MLP(self.params['hidden_size_node'] * 3, 2,
                         [self.params['hidden_size_final_mlp']],
                         self.placeholders['out_layer_dropout_keep_prob'])

    def make_model(self, mode):
        super().make_model(mode)
        with tf.variable_scope("out_layer"):
            self.ops['final_classifier'] = self.final_classifier()

            elements_repr = tf.gather(params=self.ops['final_node_representations'],
                                      indices=self.placeholders['elements'])
            elements_pooled = tf.unsorted_segment_sum(
                data=elements_repr,
                segment_ids=self.placeholders['elem_graph_nodes_list'],
                num_segments=self.placeholders['num_graphs_in_batch']
            )
            pooled_representations = self.prepare_pooled_node_representations()
            graph_pooled = pooled_representations - elements_pooled

            graph_pooled_copies = tf.gather(params=graph_pooled,
                                            indices=self.placeholders['graph_nodes_list'])

            cand_pooled_copies = tf.gather(params=elements_pooled,
                                           indices=self.placeholders['graph_nodes_list'])

            elements_graph = tf.gather(params=graph_pooled_copies, indices=self.placeholders['elements'])
            elements_pooled = tf.gather(params=cand_pooled_copies, indices=self.placeholders['elements'])
            elements_concat = tf.concat([elements_repr, elements_graph, elements_pooled], -1)

            elements_logits = self.ops['final_classifier'](elements_concat)
            labels = self.placeholders['select_targets']

            #  Subsets can be seen as independent binary classification over the elements
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=elements_logits, labels=labels)

            self.ops['loss'] = tf.reduce_mean(loss)
            self.ops['probabilities'] = tf.nn.softmax(elements_logits)
            probabilities = self.ops['probabilities']

            correct_prediction_elems = tf.cast(tf.equal(tf.argmax(probabilities, -1),
                                                        self.placeholders['select_targets']), "float")
            correct_prediction = tf.unsorted_segment_min(
                data=correct_prediction_elems,
                segment_ids=self.placeholders['elem_graph_nodes_list'],
                num_segments=self.placeholders['num_graphs_in_batch']
            )
            self.ops['accuracy_task'] = tf.reduce_mean(correct_prediction)

            top_k = tf.nn.top_k(probabilities, k=2)
            self.ops['preds'] = top_k.indices
            self.ops['probs'] = top_k.values

    def save_interface(self, path: str):
        from autopandas_v2.generators.ml.networks.ggnn.interfaces import ModelSubsetsInterface
        interface = ModelSubsetsInterface()
        with open(path, 'wb') as f:
            pickle.dump(interface, f)

    def infer(self, raw_graph_data, **kwargs):
        graphs = [self.process_raw_graph(g) for g in raw_graph_data]

        batch_iterator = utils.ThreadedIterator(self.make_minibatch_iterator(graphs, is_training=False),
                                                max_queue_size=50)

        preds = []
        for step, batch_data in enumerate(batch_iterator):
            batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
            fetch_list = [self.ops['probabilities'],
                          self.placeholders['elem_graph_nodes_list'],
                          self.placeholders['elements_true']]

            result = self.sess.run(fetch_list, feed_dict=batch_data)
            groupings = collections.defaultdict(list)
            for (discard_prob, keep_prob), graph_id, candidate in zip(*result):
                groupings[graph_id].append((discard_prob, keep_prob, len(groupings[graph_id])))

            preds += list(groupings.values())

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
        return super().get_fetch_list() + [self.ops['probabilities'],
                                           self.placeholders['elem_graph_nodes_list'],
                                           self.placeholders['select_targets']]

    def analyze_result(self, num_graphs, result):
        super().analyze_result(num_graphs, result)

        groupings = collections.defaultdict(list)
        for (discard_prob, keep_prob), graph_id, selected in zip(*result[1:]):
            groupings[graph_id].append((discard_prob, keep_prob, selected))

        for graph_id, preds in groupings.items():
            self.correct_indices.append(self.get_beam_search_idx(preds, self.params.args.top_k))

    def get_beam_search_idx(self, items: List[Tuple[float, float, int]], width: int):
        beam: List[Tuple[float, List[int]]] = [(1.0, [])]
        ground_truth = []
        for d_prob, k_prob, selected in items:
            ground_truth.append(selected)
            new_beam = []
            for cum_prob, elems in beam:
                new_beam.append((cum_prob * d_prob, elems + [0]))
                new_beam.append((cum_prob * k_prob, elems + [1]))

            beam = list(reversed(sorted(new_beam)))[:width]

        for idx, (prob, seq) in enumerate(beam):
            if seq == ground_truth:
                return idx

        return 1000000
