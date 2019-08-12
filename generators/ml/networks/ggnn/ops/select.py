import collections
import pickle
from typing import Dict
import numpy as np
import tensorflow as tf
from autopandas_v2.generators.ml.networks.ggnn import utils
from autopandas_v2.generators.ml.networks.ggnn.base import BaseSmartGenGGNN


class ModelSelect(BaseSmartGenGGNN):
    def process_raw_graph(self, graph):
        processed = super().process_raw_graph(graph)
        processed['candidates'] = graph['candidates']
        processed['selected_one_hot'] = [int(i == graph.get('selected', -1)) for i in graph['candidates']]
        return processed

    def define_placeholders(self):
        super().define_placeholders()
        self.placeholders['candidates'] = tf.placeholder(tf.int32, [None], name='candidates')
        self.placeholders['candidates_true'] = tf.placeholder(tf.int32, [None], name='candidates_true')
        self.placeholders['cand_graph_nodes_list'] = tf.placeholder(tf.int32, [None], name='candidates')
        self.placeholders['select_targets'] = tf.placeholder(tf.int64, [None], name='select_targets')

    def per_graph_custom_minibatch_iterator(self, graph_num: int, graph: Dict, node_offset: int) -> Dict:
        return {
            'candidates': np.array(graph['candidates']) + node_offset,
            'candidates_true': np.array(graph['candidates']),
            'select_targets': np.array(graph['selected_one_hot']),
            'cand_graph_nodes_list': np.full([len(graph['candidates'])], fill_value=graph_num)
        }

    def final_classifier(self):
        #  By default, a simple MLP with one hidden layer
        return utils.MLP(self.params['hidden_size_node'] * 3, 1,
                         [self.params['hidden_size_final_mlp']],
                         self.placeholders['out_layer_dropout_keep_prob'])

    def compute_candidate_softmax_and_loss(self, cand_logits):
        # The following is softmax-ing over the candidates per graph.
        # As the number of candidates varies, we can't just use tf.softmax.
        # We implement it with the logsumexp trick:

        # Step (1): Obtain shift constant as max of the logits
        max_per_graph = tf.unsorted_segment_max(
            data=cand_logits,
            segment_ids=self.placeholders['cand_graph_nodes_list'],
            num_segments=self.placeholders['num_graphs_in_batch']
        )  # Shape [G]

        # # Step (2): Distribute max out to the corresponding logits again, and shift scores:
        max_per_cand = tf.gather(params=max_per_graph,
                                 indices=self.placeholders['cand_graph_nodes_list'])

        cand_logits_shifted = cand_logits - max_per_cand

        # # Step (3): Exp, sum up per target, compute exp(score) / exp(sum) as softmax:
        scores_exped = tf.exp(cand_logits_shifted)
        scores_sum_per_graph = tf.unsorted_segment_sum(
            data=scores_exped,
            segment_ids=self.placeholders['cand_graph_nodes_list'],
            num_segments=self.placeholders['num_graphs_in_batch']
        )  # Shape [G]

        scores_sum_per_cand = tf.gather(
            params=scores_sum_per_graph,
            indices=self.placeholders['cand_graph_nodes_list']
        )

        self.ops['softmax_values'] = scores_exped / (scores_sum_per_cand + utils.SMALL_NUMBER)
        self.ops['log_softmax_values'] = cand_logits_shifted - tf.log(scores_sum_per_cand + utils.SMALL_NUMBER)
        labels = self.placeholders['select_targets']
        flat_loss_values = -tf.cast(labels, "float32") * self.ops['log_softmax_values']
        losses = tf.unsorted_segment_sum(
            data=flat_loss_values,
            segment_ids=self.placeholders['cand_graph_nodes_list'],
            num_segments=self.placeholders['num_graphs_in_batch']
        )

        self.ops['loss'] = tf.reduce_mean(losses)

        flat_correct_prediction = tf.cast(tf.equal(cand_logits, max_per_cand), "int64") * self.placeholders[
            'select_targets']

        correct_prediction = tf.unsorted_segment_max(
            data=tf.cast(flat_correct_prediction, "float32"),
            segment_ids=self.placeholders['cand_graph_nodes_list'],
            num_segments=self.placeholders['num_graphs_in_batch']
        )

        self.ops['accuracy_task'] = tf.reduce_mean(correct_prediction)

    def make_model(self, mode):
        super().make_model(mode)
        with tf.variable_scope("out_layer"):
            self.ops['final_classifier'] = self.final_classifier()

            candidates_repr = tf.gather(params=self.ops['final_node_representations'],
                                        indices=self.placeholders['candidates'])
            candidates_pooled = tf.unsorted_segment_sum(
                data=candidates_repr,
                segment_ids=self.placeholders['cand_graph_nodes_list'],
                num_segments=self.placeholders['num_graphs_in_batch']
            )
            pooled_representations = self.prepare_pooled_node_representations()
            graph_pooled = pooled_representations - candidates_pooled

            graph_pooled_copies = tf.gather(params=graph_pooled,
                                            indices=self.placeholders['graph_nodes_list'])

            cand_pooled_copies = tf.gather(params=candidates_pooled,
                                           indices=self.placeholders['graph_nodes_list'])

            candidates_graph = tf.gather(params=graph_pooled_copies, indices=self.placeholders['candidates'])
            candidates_pooled = tf.gather(params=cand_pooled_copies, indices=self.placeholders['candidates'])
            candidates_concat = tf.concat([candidates_repr, candidates_graph, candidates_pooled], -1)

            candidates_logits = tf.reshape(self.ops['final_classifier'](candidates_concat), [-1])
            self.compute_candidate_softmax_and_loss(candidates_logits)

    def save_interface(self, path: str):
        from autopandas_v2.generators.ml.networks.ggnn.interfaces import ModelSelectInterface
        interface = ModelSelectInterface()
        with open(path, 'wb') as f:
            pickle.dump(interface, f)

    def infer(self, raw_graph_data, **kwargs):
        graphs = [self.process_raw_graph(g) for g in raw_graph_data]

        batch_iterator = utils.ThreadedIterator(self.make_minibatch_iterator(graphs, is_training=False),
                                                max_queue_size=50)

        preds = []
        for step, batch_data in enumerate(batch_iterator):
            batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
            fetch_list = [self.ops['softmax_values'],
                          self.placeholders['cand_graph_nodes_list'],
                          self.placeholders['candidates_true']]

            result = self.sess.run(fetch_list, feed_dict=batch_data)
            groupings = collections.defaultdict(list)
            for probs, graph_id, candidate in zip(*result):
                groupings[graph_id].append((probs, candidate))

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
        return super().get_fetch_list() + [self.ops['softmax_values'],
                                           self.placeholders['cand_graph_nodes_list'],
                                           self.placeholders['select_targets']]

    def analyze_result(self, num_graphs, result):
        super().analyze_result(num_graphs, result)

        groupings = collections.defaultdict(list)
        for probs, graph_id, correct in zip(*result[1:]):
            groupings[graph_id].append((probs, correct))

        for graph_id, preds in groupings.items():
            preds = sorted(preds, key=lambda x: -x[0])
            for idx, (prob, correct) in enumerate(preds):
                if correct:
                    self.correct_indices.append(idx)
                    break
