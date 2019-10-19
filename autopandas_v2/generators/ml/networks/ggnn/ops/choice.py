import pickle
from typing import Dict

import tqdm
import tensorflow as tf

from autopandas_v2.generators.ml.networks.ggnn import utils
from autopandas_v2.generators.ml.networks.ggnn.base import BaseSmartGenGGNN
from autopandas_v2.utils.exceptions import AutoPandasException
from autopandas_v2.utils.ioutils import IndexedFileReader


class ModelChoice(BaseSmartGenGGNN):
    def preprocess_data(self, path, is_training_data=False):
        reader = IndexedFileReader(path)

        num_fwd_edge_types = 0
        annotation_size = 0
        num_choices = -1
        for g in tqdm.tqdm(reader, desc='Preliminary Data Pass', dynamic_ncols=True):
            num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['edges']] + [-1]) + 1)
            annotation_size = max(annotation_size, max(g['node_features']) + 1)
            if num_choices == -1:
                num_choices = g['num_choices']
            else:
                if num_choices != g['num_choices']:
                    raise AutoPandasException("Number of choices differ across training points")

        self.params['num_edge_types'] = num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2)
        self.params['annotation_size'] = annotation_size
        self.params['num_choices'] = num_choices
        reader.close()

    def process_raw_graph(self, graph):
        processed = super().process_raw_graph(graph)
        processed['chosen'] = graph.get('chosen', -1)
        return processed

    def define_placeholders(self):
        super().define_placeholders()
        self.placeholders['chosen'] = tf.placeholder(tf.int64, [None], name='chosen')

    def per_graph_custom_minibatch_iterator(self, graph_num: int, graph: Dict, node_offset: int) -> Dict:
        return {
            'chosen': [graph['chosen']]
        }

    def final_predictor(self):
        #  By default, a simple MLP with one hidden layer
        return utils.MLP(self.params['hidden_size_node'], self.params['num_choices'],
                         [self.params['hidden_size_final_mlp']],
                         self.placeholders['out_layer_dropout_keep_prob'])

    def make_model(self, mode):
        super().make_model(mode)
        with tf.variable_scope("out_layer"):
            self.ops['final_predictor'] = self.final_predictor()
            pooled_representations = self.prepare_pooled_node_representations()

            # Should return logits with dimension equal to the number of output classes
            logits = self.ops['final_predictor'](pooled_representations)
            labels = self.placeholders['chosen']

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

            self.ops['loss'] = tf.reduce_mean(loss)
            probabilities = tf.nn.softmax(logits)

            correct_prediction = tf.equal(tf.argmax(probabilities, -1), labels)
            self.ops['accuracy_task'] = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            top_k = tf.nn.top_k(probabilities, k=self.params['num_choices'])
            self.ops['preds'] = top_k.indices
            self.ops['probs'] = top_k.values
            self.ops['targets'] = self.placeholders['chosen']

    def infer(self, raw_graph_data, **kwargs):
        graphs = [self.process_raw_graph(g) for g in raw_graph_data]

        batch_iterator = utils.ThreadedIterator(self.make_minibatch_iterator(graphs, is_training=False),
                                                max_queue_size=50)

        preds = []

        for step, batch_data in enumerate(batch_iterator):
            batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
            fetch_list = [self.ops['preds'], self.ops['probs']]

            result = self.sess.run(fetch_list, feed_dict=batch_data)

            for p, v in zip(result[1], result[0]):
                preds.append(list(zip(p, v)))

        return preds

    def save_interface(self, path: str):
        from autopandas_v2.generators.ml.networks.ggnn.interfaces import ModelChoiceInterface
        interface = ModelChoiceInterface()
        with open(path, 'wb') as f:
            pickle.dump(interface, f)

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
        return super().get_fetch_list() + [self.ops['preds'], self.ops['probs'], self.ops['targets']]

    def analyze_result(self, num_graphs, result):
        super().analyze_result(num_graphs, result)
        for p, v, t in zip(result[2], result[1], result[3]):
            preds = list(zip(p, v))
            preds = sorted(preds, key=lambda x: -x[0])
            for idx, (prob, val) in enumerate(preds):
                if val == t:
                    self.correct_indices.append(idx)
                    break
