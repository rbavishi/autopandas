from abc import ABC, abstractmethod

import tqdm

from autopandas.utils.io import IndexedFileReader
from ggnn.models.sparse.base import SparseGGNN
from ggnn.models import utils


class BaseGGNNSeq(SparseGGNN, ABC):
    def preprocess_data(self, path, is_training_data=False):
        reader = IndexedFileReader(path)

        num_fwd_edge_types = 0
        annotation_size = 0
        num_classes = 0
        max_depth = 1
        for g in tqdm.tqdm(reader, desc='Preliminary Data Pass'):
            num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['edges']] + [-1]) + 1)
            annotation_size = max(annotation_size, max(g['node_features']) + 1)
            label = g.get('label_seq', [g.get('label', 0)])
            num_classes = max(num_classes, max(label) + 1)
            max_depth = max(max_depth, len(label))

        self.params['num_edge_types'] = num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2)
        self.params['annotation_size'] = annotation_size
        self.params['num_classes'] = num_classes
        self.params['max_depth'] = max_depth
        reader.close()

    @abstractmethod
    def process_raw_graph(self, graph):
        pass

    @abstractmethod
    def define_placeholders(self):
        #  You probably want to modify the target_values placeholder to reflect a seq target
        super().define_placeholders()

    @abstractmethod
    def final_predictor(self):
        pass

    @abstractmethod
    def make_model(self, mode):
        pass

    @abstractmethod
    def save_interface(self, path: str):
        pass

    @abstractmethod
    def infer_top_k(self, pred, k=10):
        pass

    def infer(self, raw_graph_data):
        graphs = [self.process_raw_graph(g) for g in raw_graph_data]

        batch_iterator = utils.ThreadedIterator(self.make_minibatch_iterator(graphs, is_training=False),
                                                max_queue_size=50)

        preds = []

        for step, batch_data in enumerate(batch_iterator):
            batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
            fetch_list = [self.ops['preds'], self.ops['probs']]

            result = self.sess.run(fetch_list, feed_dict=batch_data)

            for p, v in zip(result[0], result[1]):
                pred_top_k = self.infer_top_k(list(zip(p, v)), 10000)
                preds.append(pred_top_k)

        return preds
