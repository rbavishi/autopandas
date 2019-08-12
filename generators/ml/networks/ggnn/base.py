import collections
import json
import os
import pickle
import random
import time

import tensorflow as tf
import numpy as np
import tqdm

from abc import abstractmethod, ABC
from typing import Dict

from autopandas_v2.generators.ml.networks.ggnn import utils
from autopandas_v2.utils.ioutils import IndexedFileReader
from autopandas_v2.ml.networks.ggnn.models.sparse.base import SparseGGNN
from autopandas_v2.ml.networks.ggnn.utils import ParamsNamespace


class BaseSmartGenGGNN(SparseGGNN, ABC):
    def preprocess_data(self, path, is_training_data=False):
        reader = IndexedFileReader(path)

        num_fwd_edge_types = 0
        annotation_size = 0
        for g in tqdm.tqdm(reader, desc='Preliminary Data Pass', dynamic_ncols=True):
            num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['edges']] + [-1]) + 1)
            annotation_size = max(annotation_size, max(g['node_features']) + 1)

        self.params['num_edge_types'] = num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2)
        self.params['annotation_size'] = annotation_size
        reader.close()

    def define_placeholders(self):
        super().define_placeholders()
        self.placeholders['num_graphs_in_batch'] = tf.placeholder(tf.int32, None, name='num_graphs_in_batch')

    @abstractmethod
    def process_raw_graph(self, graph):
        (adjacency_lists, num_incoming_edge_per_type) = self.graph_to_adjacency_lists(graph)
        return {"adjacency_lists": adjacency_lists,
                "num_incoming_edge_per_type": num_incoming_edge_per_type,
                "init": self.to_one_hot(graph["node_features"], self.params['annotation_size'])
                }

    @abstractmethod
    def make_model(self, mode):
        self.define_placeholders()

        #  First, compute the node-level representations, after the message-passing algorithm
        with tf.variable_scope("graph_model"):
            self.prepare_specific_graph_model()
            self.ops['final_node_representations'] = self.compute_final_node_representations()

    def prepare_pooled_node_representations(self):
        pooled = self.perform_pooling(self.ops['final_node_representations'])
        self.ops['pooled_node_representations'] = pooled
        return pooled

    def per_graph_custom_minibatch_iterator(self, graph_num: int, graph: Dict, node_offset: int) -> Dict:
        return {}

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
            batch_adjacency_lists = [[] for _ in range(self.params['num_edge_types'])]
            batch_num_incoming_edges_per_type = []
            batch_graph_nodes_list = []
            batch_per_graph_custom = collections.defaultdict(list)
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

                for k, v in self.per_graph_custom_minibatch_iterator(num_graphs_in_batch, data[num_graphs],
                                                                     node_offset).items():
                    batch_per_graph_custom[k].append(v)

                num_graphs += 1
                num_graphs_in_batch += 1
                node_offset += num_nodes_in_graph

            batch_feed_dict = {
                self.placeholders['initial_node_representation']: np.array(batch_node_features),
                self.placeholders['num_incoming_edges_per_type']: np.concatenate(batch_num_incoming_edges_per_type,
                                                                                 axis=0),
                self.placeholders['graph_nodes_list']: np.concatenate(batch_graph_nodes_list),
                self.placeholders['num_graphs']: num_graphs_in_batch,
                self.placeholders['graph_state_keep_prob']: state_dropout_keep_prob,
                self.placeholders['edge_weight_dropout_keep_prob']: edge_weights_dropout_keep_prob,
                self.placeholders['num_graphs_in_batch']: num_graphs_in_batch
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

    @abstractmethod
    def infer(self, raw_graph_data, **kwargs):
        pass

    def setup_inference(self, model_dir: str):
        #  Load the params
        with open('{}/params.pkl'.format(model_dir), 'rb') as f:
            self.params = pickle.load(f)

        self.params.args = ParamsNamespace()

        self.model_dir: str = model_dir
        model_path = '{}/model_best.pickle'.format(self.model_dir)

        #  Setup the model
        self.build_graph_model(mode='testing', restore_file=model_path)

    @abstractmethod
    def save_interface(self, path: str):
        pass

    def train(self, train_data, valid_data):
        log_to_save = []
        total_time_start = time.time()
        with self.graph.as_default():
            if self.params.args.get('restore', None) is not None:
                _, valid_acc, _ = self.run_train_epoch("Resumed (validation)", valid_data, False)
                best_val_acc = valid_acc
                best_val_acc_epoch = 0
                print("\r\x1b[KResumed operation, initial cum. val. acc: %.5f" % best_val_acc)
                restore_path = self.params.args['restore']
                prev_log_path = restore_path.replace(os.path.basename(self.best_model_file),
                                                     os.path.basename(self.log_file))
                if os.path.exists(prev_log_path):
                    with open(prev_log_path, 'r') as f:
                        log_to_save = json.load(f)
            else:
                (best_val_acc, best_val_acc_epoch) = (0.0, 0)

            for epoch in range(1, self.params.args.get('num_epochs', self.params['num_epochs']) + 1):
                print("== Epoch %i" % epoch)
                train_loss, train_acc, train_speed = self.run_train_epoch("epoch %i (training)" % epoch,
                                                                          train_data, True)
                accs_str = "%.5f" % train_acc
                print("\r\x1b[K Train: loss: %.5f | acc: %s | instances/sec: %.2f" % (train_loss,
                                                                                      accs_str,
                                                                                      train_speed))
                valid_loss, valid_acc, valid_speed = self.run_train_epoch("epoch %i (validation)" % epoch,
                                                                          valid_data, False)
                accs_str = "%.5f" % valid_acc
                print("\r\x1b[K Valid: loss: %.5f | acc: %s | instances/sec: %.2f" % (valid_loss,
                                                                                      accs_str,
                                                                                      valid_speed))

                epoch_time = time.time() - total_time_start
                log_entry = {
                    'epoch': epoch,
                    'time': epoch_time,
                    'train_results': (train_loss, train_acc, train_speed),
                    'valid_results': (valid_loss, valid_acc, valid_speed),
                }
                log_to_save.append(log_entry)
                with open(self.log_file, 'w') as f:
                    json.dump(log_to_save, f, indent=4)

                patience = self.params.args.get('patience', self.params['patience'])
                #  Be dynamic, if the accuracy is high, don't wait too much
                #  TODO : Better to use a more formal function
                if valid_acc >= 0.90:
                    patience = min(patience, 15)
                if valid_acc >= 0.95:
                    patience = min(patience, 10)
                if valid_acc >= 0.97:
                    patience = min(patience, 5)
                if valid_acc >= 0.98:
                    patience = min(patience, 3)
                if valid_acc >= 0.99:
                    patience = min(patience, 2)
                if valid_acc >= 0.995:
                    patience = min(patience, 1)

                if valid_acc > best_val_acc:
                    self.save_model(self.best_model_file)
                    print("  (Best epoch so far, cum. val. acc increased to %.5f from %.5f. Saving to '%s')" % (
                        valid_acc, best_val_acc, self.best_model_file))
                    best_val_acc = valid_acc
                    best_val_acc_epoch = epoch

                elif epoch - best_val_acc_epoch >= patience:
                    print("Stopping training after %i epochs without improvement on validation accuracy." % patience)
                    break

    def init_analysis(self):
        self.total_loss = 0
        self.num_graphs = 0

    def finish_analysis(self):
        print("Number of Graphs :", self.num_graphs)
        print("Average Loss :", self.total_loss / self.num_graphs)
        return {'Num Graphs': self.num_graphs, 'Average Loss': self.total_loss / self.num_graphs}

    def run_analysis(self, path):
        args = self.params.args

        #  Load the params
        with open('{}/params.pkl'.format(self.params.args.model), 'rb') as f:
            self.params.update(pickle.load(f))

        self.params.args = args

        self.model_dir: str = self.params.args.model
        model_path = '{}/model_best.pickle'.format(self.model_dir)

        random.seed(self.params['random_seed'])
        np.random.seed(self.params['random_seed'])

        #  Load up the data
        test_data = self.load_data(path, is_training_data=False)

        #  Setup the model
        self.build_graph_model(mode='testing', restore_file=model_path)
        self.init_analysis()
        return self.analyze(test_data)

    def analyze(self, test_data):
        with self.graph.as_default():
            batch_iterator = utils.ThreadedIterator(self.make_minibatch_iterator(test_data, False), max_queue_size=5)
            fetch_list = self.get_fetch_list()
            processed_graphs = 0

            for step, batch_data in enumerate(batch_iterator):
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
                num_graphs = batch_data[self.placeholders['num_graphs_in_batch']]
                processed_graphs += num_graphs
                print("Running Analysis, batch {} (has {} graphs).".format(step, num_graphs), end='\r')

                result = self.sess.run(fetch_list, feed_dict=batch_data)
                self.analyze_result(num_graphs, result)

        print("\n-----\n")
        return self.finish_analysis()

    @abstractmethod
    def get_fetch_list(self):
        return [self.ops['loss']]

    @abstractmethod
    def analyze_result(self, num_graphs, result):
        self.total_loss += result[0] * num_graphs
        self.num_graphs += num_graphs
