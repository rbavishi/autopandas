import collections
import json
import pickle
import os
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from abc import ABC, abstractmethod

from typing import Iterable

import tqdm
import yaml

from autopandas.utils.io import IndexedFileReader, IndexedFileWriter
from ggnn.utils import ParamsNamespace
from ggnn.models import utils


class BaseGGNN(ABC):
    def __init__(self):
        self.params: ParamsNamespace = ParamsNamespace()
        self.params.update(self.default_params())

    @classmethod
    def default_params(cls):
        return {
            'args': ParamsNamespace(),
            'num_epochs': 3000,
            'patience': 25,
            'learning_rate': 0.001,
            'clamp_gradient_norm': 1.0,
            'out_layer_dropout_keep_prob': 1.0,

            'hidden_size_node': 100,
            'hidden_size_final_mlp': 100,
            'num_timesteps': 4,

            'tie_fwd_bkwd': True,

            'random_seed': 0,
        }

    @classmethod
    def from_params(cls, params: ParamsNamespace):
        res = cls()
        res.params.update(params)
        return res

    # --------------------------------------------------------------- #
    #  Graph Construction
    # --------------------------------------------------------------- #

    def build_graph_model(self, mode='training', restore_file=None):
        possible_modes = ['training', 'testing', 'inference']
        if mode not in possible_modes:
            raise NotImplementedError("Mode has to be one of {}".format(", ".join(possible_modes)))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            tf.set_random_seed(self.params['random_seed'])
            self.placeholders = {}
            self.weights = {}
            self.ops = {}
            self.make_model(mode=mode)
            if mode == 'training':
                self.make_train_step()

            if restore_file is None:
                self.initialize_model()
            else:
                self.restore_model(restore_file)

    def initialize_model(self):
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)

    def restore_model(self, path: str) -> None:
        debug = self.params.args.get('debug', False)
        if debug:
            print("Restoring weights from file %s." % path)
        with open(path, 'rb') as in_file:
            data_to_load = pickle.load(in_file)

        variables_to_initialize = []
        with tf.name_scope("restore"):
            restore_ops = []
            used_vars = set()
            for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                used_vars.add(variable.name)
                if variable.name in data_to_load['weights']:
                    if debug:
                        print("Restoring weights for %s" % variable.name)
                    restore_ops.append(variable.assign(data_to_load['weights'][variable.name]))
                else:
                    raise LookupError('No saved value was found for %s' % variable.name)

            if debug:
                for var_name in data_to_load['weights']:
                    if var_name not in used_vars:
                        print('Saved weights for %s not used by model.' % var_name)

            restore_ops.append(tf.variables_initializer(variables_to_initialize))
            self.sess.run(restore_ops)

    def save_model(self, path: str) -> None:
        weights_to_save = {}
        for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            assert variable.name not in weights_to_save
            weights_to_save[variable.name] = self.sess.run(variable)

        data_to_save = {
            "weights": weights_to_save
        }

        with open(path, 'wb') as out_file:
            pickle.dump(data_to_save, out_file, pickle.HIGHEST_PROTOCOL)

        # --------------------------------------------------------------- #
        #  Model Definition
        # --------------------------------------------------------------- #

    def make_model(self, mode):
        self.define_placeholders()

        #  First, compute the node-level representations, after the message-passing algorithm
        with tf.variable_scope("graph_model"):
            self.prepare_specific_graph_model()
            self.ops['final_node_representations'] = self.compute_final_node_representations()

        with tf.variable_scope("out_layer"):
            # Should return logits with dimension equal to the number of output classes
            logits = self.prepare_final_layer()
            labels = self.placeholders['target_values']

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

            self.ops['loss'] = tf.reduce_mean(loss)
            probabilities = tf.nn.softmax(logits)

            correct_prediction = tf.equal(tf.argmax(probabilities, -1), self.placeholders['target_values'])
            self.ops['accuracy_task'] = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            top_k = tf.nn.top_k(probabilities, k=self.params['num_classes'])
            self.ops['preds'] = top_k.indices
            self.ops['probs'] = top_k.values
            self.ops['targets'] = self.placeholders['target_values']

    def define_placeholders(self):
        self.placeholders['target_values'] = tf.placeholder(tf.int64, [None], name='target_values')
        self.placeholders['num_graphs'] = tf.placeholder(tf.int32, [], name='num_graphs')
        self.placeholders['out_layer_dropout_keep_prob'] = tf.placeholder(tf.float32, [],
                                                                          name='out_layer_dropout_keep_prob')

    @abstractmethod
    def prepare_specific_graph_model(self):
        pass

    @abstractmethod
    def compute_final_node_representations(self) -> tf.Tensor:
        pass

    def prepare_final_layer(self):
        #  By default, pools up the node-embeddings (sum by default),
        #  and applies a simple MLP
        pooled = self.perform_pooling(self.ops['final_node_representations'])
        self.ops['final_predictor'] = self.final_predictor()
        return self.ops['final_predictor'](pooled)

    def perform_pooling(self, last_h):
        #  By default, it simply sums up the node embeddings
        #  We do not assume sorted segment_ids
        graph_node_sums = tf.unsorted_segment_sum(data=last_h,  # [v x h]
                                                  segment_ids=self.placeholders['graph_nodes_list'],
                                                  num_segments=self.placeholders['num_graphs'])  # [g x h]

        return graph_node_sums

    def final_predictor(self):
        #  By default, a simple MLP with one hidden layer
        return utils.MLP(self.params['hidden_size_node'], self.params['num_classes'],
                         [self.params['hidden_size_final_mlp']],
                         self.placeholders['out_layer_dropout_keep_prob'])

        # --------------------------------------------------------------- #
        #  Model Training Definition
        # --------------------------------------------------------------- #

    def make_train_step(self):
        trainable_vars = self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if self.params.args.freeze_graph_model:
            graph_vars = set(self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="graph_model"))
            filtered_vars = []
            for var in trainable_vars:
                if var not in graph_vars:
                    filtered_vars.append(var)
                else:
                    print("Freezing weights of variable %s." % var.name)
            trainable_vars = filtered_vars

        optimizer = tf.train.AdamOptimizer(self.params['learning_rate'])
        grads_and_vars = optimizer.compute_gradients(self.ops['loss'], var_list=trainable_vars)
        clipped_grads = []
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, self.params['clamp_gradient_norm']), var))
            else:
                clipped_grads.append((grad, var))

        self.ops['train_step'] = optimizer.apply_gradients(clipped_grads)

        # Initialize newly-introduced variables:
        self.sess.run(tf.local_variables_initializer())

    # --------------------------------------------------------------- #
    #  Controllers
    # --------------------------------------------------------------- #

    def run(self):
        if self.params.args.mode == 'train':
            self.run_training()

        elif self.params.args.mode == 'test':
            self.run_testing()

    def run_training(self):
        self.wdir: str = self.params.args.outdir
        if self.wdir is None:
            self.wdir = "_".join(["run", time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])
            os.system('rm -rf {}'.format(self.wdir))
            os.system('mkdir -p {}'.format(self.wdir))

        if not os.path.exists(self.wdir):
            os.system('mkdir -p {}'.format(self.wdir))

        #  Save away the params
        with open('{}/params.pkl'.format(self.wdir), 'wb') as f:
            pickle.dump(self.params, f)

        #  For readability
        with open('{}/params.txt'.format(self.wdir), 'w') as f:
            print(str(self.params), file=f)

        self.log_file = '{}/log.json'.format(self.wdir)
        self.best_model_file = '{}/model_best.pickle'.format(self.wdir)

        random.seed(self.params['random_seed'])
        np.random.seed(self.params['random_seed'])

        #  Setup important data-specific params if any (such as num_classes, num_edge_types etc.)
        self.preprocess_data(self.params.args.train, is_training_data=True)

        #  Save away the params
        with open('{}/params.pkl'.format(self.wdir), 'wb') as f:
            pickle.dump(self.params, f)

        #  Save away an interface
        self.save_interface('{}/interface.pkl'.format(self.wdir))

        #  For readability
        with open('{}/params.txt'.format(self.wdir), 'w') as f:
            print(str(self.params), file=f)

        #  Setup the model
        self.build_graph_model(mode='training', restore_file=self.params.args.get('restore', None))

        #  Load up the data
        self.train_data = self.load_data(self.params.args.train, is_training_data=True)
        self.valid_data = self.load_data(self.params.args.valid, is_training_data=False)

        self.train(self.train_data, self.valid_data)

    def run_testing(self):
        args = self.params.args

        #  Load the params
        with open('{}/params.pkl'.format(self.params.args.model), 'rb') as f:
            self.params = pickle.load(f)

        self.params.args = args

        self.model_dir: str = self.params.args.model
        model_path = '{}/model_best.pickle'.format(self.model_dir)

        self.save_interface('{}/interface.pkl'.format(self.model_dir))

        #  Set the results directory
        self.wdir: str = self.params.args.outdir
        if self.wdir is None:
            self.wdir = "_".join(["test", time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])
            os.system('rm -rf {}'.format(self.wdir))
            os.system('mkdir -p {}'.format(self.wdir))

        random.seed(self.params['random_seed'])
        np.random.seed(self.params['random_seed'])

        #  Load up the data
        self.test_data = self.load_data(self.params.args.test, is_training_data=False)

        #  Setup the model
        self.build_graph_model(mode='testing', restore_file=model_path)

        self.test(self.test_data)

    def setup_inference(self, model_dir: str):
        #  Load the params
        with open('{}/params.pkl'.format(model_dir), 'rb') as f:
            self.params = pickle.load(f)

        self.params.args = ParamsNamespace()

        self.model_dir: str = model_dir
        model_path = '{}/model_best.pickle'.format(self.model_dir)

        #  Load label mapping
        with open('{}/label_mapping.yml'.format(self.model_dir), 'r') as f:
            self.label_mapping = yaml.load(f)
            tmp = {}
            for k, v in self.label_mapping.items():
                if isinstance(k, (list, tuple)):
                    k = ":".join(k)

                tmp[k] = v
                tmp[v] = k

            self.label_mapping = tmp

        #  Setup the model
        self.build_graph_model(mode='testing', restore_file=model_path)

    # --------------------------------------------------------------- #
    #  Data Processing
    # --------------------------------------------------------------- #

    @abstractmethod
    def preprocess_data(self, path, is_training_data=False):
        pass

    def load_data(self, path, is_training_data=False):
        reader = IndexedFileReader(path)
        num_training_points = self.params.args.get('num_training_points', -1)
        if self.params.args.get('use_memory', False):
            result = self.process_raw_graphs(reader, is_training_data)
            reader.close()
            if is_training_data and num_training_points != -1:
                return result[:num_training_points]

            return result

        if self.params.args.get('use_disk', False):
            wpath = path + '.processed'

            if not os.path.exists(wpath):
                w = IndexedFileWriter(wpath)
                for d in tqdm.tqdm(reader, desc='Dumping processed graphs to disk'):
                    w.append(pickle.dumps(self.process_raw_graph(d)))

                w.close()
                reader.close()

            result = IndexedFileReader(wpath)

            if is_training_data and self.params.get('load_shuffle', False):
                result.shuffle()

            if is_training_data and num_training_points != -1:
                result.indices = result.indices[:num_training_points]

            return result

        #  We won't pre-process anything. We'll convert on-the-fly. Saves memory but is very slow and wasteful
        if is_training_data and self.params.get('load_shuffle', False):
            reader.shuffle()

        if is_training_data and num_training_points != -1:
            reader.indices = reader.indices[:num_training_points]

        reader.set_loader(lambda x: self.process_raw_graph(pickle.load(x)))
        return reader

    def process_raw_graphs(self, raw_data: Iterable, is_training_data: bool = False):
        processed_graphs = []
        for d in tqdm.tqdm(raw_data, desc='Processing Raw Data'):
            processed_graphs.append(self.process_raw_graph(d))

        if is_training_data and self.params.args.get('load_shuffle', False):
            np.random.shuffle(processed_graphs)

        return processed_graphs

    @abstractmethod
    def process_raw_graph(self, graph):
        pass

    @abstractmethod
    def make_minibatch_iterator(self, data, is_training: bool):
        pass

    @abstractmethod
    def save_interface(self, path: str):
        pass

    # --------------------------------------------------------------- #
    #  Operation Modes
    # --------------------------------------------------------------- #

    # --------------------------------------------------------------- #
    #  Training
    # --------------------------------------------------------------- #

    def train(self, train_data, valid_data):
        log_to_save = []
        total_time_start = time.time()
        with self.graph.as_default():
            if self.params.args.get('restore', None) is not None:
                _, valid_acc, _ = self.run_train_epoch("Resumed (validation)", valid_data, False)
                best_val_acc = valid_acc
                best_val_acc_epoch = 0
                print("\r\x1b[KResumed operation, initial cum. val. acc: %.5f" % best_val_acc)
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

                if valid_acc > best_val_acc:
                    self.save_model(self.best_model_file)
                    print("  (Best epoch so far, cum. val. acc increased to %.5f from %.5f. Saving to '%s')" % (
                        valid_acc, best_val_acc, self.best_model_file))
                    best_val_acc = valid_acc
                    best_val_acc_epoch = epoch

                elif epoch - best_val_acc_epoch >= self.params['patience']:
                    print("Stopping training after %i epochs without improvement on validation accuracy." % self.params[
                        'patience'])
                    break

    def run_train_epoch(self, epoch_name: str, data, is_training: bool):
        loss = 0
        accuracy = 0
        accuracy_op = self.ops['accuracy_task']
        start_time = time.time()
        processed_graphs = 0
        batch_iterator = utils.ThreadedIterator(self.make_minibatch_iterator(data, is_training), max_queue_size=50)
        for step, batch_data in enumerate(batch_iterator):
            num_graphs = batch_data[self.placeholders['num_graphs']]
            processed_graphs += num_graphs
            if is_training:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = self.params[
                    'out_layer_dropout_keep_prob']
                fetch_list = [self.ops['loss'], accuracy_op, self.ops['train_step']]
            else:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
                fetch_list = [self.ops['loss'], accuracy_op]

            result = self.sess.run(fetch_list, feed_dict=batch_data)
            (batch_loss, batch_accuracy) = (result[0], result[1])
            loss += batch_loss * num_graphs
            accuracy += batch_accuracy * num_graphs

            print("Running %s, batch %i (has %i graphs). "
                  "Loss so far: %.4f. Accuracy so far: %.4f" % (epoch_name,
                                                                step,
                                                                num_graphs,
                                                                loss / processed_graphs,
                                                                accuracy / processed_graphs),
                  end='\r')

        accuracy = accuracy / processed_graphs
        loss = loss / processed_graphs
        instance_per_sec = processed_graphs / (time.time() - start_time)
        return loss, accuracy, instance_per_sec

        # --------------------------------------------------------------- #
        #  Testing
        # --------------------------------------------------------------- #

    def test(self, test_data):
        with self.graph.as_default():
            result = self.run_test_epoch(test_data, perform_analysis=self.params.args.analysis)
            valid_loss, valid_acc, valid_speed, preds, targets = result

            accs_str = "%.5f" % valid_acc
            print("\r\x1b[K Valid: loss: %.5f | acc: %s | instances/sec: %.2f" % (valid_loss,
                                                                                  accs_str,
                                                                                  valid_speed))

            if self.params.args.analysis:
                self.perform_analysis(preds, targets)

    def perform_analysis(self, preds, targets):
        #  Apply the label mapping
        label_map = {}
        with open(self.params.args.label_mapping, 'r') as f:
            seq_to_int = yaml.load(f)
            for k, v in seq_to_int.items():
                label_map[v] = k

        for pred in tqdm.tqdm(preds, desc='Mapping predictions'):
            for i in range(len(pred)):
                pred[i] = (label_map[pred[i][0]], pred[i][1])

        targets = list(map(lambda x: label_map[x], targets))

        #  First, just dump the preds as is, along with the targets
        with open('{}/preds.pkl'.format(self.wdir), 'wb') as f:
            pickle.dump(list(zip(targets, preds)), f)

        top_k = []
        for target, pred in zip(targets, preds):
            top_k.append([int(i[0] == target) for i in pred[:self.params.args.top_k]])

        top_k = np.array(top_k)
        total = len(preds)
        top_k_acc = np.cumsum(np.sum(top_k, axis=0)) / total
        for i, acc in enumerate(top_k_acc, 1):
            print("Top-{} Accuracy : {:.4f}".format(i, acc))

        #  Class-specific accuracy
        class_top_k = collections.defaultdict(list)
        for target, pred in zip(targets, preds):
            class_top_k[target].append([int(i[0] == target) for i in pred[:self.params.args.top_k]])

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

    def run_test_epoch(self, data, perform_analysis=False):
        loss = 0
        accuracy = 0
        accuracy_op = self.ops['accuracy_task']
        start_time = time.time()
        processed_graphs = 0
        batch_iterator = utils.ThreadedIterator(self.make_minibatch_iterator(data, False), max_queue_size=5)
        preds = []
        targets = []

        for step, batch_data in enumerate(batch_iterator):
            num_graphs = batch_data[self.placeholders['num_graphs']]
            processed_graphs += num_graphs
            batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
            if perform_analysis:
                fetch_list = [self.ops['loss'], accuracy_op,
                              self.ops['preds'], self.ops['probs'], self.ops['targets']]
            else:
                fetch_list = [self.ops['loss'], accuracy_op]

            result = self.sess.run(fetch_list, feed_dict=batch_data)
            batch_loss, batch_accuracy = result[0], result[1]
            loss += batch_loss * num_graphs
            accuracy += batch_accuracy * num_graphs

            print("Running Test, batch %i (has %i graphs). "
                  "Loss so far: %.4f. Accuracy so far: %.4f" % (step,
                                                                num_graphs,
                                                                loss / processed_graphs,
                                                                accuracy / processed_graphs),
                  end='\r')

            if perform_analysis:
                for p, v in zip(result[2], result[3]):
                    preds.append(list(zip(p, v)))

                targets += list(result[4])

        accuracy = accuracy / processed_graphs
        loss = loss / processed_graphs
        instance_per_sec = processed_graphs / (time.time() - start_time)
        return loss, accuracy, instance_per_sec, preds, targets

        # --------------------------------------------------------------- #
        #  Inference
        # --------------------------------------------------------------- #

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
                preds.append(list(map(lambda x: (self.label_mapping[x[0]], x[1]), zip(p, v))))

        return preds
