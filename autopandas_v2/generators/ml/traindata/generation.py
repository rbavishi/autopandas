import collections
import itertools
import logging
import os
import pickle
import sys
import time
from typing import Dict, Any, List, Generator, Set, Tuple

import pebble
from autopandas_v2.generators.base import BaseGenerator
from autopandas_v2.generators.ml.traindata.exploration.engines.functions import RandProgEngine
from autopandas_v2.generators.ml.traindata.exploration.iospecs import ExplorationSpec, GeneratorInversionSpec
from autopandas_v2.generators.utils import load_generators, load_randomized_generators
from autopandas_v2.iospecs import ArgTrainingSpec
from autopandas_v2.ml.featurization.featurizer import RelationGraph
from autopandas_v2.ml.featurization.options import GraphOptions
from autopandas_v2.synthesis.search.results.programs import Program
from autopandas_v2.utils import misc, logger
from autopandas_v2.utils.cli import ArgNamespace
from autopandas_v2.utils.exceptions import SilentException
from autopandas_v2.utils.ioutils import IndexedFileWriter, IndexedFileReader
from concurrent.futures import TimeoutError


class RawDataGenerator:
    """
    This generator implements parallelized computation of raw training data that contains random I/O examples
    along with the programs producing the output, and the generator choices made in producing that program
    """

    class Worker:
        args: ArgNamespace = None
        generators: Dict[str, BaseGenerator] = None

        @classmethod
        def init(cls, args: ArgNamespace):
            cls.args = args
            cls.generators = load_randomized_generators()
            if cls.args.debug:
                logger.info("Loaded {} generators in process {}".format(len(cls.generators), os.getpid()))

        @classmethod
        def process(cls, named_seqs: List[List[str]]):
            if named_seqs is None:
                return 0, None

            seqs: List[List[BaseGenerator]] = [list(map(lambda x: cls.generators[x], s)) for s in named_seqs]
            max_seq_trials = cls.args.max_seq_trials
            results: List[Dict] = []

            for idx, seq in enumerate(seqs):
                engine = RandProgEngine(seq, cls.args)
                for trial in range(max_seq_trials):
                    try:
                        spec: ExplorationSpec = engine.generate()
                    except Exception as e:
                        if cls.args.debug:
                            logger.warn("Encountered exception for", named_seqs[idx])
                            logger.log(e)
                            logging.exception(e)

                        continue

                    if spec is None:
                        continue

                    dpoint = {
                        'inputs': spec.inputs,
                        'output': spec.output,
                        'intermediates': spec.intermediates,
                        'program_str': str(spec.program),
                        'program': spec.program,
                        'function_sequence': named_seqs[idx],
                        'generator_tracking': spec.tracking
                    }

                    # print("-" * 50)
                    # print(dpoint)
                    # print("-" * 50)
                    # print([t.record for t in spec.tracking])
                    # print(spec.program)

                    #  Confirm it's picklable. Sometimes, unpickling throws an error
                    #  when the main process is receiving the msg, and things break down
                    #  in a very, very nasty manner
                    #  TODO : Can we switch to dill while using multiprocessing/pebble?
                    try:
                        a = pickle.dumps(dpoint)
                        pickle.loads(a)
                    except:
                        continue

                    results.append(dpoint)
                    break

            return len(named_seqs), results

    def __init__(self, args: ArgNamespace):
        self.args = args
        self.fwriter: IndexedFileWriter = None
        self.blacklist: Set[Tuple[str]] = set()
        self.whitelist: Set[Tuple[str]] = set()
        self.error_cnt_map: Dict[Tuple[str], int] = collections.defaultdict(int)
        self.sequences: List[List[str]] = None

    def load_sequences(self) -> List[List[str]]:
        generators: Dict[str, BaseGenerator] = load_randomized_generators()
        generator_name_map: Dict[str, List[str]] = collections.defaultdict(list)
        for k, v in generators.items():
            generator_name_map[v.name].append(v.qual_name)
            generator_name_map[v.qual_name].append(v.qual_name)

        sequences_src: str = self.args.sequences
        unimplemented_funcs: Set[str] = set()
        if sequences_src.endswith(".pkl"):
            with open(sequences_src, 'rb') as f:
                sequences: List[List[str]] = list(map(list, pickle.load(f)))

        else:
            sequences: List[List[str]] = [list(i.split(':')) for i in sequences_src.split(',')]

        def get_valid_sequences(seq: List[str]):
            for i in seq:
                if i not in generator_name_map:
                    unimplemented_funcs.add(i)
                    return

            if not (self.args.min_depth <= len(seq) <= self.args.max_depth):
                return

            for seq in itertools.product(*[generator_name_map[i] for i in seq]):
                yield list(seq)

        final_sequences: List[List[str]] = []
        for seq in sequences:
            final_sequences.extend(get_valid_sequences(seq))

        for i in unimplemented_funcs:
            logger.warn("Generator not implemented for : {}".format(i))

        logger.info("Found {} sequences. "
                    "Filtered out {}. "
                    "Returning {}.".format(len(sequences), len(sequences) - len(final_sequences), len(final_sequences)))
        return final_sequences

    def gen_named_seqs(self) -> Generator[List[List[str]], Any, Any]:
        while True:
            self.blacklist -= self.whitelist
            if len(self.blacklist) > 0:
                for seq in self.blacklist:
                    logger.warn("Blacklisting {} because of too many errors".format(seq))

                self.sequences = [i for i in self.sequences if tuple(i) not in self.blacklist]
                self.blacklist = set()

            for seq in self.sequences:
                yield [seq]

            if self.args.no_repeat:
                break

    def init(self):
        if os.path.exists(self.args.outfile) and self.args.append:
            self.fwriter = IndexedFileWriter(self.args.outfile, mode='a')
        else:
            self.fwriter = IndexedFileWriter(self.args.outfile, mode='w')

        self.blacklist = set()
        self.whitelist = set()
        self.error_cnt_map = collections.defaultdict(int)

    def process_dpoint(self, dpoint: Dict):
        self.fwriter.append(pickle.dumps(dpoint))

    def report_error_seqs(self, seqs: List[List[str]]):
        if seqs is None:
            return

        for seq in seqs:
            key = tuple(seq)
            if key in self.whitelist:
                continue

            self.error_cnt_map[key] += self.args.max_seq_trials
            if self.error_cnt_map[key] > self.args.blacklist_threshold:
                self.blacklist.add(key)

    def generate(self):
        self.init()
        num_generated = 0
        num_processed = 0
        num_required = self.args.num_training_points
        self.sequences = self.load_sequences()
        start_time = time.time()
        speed = 0
        time_remaining = 'inf'

        with pebble.ProcessPool(max_workers=self.args.processes, initializer=RawDataGenerator.Worker.init,
                                initargs=(self.args,)) as p:

            #  First do smaller chunksizes to allow the blacklist to take effect
            chunksize = self.args.processes * self.args.chunksize

            if self.args.blacklist_threshold == -1:
                chunksize_blacklist = chunksize
            else:
                chunksize_blacklist = max((self.args.blacklist_threshold // self.args.max_seq_trials), 1) * len(
                    self.sequences)

            for chunk in misc.grouper([chunksize_blacklist, chunksize], self.gen_named_seqs()):
                if not p.active:
                    break

                future = p.map(RawDataGenerator.Worker.process, chunk, timeout=self.args.task_timeout)
                res_iter = future.result()

                idx = -1
                while True:
                    idx += 1
                    if num_generated >= num_required:
                        p.stop()
                        try:
                            p.join(10)
                        except:
                            pass
                        break

                    try:
                        returned = next(res_iter)
                        if returned is None:
                            self.report_error_seqs(chunk[idx])
                            continue

                        num_input_seqs, results = returned
                        num_processed += num_input_seqs
                        if results is not None and len(results) > 0:
                            for seq in chunk[idx]:
                                self.whitelist.add(tuple(seq))

                            for result in results:
                                num_generated += 1
                                self.process_dpoint(result)

                            speed = round(num_generated / (time.time() - start_time), 1)
                            time_remaining = round((num_required - num_generated) / speed, 1)

                        elif num_input_seqs > 0:
                            self.report_error_seqs(chunk[idx])

                        logger.log("Num Generated : {} ({}/s, TTC={}s)".format(num_generated,
                                                                               speed,
                                                                               time_remaining), end='\r')

                    except StopIteration:
                        break

                    except TimeoutError as error:
                        pass

                    except Exception as e:
                        logger.warn("Failed for", chunk[idx])

            p.stop()
            try:
                p.join(10)
            except:
                pass

        self.fwriter.close()
        logger.log("\n-------------------------------------------------")
        logger.info("Total Time : {:.2f}s".format(time.time() - start_time))
        logger.info("Number of sequences processed :", num_processed)
        logger.info("Number of training points generated :", num_generated)


class ArgDataGenerator:
    """
    This generator implements parallelized computation of argument-level training data for training smart generators.
    It processes the raw I/O example data generated for training the function-sequence predictor, and tries to
    infer the choices made in the generator for each function and outputs that as training data.

    The inversion logic can be found in the definition of the DSL operators. The controller is present in BaseGenerator
    """

    class Worker:
        args: ArgNamespace = None
        generators: Dict[str, BaseGenerator] = None

        @classmethod
        def init(cls, args: ArgNamespace):
            cls.args = args
            cls.generators = load_generators()
            logger.info("Loaded {} generators in process {}".format(len(cls.generators), os.getpid()))

        @classmethod
        def process_without_tracking(cls, raw_data: Dict):
            #  TODO : Fix this
            if len(raw_data['prog_seq']) > 1:
                logger.warn("Training data for smart generators "
                            "does not support len-{} data right now".format(len(raw_data['prog_seq'])))
                return

            fn = raw_data['prog_seq'][0]

            if fn not in cls.generators:
                logger.warn("Generator not defined for {}".format(fn))
                return

            fn_args: Dict[str, Any] = {}
            #  The AutoPandas-v1 code stores positional and keyword argument values in two separate dicts
            fn_args.update(raw_data['args'][0][0])
            fn_args.update(raw_data['args'][0][1])

            spec: ArgTrainingSpec = ArgTrainingSpec(raw_data['inputs'], raw_data['output'], fn_args,
                                                    max_depth=1)

            return fn, cls.generators[fn].generate_arguments_training_data(spec)

        @classmethod
        def process_with_tracking(cls, raw_data: Dict):
            spec: GeneratorInversionSpec = GeneratorInversionSpec(raw_data['inputs'], raw_data['output'],
                                                                  raw_data['intermediates'],
                                                                  raw_data['generator_tracking'])

            results: List[Tuple[str, Dict[str, List[Any]]]] = []
            # print(raw_data['program'])
            # print([t.record for t in raw_data['generator_tracking']])
            for depth, fn in enumerate(raw_data['function_sequence'], 1):
                if fn not in cls.generators:
                    logger.warn("Generator not defined for {}".format(fn), use_cache=True)
                    continue

                try:
                    tracker = spec.trackers[depth - 1]
                    results.append((fn,
                                    cls.generators[fn].generate_arguments_training_data(spec,
                                                                                        depth=depth,
                                                                                        tracker=tracker)))
                except SilentException as e:
                    pass

                except Exception as e:
                    logger.err("Encountered Exception for {}".format(fn))
                    logging.exception(e)

            return results

        @classmethod
        def process(cls, raw_data: Dict):
            if raw_data is None:
                return

            if 'generator_tracking' in raw_data:
                return cls.process_with_tracking(raw_data)
            else:
                return cls.process_without_tracking(raw_data)

    def __init__(self, args: ArgNamespace):
        self.args = args
        self.file_map: Dict[str, Dict[str, IndexedFileWriter]] = None

    def raw_data_iterator(self):
        def valid(dpoint):
            for depth, record in enumerate(dpoint['generator_tracking']):
                record = record.record
                for k, v in record.items():
                    if k.startswith("ext_") and v['source'] == 'intermediates' and v['idx'] >= depth:
                        return False

            return True

        with open(self.args.raw_data_path, 'rb') as f:
            while True:
                try:
                    point = pickle.load(f)
                    if 'args' not in point and 'generator_tracking' not in point:
                        logger.warn("Raw data points are missing the 'args' attribute. Did you generate this "
                                    "data using the smart-generators branch of autopandas?")
                        return

                    if valid(point):
                        yield point

                except EOFError:
                    break

    def init(self):
        if (not os.path.exists(self.args.outdir)) or self.args.force:
            os.system('rm -rf {}'.format(self.args.outdir))
            os.system('mkdir -p {}'.format(self.args.outdir))

        if not os.path.exists(self.args.outdir):
            logger.err("Failed to create output directory at {}".format(self.args.outdir))
            sys.exit(1)

        self.file_map: Dict[str, Dict[str, IndexedFileWriter]] = {}

    def create_outdir_for_fname(self, fname: str):
        fn_dir = self.args.outdir + '/' + fname
        if not os.path.exists(fn_dir):
            os.system('mkdir -p {}'.format(fn_dir))

        self.file_map[fname] = {}

    def create_outfile_for_arg(self, fname: str, identifier: str):
        path = '{}/{}/{}.pkl'.format(self.args.outdir, fname, identifier)
        if os.path.exists(path) and self.args.append_arg_level:
            self.file_map[fname][identifier] = IndexedFileWriter(path, mode='a')
        else:
            self.file_map[fname][identifier] = IndexedFileWriter(path, mode='w')

    def process_result(self, fn, training_points: Dict[str, List[Any]]):
        if fn not in self.file_map:
            self.create_outdir_for_fname(fn)

        fmap = self.file_map[fn]
        for k, v in training_points.items():
            if k not in fmap:
                self.create_outfile_for_arg(fn, k)

            for i in v:
                fmap[k].append(pickle.dumps(i))

    def generate(self):
        self.init()
        num_generated = 0
        num_processed = 0
        num_raw_points = -1
        if os.path.exists(self.args.raw_data_path + '.index'):
            reader = IndexedFileReader(self.args.raw_data_path)
            num_raw_points = len(reader)
            reader.close()

        start_time = time.time()
        with pebble.ProcessPool(max_workers=self.args.processes, initializer=ArgDataGenerator.Worker.init,
                                initargs=(self.args,)) as p:

            chunksize = self.args.processes * self.args.chunksize
            for chunk in misc.grouper(chunksize, self.raw_data_iterator()):
                future = p.map(ArgDataGenerator.Worker.process, chunk, timeout=self.args.task_timeout)
                res_iter = future.result()

                idx = -1
                while True:
                    idx += 1
                    if idx < len(chunk) and chunk[idx] is not None:
                        num_processed += 1

                    try:
                        results = next(res_iter)
                        if chunk[idx] is None:
                            continue

                        if results is not None:
                            for fname, points in results:
                                if len(points) > 0:
                                    num_generated += 1

                                self.process_result(fname, points)

                    except StopIteration:
                        break

                    except TimeoutError as error:
                        pass

                    except Exception as e:
                        try:
                            logger.warn("Failed for", chunk[idx])
                            logging.exception(e)

                        except:
                            pass

                    finally:

                        speed = round(num_processed / (time.time() - start_time), 1)
                        if num_raw_points != -1:
                            time_remaining = round((num_raw_points - num_processed) / speed, 1)
                        else:
                            time_remaining = '???'

                        logger.log("Generated/Processed : {}/{} ({}/s, TTC={}s)".format(num_generated, num_processed,
                                                                                        speed,
                                                                                        time_remaining), end='\r')

            p.stop()
            try:
                p.join(10)
            except:
                pass

        for k, v in self.file_map.items():
            for fwriter in v.values():
                fwriter.close()

        logger.log("\n-------------------------------------------------")
        logger.info("Total Time : {:.2f}s".format(time.time() - start_time))
        logger.info("Generated {} training points from {} raw data points".format(num_generated, num_processed))


class FunctionSeqDataGenerator:
    """
    This generator implements parallelized computation of training data for training function sequence predictors.
    """

    class Worker:
        args: ArgNamespace = None
        generators: Dict[str, BaseGenerator] = None

        @classmethod
        def init(cls, args: ArgNamespace):
            cls.args = args
            cls.generators = load_generators()
            if cls.args.debug:
                logger.info("Loaded {} generators in process {}".format(len(cls.generators), os.getpid()))

        @classmethod
        def process(cls, raw_data: Dict):
            if raw_data is None:
                return None

            try:
                graph = RelationGraph(GraphOptions())
                inputs = raw_data['inputs']
                output = raw_data['output']
                graph.from_input_output(inputs, output)

                encoding = graph.get_encoding()
                encoding['label'] = raw_data['function_sequence']
                return encoding

            except SilentException:
                return None

            except Exception as e:
                try:
                    logger.warn("Failed for {}".format(raw_data))
                    logging.exception(e)
                    return None

                except:
                    pass

                return None

    def __init__(self, args: ArgNamespace):
        self.args = args
        self.fwriter: IndexedFileWriter = None

    def raw_data_iterator(self):
        def valid(dpoint):
            for depth, record in enumerate(dpoint['generator_tracking']):
                record = record.record
                for k, v in record.items():
                    if k.startswith("ext_") and v['source'] == 'intermediates' and v['idx'] >= depth:
                        return False

            return True

        with open(self.args.raw_data_path, 'rb') as f:
            while True:
                try:
                    point = pickle.load(f)
                    if 'args' not in point and 'generator_tracking' not in point:
                        logger.warn("Raw data points are missing the 'args' attribute. Did you generate this "
                                    "data using the smart-generators branch of autopandas?")
                        return

                    if valid(point):
                        yield point

                except EOFError:
                    break

    def init(self):
        if os.path.exists(self.args.outfile) and self.args.append:
            self.fwriter = IndexedFileWriter(self.args.outfile, mode='a')
        else:
            self.fwriter = IndexedFileWriter(self.args.outfile, mode='w')

    def process_result(self, training_point: Dict):
        self.fwriter.append(pickle.dumps(training_point))

    def generate(self):
        self.init()
        num_generated = 0
        num_processed = 0
        num_raw_points = -1
        if os.path.exists(self.args.raw_data_path + '.index'):
            reader = IndexedFileReader(self.args.raw_data_path)
            num_raw_points = len(reader)
            reader.close()

        start_time = time.time()
        with pebble.ProcessPool(max_workers=self.args.processes, initializer=FunctionSeqDataGenerator.Worker.init,
                                initargs=(self.args,)) as p:

            chunksize = self.args.processes * self.args.chunksize
            for chunk in misc.grouper(chunksize, self.raw_data_iterator()):
                future = p.map(FunctionSeqDataGenerator.Worker.process, chunk, timeout=self.args.task_timeout)
                res_iter = future.result()

                idx = -1
                while True:
                    idx += 1
                    if idx < len(chunk) and chunk[idx] is not None:
                        num_processed += 1

                    try:
                        result = next(res_iter)
                        if chunk[idx] is None:
                            continue

                        if result is not None:
                            self.process_result(result)
                            num_generated += 1

                    except StopIteration:
                        break

                    except TimeoutError as error:
                        pass

                    except Exception as e:
                        try:
                            logger.warn("Failed for", chunk[idx])
                            logging.exception(e)

                        except:
                            pass

                    finally:

                        speed = round(num_processed / (time.time() - start_time), 1)
                        if num_raw_points != -1:
                            time_remaining = round((num_raw_points - num_processed) / speed, 1)
                        else:
                            time_remaining = '???'

                        logger.log("Generated/Processed : {}/{} ({}/s, TTC={}s)".format(num_generated, num_processed,
                                                                                        speed,
                                                                                        time_remaining), end='\r')

            p.stop()
            try:
                p.join(10)
            except:
                pass

        self.fwriter.close()

        logger.log("\n-------------------------------------------------")
        logger.info("Total Time : {:.2f}s".format(time.time() - start_time))
        logger.info("Generated {} training points from {} raw data points".format(num_generated, num_processed))


class NextFunctionDataGenerator(FunctionSeqDataGenerator):
    """
    This generator implements parallelized computation of training data for training next function predictors.
    """

    class Worker:
        args: ArgNamespace = None
        generators: Dict[str, BaseGenerator] = None

        @classmethod
        def init(cls, args: ArgNamespace):
            cls.args = args
            cls.generators = load_generators()
            if cls.args.debug:
                logger.info("Loaded {} generators in process {}".format(len(cls.generators), os.getpid()))

        @classmethod
        def process(cls, raw_data: Dict):
            if raw_data is None:
                return None

            try:
                inputs = raw_data['inputs']
                output = raw_data['output']
                intermediates = raw_data['intermediates']
                program: Program = raw_data['program']

                function_seq = raw_data['function_sequence']
                unused_inputs = set(range(len(inputs)))
                unused_intermediates = set()
                encodings = []
                for depth, func in enumerate(function_seq, 1):
                    graph = RelationGraph(GraphOptions())
                    depth_inputs = [inputs[i] for i in unused_inputs]
                    depth_intermediates = [intermediates[i] for i in unused_intermediates]
                    graph_inputs = depth_inputs + depth_intermediates
                    graph.from_input_output(graph_inputs, output)

                    encoding = graph.get_encoding()
                    encoding['label'] = func
                    encodings.append(encoding)

                    unused_inputs -= program.call_seq[depth-1].get_used_inputs()
                    unused_intermediates -= program.call_seq[depth-1].get_used_intermediates()
                    unused_intermediates.add(depth-1)

                return encodings

            except SilentException:
                return None

            except Exception as e:
                try:
                    logger.warn("Failed for {}".format(raw_data))
                    logging.exception(e)
                    return None

                except:
                    pass

                return None

    def generate(self):
        self.init()
        num_generated = 0
        num_processed = 0
        num_raw_points = -1
        if os.path.exists(self.args.raw_data_path + '.index'):
            reader = IndexedFileReader(self.args.raw_data_path)
            num_raw_points = len(reader)
            reader.close()

        start_time = time.time()
        with pebble.ProcessPool(max_workers=self.args.processes, initializer=NextFunctionDataGenerator.Worker.init,
                                initargs=(self.args,)) as p:

            chunksize = self.args.processes * self.args.chunksize
            for chunk in misc.grouper(chunksize, self.raw_data_iterator()):
                future = p.map(NextFunctionDataGenerator.Worker.process, chunk, timeout=self.args.task_timeout)
                res_iter = future.result()

                idx = -1
                while True:
                    idx += 1
                    if idx < len(chunk) and chunk[idx] is not None:
                        num_processed += 1

                    try:
                        results = next(res_iter)
                        if chunk[idx] is None:
                            continue

                        if results is not None:
                            for result in results:
                                self.process_result(result)
                                num_generated += 1

                    except StopIteration:
                        break

                    except TimeoutError as error:
                        pass

                    except Exception as e:
                        try:
                            logger.warn("Failed for", chunk[idx])
                            logging.exception(e)

                        except:
                            pass

                    finally:

                        speed = round(num_processed / (time.time() - start_time), 1)
                        if num_raw_points != -1:
                            time_remaining = round((num_raw_points - num_processed) / speed, 1)
                        else:
                            time_remaining = '???'

                        logger.log("Generated/Processed : {}/{} ({}/s, TTC={}s)".format(num_generated, num_processed,
                                                                                        speed,
                                                                                        time_remaining), end='\r')

            p.stop()
            try:
                p.join(10)
            except:
                pass

        self.fwriter.close()

        logger.log("\n-------------------------------------------------")
        logger.info("Total Time : {:.2f}s".format(time.time() - start_time))
        logger.info("Generated {} training points from {} raw data points".format(num_generated, num_processed))
