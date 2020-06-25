## AutoPandas (OOPSLA 2019 Version)

This branch contains code and documentation for the system used in the OOPSLA 2019 paper titled 
"AutoPandas : Neural-Backed Generators for Program Synthesis". 
The paper can be accessed [here](https://dl.acm.org/citation.cfm?doid=3366395.3360594) (open-access).

## Installation and Setup

* The system has been tested against Python 3.6.5 and `tensorflow-gpu==1.9.0`

* Do a development install

  `pip install -e .`
  
* Compile generators
  
  `autopandas_v2 generators compile`
  
  `autopandas_v2 generators compile-randomized`

## Reproducing Evaluation

* Switch to the snapshot at [https://github.com/rbavishi/atlas/tree/oopsla19-snapshot](https://github.com/rbavishi/atlas/tree/oopsla19-snapshot)

* Download pre-trained models [here](https://drive.google.com/drive/folders/1XRZDAP0HSegI97jzNJSFWQUl4HOf4U7x?usp=sharing) and extract the zip files. 
There should be two - (1) `model_pandas_generators` and (2) `model_pandas_functions`. 
Note that a GPU is *necessary* to use these models. 
We have observed NaNs being returned by our models when performing inference on the CPU.

* Run the following to reproduce results in Table 2 in the [paper](https://dl.acm.org/citation.cfm?doid=3366395.3360594). 
Note that execution times may differ across runs. We have observed non-trivial deviations over different hardware due to 
different predictions by the models which have sort of a cascading effect. However, the benchmarks solved within the time-limit and the number of programs explored should be similar.

  `TF_CPP_MIN_LOG_LEVEL=3 autopandas_v2 evaluate synthesis "PandasBenchmarks.*" model_pandas_generators model_pandas_functions pandas_synthesis_results.csv --top-k-args 1000 --use-old-featurization --timeout 1200`
  
* Note that the `--use-old-featurization` is important only for the snapshot. If the models have been retrained, 
you should skip this option (after switching to the latest commit, of course).
  
## Creating Data and Training Generators

### Creating Raw Data

Raw data consists of inputs, programs and their outputs along with choices made by the generators. 
Basic usage is as follows. The `viable_sequences.pkl` file should contain a set of tuples representing valid combinations 
of functions that raw data should be generated for.

* Generate `1 million` data-points with sequences from `viable_sequences.pkl` using `32` processes with minimum and maximum length of sequences allowed as `1` and `3`. Save the data to `raw_data.pkl`

```autopandas_v2 generators training-data raw raw_data.pkl --sequences viable_sequences.pkl --processes 32 --min-depth 1 --max-depth 3 --num-training-points 1000000```

* Generate another `1 million` points with the same constraints but append it to the existing data in `raw_data.pkl`

```autopandas_v2 generators training-data raw raw_data.pkl --append --sequences viable_sequences.pkl --processes 32 --min-depth 1 --max-depth 3 --num-training-points 1000000```

Full Usage -
```
usage: autopandas_v2 generators training-data raw [-h] [--debug]
                                                  [--processes PROCESSES]
                                                  [--chunksize CHUNKSIZE]
                                                  [--task-timeout TASK_TIMEOUT]
                                                  [--max-exploration MAX_EXPLORATION]
                                                  [--max-arg-trials MAX_ARG_TRIALS]
                                                  [--max-seq-trials MAX_SEQ_TRIALS]
                                                  [--blacklist-threshold BLACKLIST_THRESHOLD]
                                                  [--min-depth MIN_DEPTH]
                                                  [--max-depth MAX_DEPTH]
                                                  [--num-training-points NUM_TRAINING_POINTS]
                                                  --sequences SEQUENCES
                                                  [--no-repeat] [--append]
                                                  outfile

positional arguments:
  outfile               Path to output file

optional arguments:
  -h, --help            show this help message and exit
  --debug               Debug-level logging
  --processes PROCESSES
                        Number of processes to use
  --chunksize CHUNKSIZE
                        Pebble Chunk Size. Only touch this if you understand
                        the source
  --task-timeout TASK_TIMEOUT
                        Timeout for a datapoint generation task (for
                        multiprocessing). Useful for avoiding enumeration-
                        gone-wrong cases, where something is taking a long
                        time or is consuming too many resources
  --max-exploration MAX_EXPLORATION
                        Maximum number of arg combinations to explore before
                        moving on
  --max-arg-trials MAX_ARG_TRIALS
                        Maximum number of argument trials to actually execute
  --max-seq-trials MAX_SEQ_TRIALS
                        Maximum number of trials to generate data for a single
                        sequence
  --blacklist-threshold BLACKLIST_THRESHOLD
                        Maximum number of trials for a sequence before giving
                        up forever. Use -1 to have no threshold
  --min-depth MIN_DEPTH
                        Minimum length of sequences allowed
  --max-depth MAX_DEPTH
                        Maximum length of sequences allowed
  --num-training-points NUM_TRAINING_POINTS
                        Number of training examples to generate
  --sequences SEQUENCES
                        Path to pickle file containing sequences that the
                        generator can stick to while generating data. Helps in
                        generating random data that mimics actual usage of the
                        API in the wild. Can also be a comma plus colon-
                        separated string containing functions to use. For
                        example - df.pivot:df.index,df.columns:df.T allows the
                        sequences (df.pivot, df.index) and (df.columns, df.T)
  --no-repeat           Produce only 1 training example for each sequence
  --append              Whether to append to an already existing dataset

```

For retraining the models from scratch, it is advisable to generate raw data for different lengths independently, and also
split them into training and validation sets. The following commands create training and validation sets of sizes 
`1000000` and `10000` respectively for each sequence length ranging from `1` to `3` separately. 
Note that this process can take a long time and may need babysitting. For example, the process may not 
exit even if the required number of data-points have been generated due to threads not terminating. In this case, manual 
intervention using `SIGINT` (`Ctrl-C`) is required.

The `pandas_mined_seqs.pkl` can be obtained [here](https://drive.google.com/file/d/13r5uaZddvmtDtL29WaHtQtrA2R8qioBS/view?usp=sharing).

```bash
autopandas_v2 generators training-data raw training_raw_data_depth1.pkl --sequences pandas_mined_seqs.pkl --processes 32 --min-depth 1 --max-depth 1 --num-training-points 1000000
autopandas_v2 generators training-data raw training_raw_data.pkl --append --sequences pandas_mined_seqs.pkl --processes 32 --min-depth 2 --max-depth 2 --num-training-points 1000000
autopandas_v2 generators training-data raw training_raw_data.pkl --append --sequences pandas_mined_seqs.pkl --processes 32 --min-depth 3 --max-depth 3 --num-training-points 1000000

autopandas_v2 generators training-data raw validation_raw_data.pkl --sequences pandas_mined_seqs.pkl --processes 32 --min-depth 1 --max-depth 1 --num-training-points 10000
autopandas_v2 generators training-data raw validation_raw_data.pkl --append --sequences pandas_mined_seqs.pkl --processes 32 --min-depth 2 --max-depth 2 --num-training-points 10000
autopandas_v2 generators training-data raw validation_raw_data.pkl --append --sequences pandas_mined_seqs.pkl --processes 32 --min-depth 3 --max-depth 3 --num-training-points 10000
```

The raw data used in the artifact can be found [here](https://drive.google.com/drive/folders/149MD9NDzEjAOiVy9KU4ju6x8BxEOvO3W?usp=sharing).

### Creating Structured Data

The raw data has to be converted into graphs for training. The basic command structure is as follows.
Currently, the implementation differentiates between the models for operators inside generators and the model used for 
predicting function sequences to explore. Hence we generate two sets of structured data.

#### Structured Data Generation (For Generators)

* Generate structured data from raw data contained in `raw_data.pkl` and store the data in a directory named `struct_data_generators` using `32` processes

```autopandas_v2 generators training-data generators raw_data.pkl struct_data_generators --processes 32```

* Generate structured data from raw data contained in `other_data.pkl` but append it to the already existing data in `struct_data_generators`, again using `32` processes

```autopandas_v2 generators training-data generators other_data.pkl struct_data_generators --append-arg-level```

*Tip* - Appending is useful for combining data of different depths, since it is a prerequisite for training. **However this does not mean that you should not parallelize**. The final combination of data can happen during training.

Full Usage - 
```
usage: autopandas_v2 generators training-data generators [-h] [--debug] [-f]
                                                         [--append-arg-level]
                                                         [--processes PROCESSES]
                                                         [--chunksize CHUNKSIZE]
                                                         [--task-timeout TASK_TIMEOUT]
                                                         raw_data_path outdir

positional arguments:
  raw_data_path         Path to pkl containing the raw I/O example data
  outdir                Path to output directory where the generated data is
                        to be stored

optional arguments:
  -h, --help            show this help message and exit
  --debug               Debug-level logging
  -f, --force           Force recreation of outdir if it exists
  --append-arg-level    Append training-data at argument-operator level
                        instead of overwriting by default
  --processes PROCESSES
                        Number of processes to use
  --chunksize CHUNKSIZE
                        Pebble Chunk Size. Only touch this if you understand
                        the source
  --task-timeout TASK_TIMEOUT
                        Timeout for a datapoint generation task (for
                        multiprocessing). Useful for avoiding enumeration-
                        gone-wrong cases, where something is taking a long
                        time or is consuming too many resources

```

#### Structured Data Generation (For Function Sequences)

* Generate structured data from raw data contained in `raw_data.pkl` and store the data in a file named `struct_data_functions.pkl` using `32` processes

```autopandas_v2 generators training-data function-seq raw_data.pkl struct_data_functions.pkl --processes 32```

* Generate structured data from raw data contained in `other_data.pkl` but append it to the already existing data in `struct_data_functions.pkl`, again using `32` processes

```autopandas_v2 generators training-data function-seq other_data.pkl struct_data_functions.pkl --append```

*Tip* - Appending is useful for combining data of different depths, since it is a prerequisite for training

Full Usage -
```
usage: autopandas_v2 generators training-data function-seq [-h] [--debug]
                                                           [--append]
                                                           [--processes PROCESSES]
                                                           [--chunksize CHUNKSIZE]
                                                           [--task-timeout TASK_TIMEOUT]
                                                           raw_data_path
                                                           outfile

positional arguments:
  raw_data_path         Path to pkl containing the raw I/O example data
  outfile               Path to output file where the generated data is to be
                        stored

optional arguments:
  -h, --help            show this help message and exit
  --debug               Debug-level logging
  --append              Append training-data to the existing dataset
                        represented by outfileinstead of overwriting by
                        default
  --processes PROCESSES
                        Number of processes to use
  --chunksize CHUNKSIZE
                        Pebble Chunk Size. Only touch this if you understand
                        the source
  --task-timeout TASK_TIMEOUT
                        Timeout for a datapoint generation task (for
                        multiprocessing). Useful for avoiding enumeration-
                        gone-wrong cases, where something is taking a long
                        time or is consuming too many resources
```

If using the script in the previous section to generate training and validation data for each depth separately, you 
can use the following set of commands to convert that raw data into structured data.

```bash
autopandas_v2 generators training-data generators training_raw_data.pkl training_struct_data_generators --processes 32
autopandas_v2 generators training-data generators validation_raw_data.pkl validation_struct_data_generators --processes 32
autopandas_v2 generators training-data function-seq training_raw_data.pkl training_struct_data_functions.pkl --processes 32
autopandas_v2 generators training-data function-seq validation_raw_data.pkl validation_struct_data_functions.pkl --processes 32
```

### Training
Now we are ready to train the models.

#### Training function sequence prediction model

Basic usage is as follows. The command trains a model for 100 epochs, and early-stopping after 25 epochs with no improvement in accuracy.

```autopandas_v2 generators training train-functions model_functions --train training_struct.pkl --valid validation_struct.pkl --use-disk --config-str '{"batch_size": 50000}' --patience 25 --num-epochs 100```


Full usage is the following.
```
usage: autopandas_v2 generators training train-functions [-h]
                                                         [--device DEVICE]
                                                         [--config CONFIG]
                                                         [--config-str CONFIG_STR]
                                                         [--use-memory]
                                                         [--use-disk] --train
                                                         TRAIN --valid VALID
                                                         [--restore-file RESTORE_FILE]
                                                         [--restore-params RESTORE_PARAMS]
                                                         [--freeze-graph-model]
                                                         [--load-shuffle]
                                                         [--num-epochs NUM_EPOCHS]
                                                         [--patience PATIENCE]
                                                         [--num-training-points NUM_TRAINING_POINTS]
                                                         modeldir

positional arguments:
  modeldir              Path to the directory to save the model in

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE       ID of Device (GPU) to use
  --config CONFIG       File containing hyper-parameter configuration (JSON
                        format)
  --config-str CONFIG_STR
                        String containing hyper-parameter configuration (JSON
                        format)
  --use-memory          Store all processed graphs in memory. Fastest
                        processing, but can easilyrun out of memory
  --use-disk            Use disk for storing processed graphs as opposed to
                        computing them every timeSpeeds things up a lot but
                        can take a lot of space
  --train TRAIN         Path to train file
  --valid VALID         Path to validation file
  --restore-file RESTORE_FILE
                        File to restore weights from
  --restore-params RESTORE_PARAMS
                        File to restore params from (pkl)
  --freeze-graph-model  Freeze graph model components
  --load-shuffle        Shuffle data when loading. Useful when passing num-
                        training-points
  --num-epochs NUM_EPOCHS
                        Maximum number of epochs to run training for
  --patience PATIENCE   Maximum number of epochs to wait for validation
                        accuracy to increase
  --num-training-points NUM_TRAINING_POINTS
                        Number of training points to use. Default : -1 (all)
```

#### Training generator operator models

Basic usage is as follows. The command trains a model for 100 epochs, and early-stopping after 25 epochs with no improvement in accuracy.

```autopandas_v2 generators training train-generators --train training_struct_data --valid validation_struct_data --use-disk --num-epochs 100 --patience 25 --config-str '{"layer_timesteps": [1,1,1], "batch_size": 50000}' --ignore-if-exists model_generators```

Full usage is as follows.
```
usage: autopandas_v2 generators training train-generators [-h]
                                                          [--device DEVICE]
                                                          [--config CONFIG]
                                                          [--config-str CONFIG_STR]
                                                          [--use-memory]
                                                          [--use-disk] --train
                                                          TRAIN --valid VALID
                                                          [--restore-file RESTORE_FILE]
                                                          [--restore-params RESTORE_PARAMS]
                                                          [--freeze-graph-model]
                                                          [--load-shuffle]
                                                          [--num-epochs NUM_EPOCHS]
                                                          [--patience PATIENCE]
                                                          [--num-training-points NUM_TRAINING_POINTS]
                                                          [--include INCLUDE [INCLUDE ...]]
                                                          [--restore-if-exists]
                                                          [--ignore-if-exists]
                                                          modeldir

positional arguments:
  modeldir              Path to the directory to save the model(s) in

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE       ID of Device (GPU) to use
  --config CONFIG       File containing hyper-parameter configuration (JSON
                        format)
  --config-str CONFIG_STR
                        String containing hyper-parameter configuration (JSON
                        format)
  --use-memory          Store all processed graphs in memory. Fastest
                        processing, but can easilyrun out of memory
  --use-disk            Use disk for storing processed graphs as opposed to
                        computing them every timeSpeeds things up a lot but
                        can take a lot of space
  --train TRAIN         Path to train file
  --valid VALID         Path to validation file
  --restore-file RESTORE_FILE
                        File to restore weights from
  --restore-params RESTORE_PARAMS
                        File to restore params from (pkl)
  --freeze-graph-model  Freeze graph model components
  --load-shuffle        Shuffle data when loading. Useful when passing num-
                        training-points
  --num-epochs NUM_EPOCHS
                        Maximum number of epochs to run training for
  --patience PATIENCE   Maximum number of epochs to wait for validation
                        accuracy to increase
  --num-training-points NUM_TRAINING_POINTS
                        Number of training points to use. Default : -1 (all)
  --include INCLUDE [INCLUDE ...]
                        fn:identifier tuples to include in training list
  --restore-if-exists   If a model already exists, pick up training from there
  --ignore-if-exists    If the model exists, skip.
```

If using the scripts from previous section to generate data for each depth separately, the following set of commands 
can be used to train the models. Again, babysitting may be required to restart the training in case of crashes.

```
autopandas_v2 generators training train-functions model_functions --train training_struct_data_functions.pkl --valid validation_struct_data_functions.pkl --use-disk --config-str '{"batch_size": 50000}' --patience 25 --num-epochs 100
autopandas_v2 generators training train-generators --train training_struct_data_generators --valid validation_struct_data_generators --use-disk --num-epochs 100 --patience 25 --config-str '{"layer_timesteps": [1,1,1], "batch_size": 50000}' --ignore-if-exists model_generators
```

## Contact

For questions, contact `rbavishi@cs.berkeley.edu`.

