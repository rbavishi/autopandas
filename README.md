## AutoPandas (OOPSLA 2019 Version)

This branch contains code and commands used for the evaluation in the OOPSLA 2019 paper titled "AutoPandas : Neural-Backed Generators for Program Synthesis". The paper can be accessed [here](https://dl.acm.org/citation.cfm?doid=3366395.3360594) (open-access).

Although useful for reproducing results, we strongly recommend to switch to Atlas on the `master` branch for building on top of this work or extending to other domains. The code in this repo is not well-structured and is inextensible. We have also made a number of improvements since publication that will be integrated into the Atlas version. The complete porting of AutoPandas to Atlas is ongoing and should be done soon. This README will then be updated accordingly.

## Installation and Setup

* The system has been tested against Python 3.6.5 and Tensorflow 1.9.0

* Do a development install

  `pip install -e .`
  
* Install tensorflow (Make sure you have Cuda 9)

  `pip install tensorflow-gpu==1.9.0`
  
* Compile generators
  
  `autopandas_v2 generators compile`
  
  `autopandas_v2 generators compile-randomized`

## Running Evaluation (Synthesis-only)

* Download pre-trained models [here](https://drive.google.com/drive/folders/1XRZDAP0HSegI97jzNJSFWQUl4HOf4U7x?usp=sharing) and extract the zip files. There should be two - (1) `model_pandas_generators` and (2) `model_pandas_functions`

* Run the following to reproduce results in Table 2 in the [paper](https://dl.acm.org/citation.cfm?doid=3366395.3360594). Note that execution times may differ across runs. We believe this is mostly due to the use of set comprehensions (list comprehension for sets) inside the generators which have a random iteration order of the elements. However, the benchmarks solved within the time-limit will be the same.

  `TF_CPP_MIN_LOG_LEVEL=3 autopandas_v2 evaluate synthesis "PandasBenchmarks.*" model_pandas_generators model_pandas_functions pandas_synthesis_results.csv --top-k-args 1000 --use-old-featurization --timeout 1200`
  
## Creating Data and Training Generators

* TODO
