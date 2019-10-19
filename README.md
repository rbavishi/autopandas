## AutoPandas (OOPSLA 2019 Version)

This branch contains code and commands used for the evaluation in the OOPSLA 2019 paper titled "AutoPandas : Neural-Backed Generators for Program Synthesis". The paper can be accessed [here](https://dl.acm.org/citation.cfm?doid=3366395.3360594) (open-access).

Although useful for reproducing results, we strongly recommend to switch to Atlas on the `master` branch for building on top of this work or extending to other domains. The porting is on-going and should be done soon. This README will then be updated accordingly.

## Installation and Setup

* The system has been tested against Python 3.6.5 and Tensorflow 1.9.0

* Do a development install

  `pip install -e .`
  
* Install tensorflow (Make sure you have Cuda 9)

  `pip install tensorflow-gpu==1.9.0`
  
* Compile generators
  
  `autopandas_v2 generators compile`
  
  `autopandas_v2 generators compile-randomized`
