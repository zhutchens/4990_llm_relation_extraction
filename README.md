# Environment Setup
Create a virtual environment using pip or conda:<br/>
Using Linux with pip:
```
python3 -m venv env
```
Using Linux with conda:
```
conda create -n env python=3.10.12 anaconda
```
Using Windows with pip:
```
python -m venv env
```
Using Windows with conda:
```
conda create -n env python=3.10.12 anaconda
```

# Activation and package installation
## Activation
### Pip
Using Linux, activate python virtual environment using:<br/>
```
source env/bin/activate
```
<br/>
Using Windows Powershell, activate python virtual environment using:
```
.\venv\Scripts\activate
```
<br/>
or, if you are using Windows with a Unix-like CLI:
```
source env/Scripts/activate
```
### Conda
On Windows and Linux, activate conda virtual environment using:
```
conda activate env
```
## Package Installation
Using pip, install packages using:
```
pip install -r requirements.
```
<br/>
Conda uses an environment.yaml file to specify dependencies instead of requirements.txt, so install pip first:
```
conda install pip
```
<br/>
Then, install packages:
```
pip install -r requirements.txt
```

# data
Concepts for data structures course. Sorting file is only concepts for searching and sorting chapters

# experiments
Experimentation on three different textbooks (cloud computing, graduate DS, undergraduate DS)
These are based on the old architecture and will not currently run

# src
Source code
Extractor file - LLM_Relation_Extractor class file of the algorithms.
utils file - utilities used in main class (splitting textbook, creating vector store, etc)

# testing and playground files
playground - trying out different code and functions and seeing what works best 
testing - evaluation testing of the entire architecture using undergrad DS textbook

# notes / observations
The testing file is the main focus right now. I have observed several patterns when playing around with the validate function of the LLM_Relation_Extractor class. The evaluate() function provided by ragas is a little weird, and produces ambigious errors sometimes. Additionally, if llm and embedding arguments are given the metrics, the results significantly vary. 

For example, without an llm argument to LLMContextPrecisionWithReference, it's very high (1.0). Additionally, without an llm passed to LLMContextRecall, the performance is very poor (~0.20 - 0.30). However, if you add in arguments to these they essentially reverse. Precision becomes very poor (~0.0) and recall becomes very high (~1.0). 

Similarly, the same thing occurs with the FactualCorrectness metric. However, it tends to stay poor regardless of an argument. Without an llm it tends to be (0.0 - 0.06) and with an llm it becomes (0.2 - 0.3)

The only evaluations that are consistent are response relevancy (0.83 - 0.86), semantic similarity (0.80 - 0.83), and faithfulness (0.90 - 0.1).

Finally, evaluate() will sometimes produce these statements saying 'No statements were generated in the answer' and skips ahead. I'm not really sure what its doing, so I'm going to raise an issue on the github page and ask. My current theory of this is that if an llm is passed as an arg, it might say this and cause the metric to be unusually high.


TLDR: the evaluation results REALLY depend on the args passed to the metrics
