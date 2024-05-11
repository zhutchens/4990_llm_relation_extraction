# Using Large Language Models to generate knowledge graphs

## Relation Algorithms
Contains a class that can be used to instantiate a LLM_Relation_Extractor. This object can identify main topics, concepts, outcomes, associations, and dependencies given a textbook link. Code is also included to print the dependencies and associations graphs. The flat graph uses Graphviz to visualize the relationships, but the interactive function uses pyvis. It's important to note that the interactive graph can only be shown in notebooks that can load html, such as Jupyter Notebook/Lab. Otherwise, you may have to load the html file in a browser.

## openAI learning concepts associations - 2214
Original file for experimentation with creating knowledge graphs, and what the class in the relation algorithms file is based on

## grad_dsa and hadoop
These files utilize the class in the relation algorithms folder to generate the knowledge graph

## S2_2214 Ontology Knowledge terms
The true concepts for the 2214 course. These are going to be used to evaluate the models performance at identifying learning concepts using precision and recall scores 