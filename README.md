# Usage of LLM's to extract associations and dependencies from textbooks

## Relation Algorithms
Contains a class that can be used to instantiate a LLM_Relation_Extractor. This object can identify main topics, concepts, outcomes, associations, and dependencies given a textbook link. Code is also included to print the dependencies and associations graphs. The flat graph uses Graphviz to visualize the relationships, but the interactive function uses pyvis. It's important to note that the interactive graph can only be shown in notebooks that can load html, such as Jupyter Notebook/Lab. Otherwise, you may have to load the html file in a browser.

## Assocations_2214 files
Testing with langchain LLM to extract key concepts, outcomes, topics, and their associations and dependencies on each other. This file does not use the object in the relation algorithms folder, but was the original basis for that object. This file is based on the ITCS 2214: Data Structures and Algorithms course textbook. 

## Associations_hadoop
Testing the class from the relation_extraction_functions file. I believe that some of the functions work rather well, such as the functions that identify the main topics, key learning concepts, key learning outcomes, and retrieving/printing the graph visualizations. However, some the functions do not work well, such as the functions that identify the chapters, identify the associations between chapters, and the function that identifies the dependencies between chapters. This file is based on the ITCS 3190: Cloud Computing for Data Analysis textbook. 