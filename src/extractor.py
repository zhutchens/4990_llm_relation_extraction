from langchain_openai.chat_models import ChatOpenAI
import graphviz
import os
import re
from IPython.display import display, Image
from pyvis.network import Network
import hypernetx as hnx
import matplotlib.pyplot as plt
import graphlib
from py3plex.core import multinet
import pymupdf
from urllib.request import urlopen
from io import BytesIO
from ragas import evaluate, SingleTurnSample
from ragas.dataset_schema import EvaluationDataset
from src.utils import create_concept_graph_structure, split, get_node_id_dict, create_retriever, invoke_retriever



# NOTE: Currently one main issue, the function that finds the associations between chapters seems to be broken. I think its the algorithm thats wrong. It also takes 10+ minutes to run
class LLM_Relation_Extractor:
    def __init__(self, link: str, token: str, chapters: list[str], stopword: str):
        # NOTE: Would it also be helpful to add a chapter dictionary as a parameter?
        '''
        Constructor to create a large language model relation extractor class. 

        Args:
            link (str): the web link to generate answers from
            token (str): OpenAI token
            chapters (list): list of chapter names in link
            stopword (str): word where last chapter ends (usually appendix or bibliography)
        '''
        self.link = link
        self.chapters = chapters
        self.stopword = stopword

        os.environ['OPENAI_API_KEY'] = token
        self.llm = ChatOpenAI(temperature = 0)

        create_retriever(self.link)


    def identify_key_terms(self, chapter_name: str, n_terms: int) -> list[str]:
        '''
        Identify the key terms for each chapter
        
        Args:
            chapter_name: name of chapter to get key terms for 
            n_terms: number of key terms to use

        Returns:
            list[str]: key terms of chapter
        '''
        if type(n_terms) is not int:
            raise ValueError(f"n_terms should be of type int, got type {type(n_terms)}")

        if chapter_name is None:
            raise ValueError('chapter_name cannot be None')
        
        terms = self.llm.invoke(f'Identify {n_terms} key terms for chapter {chapter_name}. The textbook can be found here: {self.link}. Finally, with each term provide an explanation as to why that term should be included.').content
    
        # return terms.split('\n')[2:]
        return [string for string in terms.split('\n') if string != '']


    def summarize(self) -> str:
        '''
        Returns summary of class link

        Args:
            None

        Returns:
            str: summary of textbook
        '''
        return self.llm.invoke(f"Please summarize this textbook {self.link}")


    def create_chapter_dict(self, outcomes: list[str], concepts: list[str]) -> dict[str, tuple[str, str]]:
        '''
        Create a chapter dictionary containing the chapter names as keys and its concepts and outcomes in a tuple as the value

        Args:
            outcomes (list[str]): a list of learning outcomes created from the identify_learning_outcomes function
            concepts (list[str]): a list of learning concepts created from the identify_learning_concepts function

        Returns:
            dict[str, tuple[str, str]]: dictionary containing chapter names as keys and concepts/outcomes as tuple
        '''
        outcome_concept_graph = {}

        for idx, name in enumerate(self.chapters):
            outcome_concept_graph[name] = (concepts[idx], outcomes[idx])
                
        return outcome_concept_graph
        
    
    def identify_chapters(self) -> dict[str, str]:
        '''
        Identify the chapters within the class provided link using a large language model
        It is very, very inconsistent and I highly recommened manually creating the dictionary

        Args:
            None

        Returns:
            dict[str, str]: key is chapter number, value is chapter name
        '''
        # NOTE: This is very, very, inconsistent. Do not recommend using this.
        chapters = self.llm.invoke(f"Please identify the chapters in this textbook: {self.link}")
        chapters = chapters.split('\n')
        chapter_dict = {}
        
        for idx, chapter in enumerate(chapters):
            chapter_dict[f"Chapter {idx + 1}"] = chapter

        return chapter_dict


    def identify_main_topics(self) -> list[str]:
        '''
        Identify the main topics within the class provided link

        Args:
            None
        
        Returns:
            list[str]: list of main topics 
        '''
        main_topics = self.llm.invoke(f"Please identify the main topics from this textbook: {self.link}")
        return [topic for topic in main_topics.split('\n')][2:]
    
    
    def identify_main_topic_relations(self, main_topic_list: list[str]) -> dict[str, list[str]]:
        '''
        Identify the relationships between the main topics of the textbook

        Args:
            main_topics_list (list[str]): A list of main topics from the textbook. Can be automatically created using the identify_main_topics() function
        
        Returns:
            dict[str, list[str]]: relationships between main topics as adjacency list
        '''
        topic_relations = create_concept_graph_structure(main_topic_list)

        relation = ''
        for i in range(len(main_topic_list)):
            for j in range(len(main_topic_list)):
                if i != j:
                    relation = self.llm.invoke(f"Is there a relationship between this topic: {main_topic_list[i]}, and this topic: {main_topic_list[j]}? If there is NOT, please respond with 'No' and 'No' only.")
                    relation = re.sub(re.compile('^[a-zA-Z\s\.,!?]'), '', relation)
                    if relation.split(',')[0].strip() != 'No':
                        topic_relations[main_topic_list[i]].append(main_topic_list[j])

        return topic_relations

    
    def identify_outcomes(self) -> list[list[str]]:
        '''
        Identify the main learning outcomes within the class provided link

        Args:
            None
        
        Returns:
            list[list[str]]: list of outcomes for each chapter
        '''
        outcome_list = []
        current_outcome = ''
    
        for name in self.chapters:
            # current_outcome = self.llm.invoke(f"Please identify the main learning outcomes given this chapter: {name}. Here is the textbook in which to retrieve them: {self.link}")
            current_outcome = self.llm.invoke(f'{self.outcome_prompt}, chapter name: {name}')
            current_outcome = re.sub(re.compile('^[a-zA-Z\s\.,!?]'), '', current_outcome)
            outcome_list.append(current_outcome.split('\n'))
        
        return outcome_list
    

    def identify_concepts(self) -> list[list[str]]:
        '''
        Identify the main learning concepts within the class provided link

        Args:
            None
        
        Returns:
            list: list of concepts for each chapter
        '''

        concept_list = []
        current_concept = ''
        global retrieved_contexts
        retrieved_contexts = {}

        # print(key_terms)

        for name in self.chapters:
            # print('Chapter name:', name)

            key_terms = self.identify_key_terms(chapter_name = name, n_terms = 10) # 10 for now
            key_terms.append(name)
            # print('Key terms:', key_terms)

            relevant_docs = invoke_retriever(' '.join(key_terms))
            # print(relevant_docs)
            # print(''.join([text.page_content for text in relevant_docs]))
            retrieved_contexts[name] = ''.join([text.page_content for text in relevant_docs])
            # print(''.join([text.page_content for text in relevant_docs]))
            # print(retrieved_contexts[name])
            # print(f'Chapter: {name}, retrieved: {retrieved_contexts[name]}')
            # break

            current_concept = self.llm.invoke(f'Identify the ten most important learning concepts for chapter: {name}. The relevant documents can be found here: {relevant_docs}').content
            # current_concept = self.llm.invoke(f'Identify the ten most important learning concepts from these documents: {relevant_docs}. Do not provide explanations, only list them.')

            current_concept = re.sub(re.compile('^[a-zA-Z\s\.,!?]'), '', current_concept)
            concept_list.append([concept for concept in current_concept.split('\n') if concept != ''])

        return concept_list
        

    def identify_associations(self, learning_dict: dict[str, tuple[str, str]]) -> dict[str, list[str]]:
        '''
        Identify associations between chapters. For example, if there is an association between Chapter 1 and 3 it will be added to the dictionary. The return dict contains chapter names as keys and the chapter names its associated with as values

        Args:
            learning_dict (dict[str, tuple[str, str]]): The dictionary returned from the create_chapter_dict() function

        Returns:
            dict[str, list[str]]: associations between chapters as adjacency list
        '''
        association_dict = create_concept_graph_structure(param_list = list(learning_dict.keys()))
        new_association = ''
        values = list(learning_dict.values())
        keys = list(learning_dict.keys())

        # NOTE: I think this algorithm is wrong? Maybe its better to loop through in reverse order (Ex. start with the last chapter and work downwards)
        for i in range(len(values)):
            current_tuple = values[i]
            for j in range(len(values)):
                if i != j:
                    next_tuple = values[j]
                    new_association = self.llm(f"Please identify if there is an association between this concept: {current_tuple[0]}, and this other concept: {next_tuple[0]}. If there is NO association, please start your response with 'No' and 'No' only.")
                    new_association = re.sub(re.compile('[^a-zA-Z\s\.,!?]'), '', new_association)
                    # Try to only add associations to the graph, but its difficult because sometimes the llm won't start its response with 'No'
                    if new_association.split(',')[0].strip() != 'No':
                        association_dict[keys[i]].append(keys[j])
            
        return association_dict


    def identify_dependencies(self, concept_dict: dict[str, list[str]]) -> dict[str, list[str]]:
        '''
        Identify the dependency relationships between chapters, returns a dictionary where the key is a chapter name and the value is a list of chapters it depends on

        Args: 
            concept_dict (dict[str, list[str]]): a dictionary containing the chapter names as keys and their learning concepts as values. Dictionary should come from the create_chapter_dict() function

        Returns:
            dict[str, list[str]]: depedencies between chapters as adjacency list
        '''

        relation = ''

        keys = list(concept_dict.keys())
        relations_dict = create_concept_graph_structure(keys)

        for i in range(len(keys)):
            current_concept = concept_dict[keys[i]][0]
            for j in range(i + 1, len(keys)):
                next_concept = concept_dict[keys[j]][0]
                relation = self.llm(f"Please identify if these concepts: {next_concept} are a prerequisite for these concepts: {current_concept}. If there is NO prerequisite, please respond with 'No' and 'No' only.")
                relation = re.sub(re.compile('[^a-zA-Z\s\.,!?]'), '', relation)
                if relation.split(',')[0].strip() != 'No':
                    relations_dict[keys[j]].append(keys[i])

        return relations_dict


    def print_flat_graph(self, concept_graph: dict[str, list[str]]) -> None:
        '''
        Print a directed graph using either the association dictionary or dependency dictionary

        Args:
            learning_concept_graph: The dictionary to build the graph from. This should come from either the identify_associations function or identify_dependencies function
        
        Returns:
            None
        '''
        graph = graphviz.Digraph()

        for key, values in concept_graph.items():
            graph.node(name = key)
            for value in values:
                graph.edge(key, value)

        display(Image(graph.pipe(format = "png", renderer = "cairo")))


    # NOTE: I think I might be able to combine these two functions into one
    def get_assocation_interactive_graph(self, graph: dict[str, tuple[str, str]], associations: dict[str, list[str]]) -> Network:
        '''
        Retrieve the interactive graph using the association dictionary. Nodes are chapter names and edges are the associations. Hovering over a node results in displaying that nodes learning outcomes and concepts. The function is not able to automatically display the graph so the .show() method must be called on the return object

        Args:
            graph: The dictionary containing chapter names as keys and the values as a tuple containing the concept at index 0 and outcome at index 1. Can be created automatically using the create_chapter_dict function
            assocations: The dictionary containing the associations between chapters. Can be created automatically using the identify_associations() function
        
        Returns:
            A pyvis Network object
        '''

        graph = Network(notebook = True, cdn_resources = "remote")

        graph.toggle_physics(False)

        # Showing all interactivity options, but can be parameterized to only include some
        graph.show_buttons()
        
        node_id_dict = get_node_id_dict(graph)

        for chapter_name, chapter_id in node_id_dict.items():
            graph.add_node(n_id = chapter_id, label = chapter_name, title = "Main Learning Concepts: " + graph[chapter_name][0] + "\n" + "Main Learning Outcomes:" + graph[chapter_name][1])

        for key, values in associations.items():
            for value in values:
                graph.add_edge(node_id_dict[key], node_id_dict[value])

        return graph


    def get_dependency_interactive_graph(self, dependency_dict: dict[str, list[str]]) -> Network:
        '''
        Retrieve the interactive graph using the dependency dictionary. The function is not able to automatically display the graph so the .show() method must be called on the return object

        Args:
            dependency_dict (dict[str, list[str]]): A dictionary containing the chapter_names between chapters. Can be created automatically using the dependency_relation_extraction() function
        
        Returns:
            A pyvis Network object
        '''

        dependency_graph = Network(notebook = True, cdn_resources = "remote")
        dependency_graph.toggle_physics(False)
        dependency_graph.show_buttons()

        node_id_dict = get_node_id_dict(dependency_dict)

        for chapter_name, chapter_id in node_id_dict.items():
            dependency_graph.add_node(n_id = chapter_id, label = chapter_name)


        for key, values in dependency_dict.items():
            for value in values:
                dependency_graph.add_edge(node_id_dict[key], node_id_dict[value])

        return dependency_graph
    

    def draw_hypergraph(self, dictionary: dict[str, list[str]]) -> None:
        '''
        Generate and display a hypergraph given a dependency dictionary generated from identify_dependencies() or association dict generated by identify_associations()

        Args:
            dependencies (dict[str, list[str]]): A dictionary of dependencies. Can be generated by identify_dependencies(). The key should be a chapter name and the value a list of chapters it depends on

        Returns:
            None
        '''
                
        sorted_dependencies = graphlib.TopologicalSorter(graph = dictionary)
        sorted_dependencies = tuple(sorted_dependencies.static_order())

        temp = sorted_dependencies
        sorted_dependencies = {}

        for value in temp:
            sorted_dependencies[value] = dictionary[value]

        hypergraph = hnx.Hypergraph(sorted_dependencies)
        hnx.draw(hypergraph)
        plt.title("Hypergraph")
        plt.show()


    def draw_multi_layered_graph(self, dictionary: dict[str, list[str]]) -> None:
        '''
        Generate and display a multilayered graph given a dependency dictionary generated from identify_dependencies() or association dict generated by identify_associations()

        Args:
            dependencies (dict[str, list[str]]): A dictionary of dependencies. Can be generated by identify_dependencies(). The key should be a chapter name and the value a list of chapters it depends on

        Returns:
            None
        '''

        sorted_dependencies = graphlib.TopologicalSorter(graph = dictionary)
        sorted_dependencies = tuple(sorted_dependencies.static_order())

        temp = sorted_dependencies
        sorted_dependencies = {}

        for value in temp:
            sorted_dependencies[value] = dictionary[value]
        

        multi_graph = multinet.multi_layer_network(network_type = "multiplex")

        for node, edges in sorted_dependencies.items():
            node_data = {"source": node, "type": node}
            multi_graph.add_nodes(node_data)
            for edge in edges:
                simple_edge = {
                        "source": node,
                        "target": edge,
                        "source_type": node,
                        "target_type": edge
                        }
                
                multi_graph.add_edges(simple_edge, input_type = "dict") 

        multi_graph.visualize_network(style = "diagonal")
        plt.title("Multilayered Dependency Graph")
        plt.show()


    def validate(self, concepts: dict[str, list[str]], ground_truth: list[str], metrics: list, multi_turn: bool = False) -> list[SingleTurnSample]:
        '''
        Validate concepts from LLM  

        Args:
            prompt (str): the prompt the LLM was given
            concepts (dict): LLM generated concepts from object link
            ground_truth (list): ground truth concepts 
            stopword (str): first word where the textbook chapters end (usually appendix or bibliography)
            metrics (list, default None): list of metrics to use from ragas library
            multi_turn (bool, default False): if True, use MultiTurnSample. Otherwise uses SingleTurnSample

        Returns:
            list[SingleTurnSample]: list of samples used for evaluation  
        '''      
        samples = []

        textbook = split(self.link, self.chapters, self.stopword) # for reference contexts
        for i in range(len(ground_truth)):
            samples.append(SingleTurnSample(
                user_input = f'Identify the ten most important learning concepts for chapter: {self.chapters[i]}. The context can be found here: {retrieved_contexts[self.chapters[i]]}',
                response = concepts[self.chapters[i]],
                retrieved_contexts = [retrieved_contexts[self.chapters[i]]],
                reference = ground_truth[i],
                # reference_contexts = [textbook[self.chapters[i]]] # for now this is just the chapter text, maybe should remove
            ))

        dataset = EvaluationDataset(samples = samples)

       # i really dont like this evaluate() function. its been very inconsistent for me
        if metrics is None:
            print(evaluate(dataset = dataset))
        else:
            print(evaluate(dataset = dataset, metrics = metrics))

        return samples
        
