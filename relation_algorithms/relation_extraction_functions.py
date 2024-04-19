from langchain.llms import OpenAI
import graphviz
import os
import re
from IPython.display import display, Image
from pyvis.network import Network
import hypernetx as hnx
import matplotlib.pyplot as plt
import graphlib
from py3plex.core import multinet

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Functions for internal use only
def create_concept_graph_structure(param_list: list) -> dict:
    return_dict = {}

    for list_itm in param_list:
        return_dict[list_itm] = []

    return return_dict


def get_node_id_dict(dictionary: dict) -> dict:
    
    node_id_dict = {}
    id_count = 1
    
    for name in dictionary.keys():
        node_id_dict[name] = id_count
        id_count += 1

    return node_id_dict


# NOTE: Currently one main issue, the function that finds the associations between chapters seems to be broken. I think its the algorithm thats wrong. It also takes 10+ minutes to run
class LLM_Relation_Extractor:

    def __init__(self, link: str, token: str):
        # NOTE: Would it also be helpful to add a chapter dictionary as a parameter?
        '''
        Constructor to create a large language model relation extractor class. 

        Parameters / Attributes:
            link: The link to generate answers from
            token: OpenAI token
        
        Methods:
            find_commonalities()\n
            create_chapter_dict()\n
            identify_chapters()\n
            identify_main_topics()\n
            identify_main_topic_relations()\n
            identify_learning_outcomes()\n
            identify_learning_concepts()\n
            identify_assocations()\n
            identify_dependencies()\n
            print_flat_graph()\n
            get_assocation_interactive_graph()\n
            get_dependency_interactive_graph()\n

            
        Note that some of these functions may quite the amount of time to run depending on the size of whats in the link
        '''

        self.link = link

        os.environ['OPENAI_API_KEY'] = token
        self.llm = OpenAI()


    def create_chapter_dict(self, outcomes: list, concepts: list, chapters: dict | list) -> dict:
        '''
        Create a chapter dictionary containing the chapter names as keys and its learning concepts and outcomes in a tuple as the value. Concept is at index 0 and outcome is at index 1

        Parameters:
            outcomes: a list of learning outcomes created from the identify_learning_outcomes function
            concepts: a list of learning concepts created from the identify_learning_concepts function
            chapter_dict: a dictionary containing the chapter #'s as keys and the name as the value. Chapters must be in ascending order

        Returns:
            dict
        '''

        outcome_concept_graph = {}

        if type(chapters) != dict and type(chapters) != list:
            raise ValueError('Value of chapters is wrong. Should be a dictionary or a list')

        if type(chapters) is dict:
            i = 0
            for chapter_name in chapters.values():
                outcome_concept_graph[chapter_name] = (concepts[i], outcomes[i])
                i += 1
        else:
            for idx, name in enumerate(chapters):
                outcome_concept_graph[name] = (concepts[idx], outcomes[idx])
                
        return outcome_concept_graph
        
    
    def identify_chapters(self) -> dict:
        '''
        Identify the chapters within the class provided link using a large language model
        It is very, very inconsistent and I highly recommened manually creating the dictionary

        Parameters:
            None

        Returns:
            dict
        '''
        # NOTE: This is very, very, inconsistent. Do not recommend using this.

        chapters = self.llm(f"Please identify the chapters in this textbook: {self.link}")
        chapters = chapters.split('\n')
        chapter_dict = {}
        for idx, chapter in enumerate(chapters):
            chapter_dict[f"Chapter {idx + 1}"] = chapter

        return chapter_dict


    def identify_main_topics(self) -> list:
        '''
        Identify the main topics within the class provided link

        Parameters:
            None
        
        Returns:
            list
        '''
        main_topics = self.llm(f"Please identify ten main topics from this textbook: {self.link}")
        return [topic for topic in main_topics.split('\n')][2:]
    
    
    def identify_main_topic_relations(self, main_topic_list: list) -> dict:
        '''
        Identify the relationships between the main topics of the textbook

        Parameters:
            main_topics_list: A list of main topics from the textbook. Can be automatically created using the identify_main_topics() function
        
        Returns:
            dict
        '''
        topic_relations = create_concept_graph_structure(main_topic_list)

        relation = ''
        for i in range(len(main_topic_list)):
            for j in range(len(main_topic_list)):
                if i != j:
                    relation = self.llm(f"Is there a relationship between this topic: {main_topic_list[i]}, and this topic: {main_topic_list[j]}? If there is NOT, please respond with 'No' and 'No' only.")
                    relation = re.sub(re.compile('^[a-zA-Z\s\.,!?]'), '', relation)
                    if relation.split(',')[0].strip() != 'No':
                        topic_relations[main_topic_list[i]].append(main_topic_list[j])

        return topic_relations

    
    def identify_learning_outcomes(self, chapters: dict | list) -> list:
        '''
        Identify the main learning outcomes within the class provided link

        Parameters:
            chapter_dict: a dictionary containing the chapter #'s as keys and the name as the value. Chapters must be in ascending order. If a list is passed in, the list only needs to be the name of the chapters
        
        Returns:
            list
        '''
        outcome_list = []
        current_outcome = ''
    
        if type(chapters) is dict:
            for chapter_num, chapter_name in chapters.items():
                current_outcome = self.llm(f"Please identify the main learning outcomes given {chapter_num}, the chapter name is {chapter_name}. Here is the textbook in which to retrieve them: {self.link}")
                current_outcome = re.sub(re.compile('^[a-zA-Z\s\.,!?]'), '', current_outcome)
                outcome_list.append(current_outcome.split('\n'))
        else:
            for name in chapters:
                current_outcome = self.llm(f"Please identify the main learning outcomes given this chapter: {name}. Here is the textbook in which to retrieve them: {self.link}")
                current_outcome = re.sub(re.compile('^[a-zA-Z\s\.,!?]'), '', current_outcome)
                outcome_list.append(current_outcome.split('\n'))
        
        return outcome_list
    

    def identify_learning_concepts(self, chapters: dict | list) -> list:
        '''
        Identify the main learning concepts within the class provided link

        Parameters:
            chaptes: a dictionary containing the chapter #'s as keys and the name as the value. Chapters must be in ascending order. If a list, only the chapter names are needed
        
        Returns:
            list
        '''

        concept_list = []
        current_concept = ''

        if type(chapters) is dict:
            for chapter_num, chapter_name in chapters.items():
                current_concept = self.llm(f"Please identify the main learning concepts given {chapter_num}, the chapter name is {chapter_name}. Here is the textbook in which to retrieve them: {self.link}. Additionally, please limit your response to 10 concepts.")
                current_concept = re.sub(re.compile('^[a-zA-Z\s\.,!?]'), '', current_concept)
                concept_list.append([concept for concept in current_concept.split('\n') if concept != ''])
        else:
            for name in chapters:
                current_concept = self.llm(f"Please identify the main learning concepts given this chapter: {name}. Here is the textbook in which to retrieve them: {self.link}. Additionally, please limit your response to 10 concepts.")
                current_concept = re.sub(re.compile('^[a-zA-Z\s\.,!?]'), '', current_concept)
                concept_list.append([concept for concept in current_concept.split('\n') if concept != ''])

        return concept_list
        

    def identify_associations(self, learning_dict: dict) -> dict:
        '''
        Identify associations between chapters. For example, if there is an association between Chapter 1 and 3 it will be added to the dictionary. The return dict contains chapter names as keys and the chapter names its associated with as values

        Parameters:
            learning_dict: The dictionary returned from the create_chapter_dict() function

        Returns:
            dict
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


    def identify_dependencies(self, concept_dict: dict) -> dict:
        '''
        Identify the dependency relationships between chapters, returns a dictionary where the key is a chapter name and the value is a list of chapters it depends on

        Parameters: 
            concept_dict: a dictionary containing the chapter names as keys and their learning concepts as values. Dictionary should come from the create_chapter_dict() function

        Returns:
            dict
        '''

        relation = ''

        keys = list(concept_dict.keys())
        relations_dict = create_concept_graph_structure(keys)

        for i in range(len(keys)):
            for j in range(1, len(keys)):
                relation = self.llm(f"Please identify if these concepts: {concept_dict[keys[i]][0]} are a prerequisite for these concepts: {concept_dict[keys[j]][0]}. If there is NO prerequisite, please respond with 'No' and 'No' only.")
                relation = re.sub(re.compile('[^a-zA-Z\s\.,!?]'), '', relation)
                if relation.split(',')[0].strip() != 'No':
                    relations_dict[keys[j]].append(keys[i])

        return relations_dict


    def print_flat_graph(self, learning_concept_graph: dict) -> None:
        '''
        Print a directed graph to the screen using either the association dictionary or dependency dictionary

        Parameters:
            learning_concept_graph: The dictionary to build the graph from. This should come from either the identify_associations function or identify_dependencies function
        
        Returns:
            None
        '''
        graph = graphviz.Digraph()

        for key, values in learning_concept_graph.items():
            graph.node(name = key)
            for value in values:
                graph.edge(key, value)

        display(Image(graph.pipe(format = "png", renderer = "cairo")))


    # NOTE: I think I might be able to combine these two functions into one
    def get_assocation_interactive_graph(self, learning_graph: dict, associations: dict) -> Network:
        '''
        Retrieve the interactive graph using the association dictionary. Nodes are chapter names and edges are the associations. Hovering over a node results in displaying that nodes learning outcomes and concepts. The function is not able to automatically display the graph so the .show() method must be called on the return object

        Parameters:
            learning_graph: The dictionary containing chapter names as keys and the values as a tuple containing the concept at index 0 and outcome at index 1. Can be created automatically using the create_chapter_dict function
            assocations: The dictionary containing the associations between chapters. Can be created automatically using the identify_associations() function
        
        Returns:
            A pyvis Network object
        '''

        graph = Network(notebook = True, cdn_resources = "remote")

        graph.toggle_physics(False)

        # Showing all interactivity options, but can be parameterized to only include some
        graph.show_buttons()
        
        node_id_dict = get_node_id_dict(learning_graph)

        for chapter_name, chapter_id in node_id_dict.items():
            graph.add_node(n_id = chapter_id, label = chapter_name, title = "Main Learning Concepts: " + learning_graph[chapter_name][0] + "\n" + "Main Learning Outcomes:" + learning_graph[chapter_name][1])

        for key, values in associations.items():
            for value in values:
                graph.add_edge(node_id_dict[key], node_id_dict[value])

        return graph


    def get_dependency_interactive_graph(self, dependency_dict: dict) -> Network:
        '''
        Retrieve the interactive graph using the dependency dictionary. The function is not able to automatically display the graph so the .show() method must be called on the return object

        Parameters:
            dependency_dict: A dictionary containing the chapter_names between chapters. Can be created automatically using the dependency_relation_extraction() function
        
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
    

    def draw_hypergraph(self, dictionary: dict) -> None:
        '''
        Generate and display a hypergraph given a dependency dictionary generated from identify_dependencies() or association dict generated by identify_associations()

        Parameters:
            dependencies: A dictionary of dependencies. Can be generated by identify_dependencies(). The key should be a chapter name and the value a list of chapters it depends on

        Returns:
            None
        '''
                
        sorted_dependencies = graphlib.TopologicalSorter(graph = dictionary)

        temp = sorted_dependencies
        sorted_dependencies = {}

        for value in temp:
            sorted_dependencies[value] = dictionary[value]

        hypergraph = hnx.Hypergraph(sorted_dependencies)
        hnx.draw(hypergraph)
        plt.title("Hypergraph")
        plt.show()


    def draw_multi_layered_graph(dictionary: dict) -> None:
        '''
        Generate and display a multilayered graph given a dependency dictionary generated from identify_dependencies() or association dict generated by identify_associations()

        Parameters:
            dependencies: A dictionary of dependencies. Can be generated by identify_dependencies(). The key should be a chapter name and the value a list of chapters it depends on

        Returns:
            None
        '''
        
        multi_graph = multinet.multi_layer_network(network_type = "multiplex")

        for node, edges in dictionary.items():
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