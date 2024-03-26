from langchain.llms import OpenAI
import graphviz
import os
from dotenv import load_dotenv
import re
from IPython.display import display, Image
from pyvis.network import Network
import hypernetx as hnx

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Retrieve OpenAI token
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('team_token')

# LLM to use in relation extraction 
llm = OpenAI()

# For internal use only
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

class LLM_Relation_Extractor:

    def __init__(self, link: str):
        # NOTE: Would it also be helpful to add a chapter dictionary as a parameter?
        '''
        Constructor to create a large language model relation extractor class. 

        Parameters / Attributes:
            link: The link to generate answers from
        
        Methods:
            create_chapter_dict(),
            identify_chapters(),
            identify_main_topics(),
            identify_main_topic_relations(),
            identify_learning_outcomes(),
            identify_learning_concepts(),
            identify_assocations(),
            dependency_relation_extraction(),
            print_flat_graph(),
            get_assocation_interactive_graph(),
            get_dependency_interactive_graph(),

            
        Note that some of these functions may take a long time to run depending on the size of whats in the link. Especially the identify_assocations() and dependency_relation_extraction() functions
        '''
        self.link = link

    def find_commonalities(chapters: dict, link: str) -> tuple:
        '''
        Find the learning concepts and learning outcomes for each chapter in the textbook 5 times, compare the results, and find what is in common. Run this twice and return final results as a tuple.
        
        Parameters:
            chapters: a dictionary of chapters
            link: the textbook link

        Returns:
            tuple, the concepts are at index 0 and outcomes are at index 1 
        '''

        common_concepts_dict = {}
        common_outcomes_dict = {}

        for name in chapters.values():
            common_concepts_dict[name] = []
            common_outcomes_dict[name] = []

        for i in range(2):
            for j in range(5):
                for chapter, chapter_name in chapters.items():
                    learning_concept = llm(f"Please identify the main learning concepts given {chapter}, the chapter name is {chapter_name}. Here is the textbook in which to retrieve them: {link}")
                    learning_concept = re.sub(re.compile('[^a-zA-Z\s\.,!?]'), '', learning_concept)
                    common_concepts_dict[chapter_name][i].append([concept for concept in learning_concept.split('\n')[2:] if concept != ''])
            
                    learning_outcome = llm(f"Please identify the main learning outcomes given {chapter}, the chapter name is {chapter_name}. Here is the textbook in which to retrieve them: {link}")
                    learning_outcome = re.sub(re.compile('[^a-zA-Z\s\.,!?]'), '', learning_outcome)
                    common_outcomes_dict[chapter_name][i].append([outcome for outcome in learning_outcome.split('\n')[2:] if outcome != ''])

        in_common_concepts = {}
        in_common_outcomes = {}

        for key in chapters.values():
            in_common_concepts[key] = []
            in_common_outcomes[key] = []

        first_key = list(common_concepts_dict.keys())[0]

        for idx in range(len(common_concepts_dict[first_key])):
            for chapter_name, chapter_concepts in common_concepts_dict.items():
                content = llm(f"Can you identify the common concepts between these lists of concepts for chapter {chapter_name}? {chapter_concepts[idx][0]}, {chapter_concepts[idx][1]}, {chapter_concepts[idx][2]}, {chapter_concepts[idx][3]}, {chapter_concepts[idx][4]}?")
                in_common_concepts[chapter_name].append(content.split('\n')[2:])
            
            for chapter_name, chapter_outcomes in common_outcomes_dict.items():
                content = llm(f"Can you identify the common learning outcomes between these lists for chapter {chapter_name}? {chapter_outcomes[idx][0]}, {chapter_outcomes[idx][1]}, {chapter_outcomes[idx][2]}, {chapter_outcomes[idx][3]}, {chapter_outcomes[idx][4]}?")
                in_common_outcomes[chapter_name].append(content.split('\n')[2:])

        final_common_concept_dict = {}
        final_common_outcome_dict = {}

        for key in in_common_concepts.keys():
            content = llm(f"Please identify the common concepts between these two lists: {in_common_concepts[key][0]}, {in_common_concepts[key][1]}")
            final_common_concept_dict[key] = content.split('\n')[2:]

            content = llm(f"Please identify the common learning outcomes between these two lists: {in_common_outcomes[key][0]}, {in_common_outcomes[key][1]}")
            final_common_outcome_dict[key] = content.split('\n')[2:]

        return final_common_concept_dict, final_common_outcome_dict

    
    def create_chapter_dict(self, outcomes: list, concepts: list, chapter_dict: dict) -> dict:
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

        i = 0
        for chapter_name in chapter_dict.values():
            outcome_concept_graph[chapter_name] = (concepts[i], outcomes[i])
            i += 1
                
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

        chapters = llm(f"Please identify the chapters in this textbook: {self.link}")
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
        main_topics = llm(f"Please identify ten main topics from this textbook: {self.link}")
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
                    relation = llm(f"Is there a relationship between this topic: {main_topic_list[i]}, and this topic: {main_topic_list[j]}? If there is NOT, please respond with 'No' and 'No' only.")
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
                current_outcome = llm(f"Please identify the main learning outcomes given {chapter_num}, the chapter name is {chapter_name}. Here is the textbook in which to retrieve them: {self.link}")
                current_outcome = re.sub(re.compile('^[a-zA-Z\s\.,!?]'), '', current_outcome)
                outcome_list.append(current_outcome)
        else:
            for name in chapters:
                current_outcome = llm(f"Please identify the main learning outcomes given this chapter: {name}. Here is the textbook in which to retrieve them: {self.link}")
                current_outcome = re.sub(re.compile('^[a-zA-Z\s\.,!?]'), '', current_outcome)
                outcome_list.append(current_outcome)
        
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
                current_concept = llm(f"Please identify the main learning concepts given {chapter_num}, the chapter name is {chapter_name}. Here is the textbook in which to retrieve them: {self.link}")
                current_concept = re.sub(re.compile('^[a-zA-Z\s\.,!?]'), '', current_concept)
                concept_list.append(current_concept)
        else:
            for name in chapters:
                current_concept = llm(f"Please identify the main learning concepts given this chapter: {name}. Here is the textbook in which to retrieve them: {self.link}")
                current_concept = re.sub(re.compile('^[a-zA-Z\s\.,!?]'), '', current_concept)
                concept_list.append(current_concept)

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

        # I think this algorithm is wrong? Maybe its better to loop through in reverse order (Ex. start with the last chapter and work downwards)
        for i in range(len(values)):
            current_tuple = values[i]
            for j in range(i + 1, len(values)):
                next_tuple = values[j]
                new_association = llm(f"Please identify if there is an association between this concept: {current_tuple[0]}, and this other concept: {next_tuple[0]}. If there is NO association, please start your response with 'No' and 'No' only.")
                new_association = re.sub(re.compile('[^a-zA-Z\s\.,!?]'), '', new_association)
                # Try to only add associations to the graph, but its difficult because sometimes the LLM won't start its response with 'No'
                if new_association.split(',')[0].strip() != 'No':
                    association_dict[keys[i]].append(keys[j])

        # Would identifying the assocations like this be better? It might be too large to be processed like this though
        # assocation = llm(f"Given this dictionary containing chapter names and their learning concepts and outcomes, please identify the associations between chapters: {learning_dict}")
                
        
        return association_dict


    def dependency_relation_extraction(self, dependencies: list) -> dict:
        '''
        Identify the dependency relationships between chapters, returns a dictionary where the key is a chapter name and the value is a list of chapters it depends on

        Parameters: a list containing the chapter names to test which ones depend on each other

        Returns:
            dict
        '''
        relation = ''

        relations_dict = create_concept_graph_structure(dependencies)

        for i in range(len(dependencies)):
            for j in range(len(dependencies)):
                if i != j:
                    # NOTE: Should this be modified any? It seems to work ok
                    relation = llm(f"Is there a dependency relationship from {dependencies[i]} to {dependencies[j]}? If there is NOT, please respond with 'No' and 'No' only.")
                    relation = re.sub(re.compile('[^a-zA-Z\s\.,!?]'), '', relation)
                    if relation.split()[0] != 'No':
                        relations_dict[dependencies[j]].append(dependencies[i])

        return relations_dict


    def print_flat_graph(self, learning_concept_graph: dict) -> None:
        '''
        Print a directed graph to the screen using either the association dictionary or dependency dictionary

        Parameters:
            learning_concept_graph: The dictionary to build the graph from. This should come from either the identify_associations function or dependency_relation_extraction function
        
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
            dependency_dict: A dictionary containing the dependencies between chapters. Can be created automatically using the dependency_relation_extraction() function
        
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
    
    def draw_hypergraph(common_concepts = None | dict, common_outcomes = None | dict) -> None:
        '''
        Generate and print a hypergraph provided the common outcomes or concepts from the find_commonalities() function. One of the parameters must be None, both hypergraphs cannot be printed at the same time

        Parameters:
            common_concepts: dict or None. The common concepts dictionary returned from the find_commonalities() function
            common_outcomes: dict or None. The common outcomes dictionary returned from the find_commonalities() function

        Returns:
            None
        '''

        if common_concepts != None and common_outcomes != None:
            print('Cannot print hypergraph for both dictionaries. Please set one of the parameters to None.')
            return None
        
        keys = list(common_concepts.keys())
        dependencies = {}
        
        if common_concepts is not None:
            for key in keys:
                dependencies[key] = []

            for i in range(len(keys)):
                current_concept = common_concepts[keys[i]]
                for j in range(i + 1, len(keys)):
                    next_concept = common_concepts[keys[j]]
                    content = llm(f"Please identify if the current learning concept: {next_concept} has a prerequisite for the previous learning concept: {current_concept}. If there is NO prerequisite, please respond with 'No' and 'No' only.")
                    if content.split(',')[0].strip() != 'No':
                        dependencies[keys[j]].append(keys[i])
        else:
            for key in keys:
                dependencies[key] = []

            for i in range(len(keys)):
                current_outcome = common_outcomes[keys[i]]
                for j in range(i + 1, len(keys)):
                    next_outcome = common_outcomes[keys[j]]
                    content = llm(f"Please identify if the current learning outcome: {next_outcome} has a prerequisite for the previous learning outcome: {current_outcome}. If there is NO prerequisite, please respond with 'No' and 'No' only.")
                    if content.split(',')[0].strip() != 'No':
                        dependencies[keys[j]].append(keys[i])
                
        hypergraph = hnx.Hypergraph(dependencies)
        hnx.draw(hypergraph)
        