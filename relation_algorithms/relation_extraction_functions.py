from langchain.llms import OpenAI
import graphviz
import os
from dotenv import load_dotenv
import re
from IPython.display import display, Image
from pyvis.network import Network

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


class LLM_Relation_Extractor:

    def __init__(self, textbook_link: str):
        self.link = textbook_link
    
    def create_chapter_dict(self, outcomes: list, concepts: list, chapter_dict: dict) -> dict:
        outcome_concept_graph = {}

        for chapter, chapter_name in chapter_dict.items():
            for i in range(len(outcomes)):
                outcome_concept_graph[chapter] = (chapter_name, concepts[i], outcomes[i])

        return outcome_concept_graph

    def identify_chapters(self) -> dict:
        # NOTE: This is very, very, inconsistent. Do not recommend using this.

        chapters = llm(f"Please identify the chapters in this textbook: {self.link}")
        chapters = chapters.split('\n')
        chapter_dict = {}
        for idx, chapter in enumerate(chapters):
            chapter_dict[f"Chapter {idx + 1}"] = chapter

        return chapter_dict


    def identify_main_topics(self) -> list:
        main_topics = llm(f"Please identify ten main topics from this textbook: {self.link}")
        return [topic for topic in main_topics.split('\n')][2:]


    def identify_learning_outcomes(self, chapter_dict: dict) -> list:
        outcome_list = []
        current_outcome = ''

        for chapter, chapter_name in chapter_dict.items():
            current_outcome = llm(f"Please identify the main learning outcomes given {chapter}, the chapter name is {chapter_name}. Here is the textbook in which to retrieve them: {self.link}")
            outcome_list.append(current_outcome)
        
        return outcome_list
    

    def identify_learning_concepts(self, chapter_dict: dict) -> list:
        concept_list = []
        current_concept = ''

        for chapter, chapter_name in chapter_dict.items():
            current_concept = llm(f'Please identify the main learning concept given {chapter}, the chapter name is {chapter_name}. Here is the textbook in which to retrieve them: {self.link}')
            concept_list.append(current_concept)

        return concept_list


    def identify_associations(self, learning_dict: dict) -> dict:
        chapter_names = [values[0] for values in list(learning_dict.values())]
        association_dict = create_concept_graph_structure(param_list = chapter_names)
        new_association = ''

        # Use dictionary created from create_chapter_dict function
        
        for i in range(len(list(learning_dict.values()))):
            current_tuple = list(learning_dict.values())[i]
            for j in range(len(list(learning_dict.values()))):
                next_tuple = list(learning_dict.values())[j]
                new_association = llm(f"Please identify if there is an association between this concept: {current_tuple[1]}, and this other concept: {next_tuple[1]}. If there is NO association, please start your response with 'No' and 'No' only.")
                new_association = re.sub(re.compile('[^a-zA-Z\s\.,!?]'), '', new_association)
                # Try to only add associations to the graph, but its difficult because sometimes the LLM won't start its response with 'No'
                if new_association.split(',')[0].strip() != 'No':
                    association_dict[current_tuple[0]].append(next_tuple[0])
        
        return association_dict


    def dependency_relation_extraction(self, dependiences: list, relations_dict = None) -> dict:
        relation = ''

        if relations_dict == None:
            relations_dict = create_concept_graph_structure(dependiences)

        for i in range(len(dependiences)):
            for j in range(len(dependiences)):
                if i != j:
                    relation = llm(f"Is there a dependency relationship from {dependiences[i]} to {dependiences[j]}? If there is NOT, please respond with 'No' and 'No' only.")
                    relation = re.sub(re.compile('[^a-zA-Z\s\.,!?]'), '', relation)
                    if relation.split()[0] != 'No':
                        relations_dict[dependiences[j]].append(dependiences[i])

        return relations_dict


    def print_flat_graph(self, learning_concept_graph: dict) -> None:
        graph = graphviz.Graph()

        for key, values in learning_concept_graph.items():
            graph.node(name = key)
            for value in values:
                graph.edge(key, value)

        display(Image(graph.pipe(format = "png", renderer = "cairo")))


    def print_interactive_graph(self, learning_graph: dict, associations: dict) -> Network:

        graph = Network(notebook = True, cdn_resources = "remote")

        node_id_dict = {}
        id_count = 1
        for values in learning_graph.values():
            node_id_dict[values[0]] = id_count
            id_count += 1

        for chapter_name, chapter_id in node_id_dict.items():
            graph.add_node(n_id = chapter_id, label = chapter_name)


        for key, values in associations.items():
            for value in values:
                graph.add_edge(node_id_dict[key], node_id_dict[value])

        return graph