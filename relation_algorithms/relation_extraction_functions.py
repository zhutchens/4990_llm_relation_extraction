from langchain.llms import OpenAI
import graphviz
import os
from dotenv import load_dotenv
import re
from IPython.display import display, Image
import plotly

# Retrieve OpenAI token
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('team_token')

# LLM to use in relation extraction 
llm = OpenAI()

# For internal usage only
def create_concept_graph_structure(param_list: list) -> dict:

    return_dict = {}
    for list_itm in param_list:
        return_dict[list_itm] = []

    return return_dict


def identify_main_topics(textbook_link: str) -> list:
    main_topics = llm(f"Please identify ten main topics from this textbook: {textbook_link}")
    return [topic for topic in main_topics.split('\n')]


def identify_learning_concepts_outcomes(chapter_dict: dict, link: str) -> list:

    return_list = []
    current_concept = ''
    for chapter, chapter_name in chapter_dict.items():
        current_concept = llm(f"Please identify the main learning outcomes and main learning concepts given {chapter}, the chapter name is {chapter_name}. Here is the textbook in which to retrieve them: {link}")
        current_concept = re.sub(re.compile('[^a-zA-Z\s\.,!?]'), '', current_concept)
        return_list.append(current_concept)
    
    return return_list


def association_identification(concept_list: list) -> dict:

    concept_graph = create_concept_graph_structure(concept_list)
    new_association = ''
    for i in range(len(concept_list)):
        for j in range(i + 1, len(concept_graph)):
            new_association = llm(f"Please identify if there is an association between this concept: {concept_list[i]}, and this other concept: {concept_list[j]}. If there is NO association, please start your response with 'No' and 'No' only.")
            new_association = re.sub(re.compile('[^a-zA-Z\s\.,!?]', '', new_association))
            if new_association.split(',')[0].strip() != 'No':
                concept_graph[concept_list[i]].append(concept_list[j])
    
    return concept_graph


def dependency_relation_extraction(keys: list, relations_dict = None | dict) -> dict:
    
    relation = ''

    if relations_dict == None:
        relations_dict = create_concept_graph_structure(keys)

    for i in range(len(keys)):
        for j in range(len(keys)):
            if i != j:
                relation = llm(f"Is there a dependency relationship from {keys[i]} to {keys[j]}? If there is NOT, please respond with 'No' and 'No' only.")
                relation = re.sub(re.compile('[^a-zA-Z\s\.,!?]',), '', relation)
                if relation.split()[0] != 'No':
                    relations_dict[keys[j]].append(keys[i])

    return relations_dict


def print_flat_graph(learning_concept_graph: dict) -> None:

    graph = graphviz.Graph()

    for key, values in learning_concept_graph.items():
        graph.node(name = key)
        for value in values:
            graph.edge(key, value)

    display(Image(graph.pipe(format = "png", renderer = "cairo")))


def print_interactive_graph() -> None:
    # TODO: Finish code here to print an interactive graph 
    print('hello!!')

