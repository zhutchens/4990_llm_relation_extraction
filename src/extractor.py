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
from ragas import evaluate as ev
from ragas import SingleTurnSample, MultiTurnSample
from ragas.dataset_schema import EvaluationDataset
from src.utils import create_concept_graph_structure, split, get_node_id_dict, create_retriever, invoke_retriever
import networkx as nx


# NOTE: Currently one main issue, the function that finds the associations between chapters seems to be broken. I think its the algorithm thats wrong. It also takes 10+ minutes to run
class relationExtractor:
    def __init__(self, link: str, token: str, chapters: list[str], stopword: str):
        '''
        Constructor to create a large language model relation extractor class. 

        Args:
            link (str): the web link to generate answers from
            token (str): OpenAI token
            chapters (list): list of chapter/section names in link
            stopword (str): word where last chapter ends (usually appendix or bibliography)
        '''
        self.link = link
        self.chapters = chapters
        self.stopword = stopword
  
        os.environ['OPENAI_API_KEY'] = token
        self.llm = ChatOpenAI(temperature = 0)

        create_retriever(self.link)


    def identify_key_terms(self, n_terms: int, input_type: str = 'chapter', chapter_name: str = None, concepts: list[str] = None) -> list[str] | dict[str, list[str]]:
        '''
        Identify the key terms for a chapter or group of concepts
        
        Args:
            n_terms (int): number of key terms to use
            input_type (str, default chapter): if chapter, get key terms of a chapter. if concepts, get key terms for a list of concepts
            chapter_name (str, default None): name of chapter to get key terms for 
            concepts (list[str], default None): list of concepts to get key terms for

        Returns:
            list[str]: if getting key terms for a chapter\n
            dict[str, list[str]]: if getting key terms from a list of concepts
        '''
        if not isinstance(n_terms, int):
            raise TypeError(f'n_terms must be int type, got {type(n_terms)}')

        if concepts is None and chapter_name is None:
            raise ValueError(f'chapter_name and concepts cannot both be None')
        
        if input_type == 'chapter':
            prompt = f'''
                    Identify {n_terms} key terms from Chapter {chapter_name} in descending order of significance, listing the most significant terms first. The textbook is available here: {self.link}.
                    For each term, provide the following:
                    - Confidence Interval (CI),
                    - Corresponding Statistical Significance (p-value),
                    - A brief explanation of the term's relevance and importance,
                    Format:
                    term :: confidence interval :: p-value :: explanation
                    '''

            terms = self.llm.invoke(prompt).content 

            return [string for string in terms.split('\n') if string != '']

        elif input_type == 'concepts':
            concept_terms = {}
            for concept in concepts:
                words = self.llm.invoke(f'Identify {n_terms} key terms for the following concept: {concept}.').content
                concept_terms[concept] = [string for string in words.split('\n') if string != '']

            return concept_terms

        else:
            raise ValueError(f'input_type value must be chapter or concepts, got {input_type}')


    def summarize(self) -> str:
        '''
        Returns summary of object web link

        Args:
            None

        Returns:
            str: summary of textbook
        '''
        return self.llm.invoke(f"Please summarize this web page: {self.link}").content


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
        
    # note: fix later
    # def identify_chapters(self) -> dict[str, str]:
    #     '''
    #     Identify the chapters within the class provided link using a large language model
    #     It is very, very inconsistent and I highly recommened manually creating the dictionary

    #     Args:
    #         None

    #     Returns:
    #         dict[str, str]: key is chapter number, value is chapter name
    #     '''
    #     # NOTE: This is very, very, inconsistent. Do not recommend using this.

    #     # chapters = self.llm.invoke(f"Please identify the chapters in this textbook: {self.link}").content
    #     chapters = self.llm.invoke().content
    #     chapters = chapters.split('\n')
    #     chapter_dict = {}
        
    #     for idx, chapter in enumerate(chapters):
    #         chapter_dict[f"Chapter {idx + 1}"] = chapter

    #     return chapter_dict


    def identify_main_topics(self) -> list[str]:
        '''
        Identify the main topics within the class provided link

        Args:
            None
        
        Returns:
            list[str]: list of main topics 
        '''

        prompt = f'''
                    Please identify the main topics from this textbook: {self.link}. Please provide justification.
                    Format:
                    main topic :: justification
                  '''
        main_topics = self.llm.invoke(prompt).content

        return [topic for topic in main_topics.split('\n') if topic != '']
    
    
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

    
    def identify_outcomes(self, concepts: list[list[str]] = None, key_terms: dict[str, list[str]] = None) -> list[list[str]]:
        '''
        Identify main learning outcomes within the class provided link. If concepts and key_terms are both None, get key terms from web link content

        Args:
            concepts (list[list[str]]):
            key_terms (dict[str, list[str]]): 
            

        Returns:
            list[list[str]]: list of learning outcomes
        '''
        outcome_list = []

        if concepts is None and key_terms is None:
            for name in self.chapters:
                response = self.llm.invoke(f"Identify the main learning outcomes given this chapter: {name}. Here is the textbook in which to retrieve them: {self.link}").content
                outcome_list.append([outcome for outcome in response.split('\n') if outcome != ''])

            return outcome_list
        
        elif concepts is not None and key_terms is None:
            for concept in concepts:
                concept = ' '.join(concept)
                response = self.llm.invoke(f'Identify the main learning outcomes from these concepts: {concept}').content
                outcome_list.append([outcome for outcome in response.split('\n') if outcome != ''])
            
            return outcome_list

        elif concepts is None and key_terms is not None:
            for k in key_terms.keys():
                terms = ' '.join(key_terms[k])
                response = self.llm.invoke(f'Identify the main learning outcomes given this concept {k} and these key terms {terms}')
                outcome_list.append([outcome for outcome in response.split('\n') if outcome != ''])

            return outcome_list
        else:
            raise ValueError(f'concepts and key_terms cannot both have a value. one of them must be None')


    def identify_concepts(self) -> list[list[str]]:
        '''
        Identify the main learning concepts within the class provided link

        Args:
            None
        
        Returns:
            list[list[str]]: list of concepts for each chapter
        '''

        concept_list = []
        current_concept = ''
        global retrieved_contexts
        retrieved_contexts = {}


        for name in self.chapters:

            key_terms = self.identify_key_terms(chapter_name = name, n_terms = 10) # 10 for now
            key_terms.append(name)

            relevant_docs = invoke_retriever(' '.join(key_terms))
            retrieved_contexts[name] = ''.join([text.page_content for text in relevant_docs])

            # single shot prompt 
            single_prompt = f'''
                             Identify the ten most important learning concepts for chapter: {name}. 
                             The relevant documents can be found here: {relevant_docs}
                             '''

            # not sure how this works
            # few_shot_prompt = f''' 
            #                    Q:
            #                    A:

            #                    Q:
            #                    A:

            #                    Q: 
            #                    '''

            current_concept = self.llm.invoke(single_prompt).content
            concept_list.append([concept for concept in current_concept.split('\n') if concept != ''])

        return concept_list

    
    # def get_assocations(self, first: list[list[str]] | list[str], second: list[list[str]] | list[str]) -> list[str]:
    #     '''
    #     Get the assocations between the first list and second list

    #     Args:
    #         first (list[list[str]] | list[str]): first list (concepts, key terms, outcomes, etc)
    #         second (list[list[str]] | list[str]): second list (concepts, key terms, outcomes, etc)

    #     Returns:
    #         list[tuple[str, str, str]]
    #     '''
    #     associations = []

    #     for f in first:
    #         if isinstance(f, list):
    #             f = ' '.join(f)

    #         for s in second:
    #             if isinstance(s, list):
    #                 s = ' '.join(s)

    #             prompt = f'''
    #                     Identify if there is some assocation between {f} and {s}. If there is an assocation, your response should ONLY include the keywords of the assocation. 
    #                     Otherwise, respond with No and No only.
    #                     '''
    #             response = self.llm.invoke(prompt).content

    #             print(response)
    #             if response.lower() != 'no':
    #                 associations.append((f, s, response))

    #     return associations


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


    def draw_layered_graph(self, dictionary: dict[str, list[str]]) -> None:
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


    def evaluate(self, concepts: list[list[str]], ground_truth: list[str], metrics: list, multi_turn: bool = False) -> list[SingleTurnSample]:
        '''
        Evaluate concepts or outcomes from LLM  

        Args:
            concepts (dict): dictionary with chapter name as key and value as the list of concepts
            ground_truth (list): ground truth concepts 
            metrics (list, default None): list of metrics to use from ragas library
            multi_turn (bool, default False): if True, use MultiTurnSample() for evaluation

        Returns:
            list[SingleTurnSample]: list of samples used for evaluation  
        '''      
        samples = []

        textbook = split(self.link, self.chapters, self.stopword) # for reference contexts. idk if this should stay or not tbh
        for i in range(len(ground_truth)):
            if not multi_turn:
                samples.append(SingleTurnSample(
                    user_input = f'Identify the ten most important learning concepts for chapter: {self.chapters[i]}. The context can be found here: {retrieved_contexts[self.chapters[i]]}',
                    response = ' '.join(concepts[i]),
                    retrieved_contexts = [retrieved_contexts[self.chapters[i]]],
                    reference = ground_truth[i],
                    # reference_contexts = [textbook[self.chapters[i]]] # for now this is just the chapter text, maybe should remove
                ))
            else: # use multiturn samples with chain of thought (how? i dont get it)
                # samples.append(MultiTurnSample(
                #     user_input = []
                # ))
                pass

        dataset = EvaluationDataset(samples = samples)

        if metrics is None:
            print(ev(dataset = dataset))
        else:
            print(ev(dataset = dataset, metrics = metrics))

        return samples


    def build_terminology(self, concept_terms: list[list[str]]) -> list[str]:
        '''
        Build a terminology using is-a relationships between key terms

        Args:
            key_terms (list[str]): key terms to use

        Returns:
            list[str]: terminology 
        '''
        terminology = []
        concept_terms = [word for concepts in concept_terms for word in concepts]

        for i in range(len(concept_terms)):
            for j in range(len(concept_terms)):
                if i != j:
                    prompt = f'''
                            Q: Is there an is-a relationship present between dog and mammal?
                            A: Yes

                            Q: Is there an is-a relationship present between vehicle and car?
                            A: No

                            Q: Is there an is-a relationship presnet between {concept_terms[i]} and {concept_terms[j]}?
                            '''
                    response = self.llm.invoke(prompt).content
                    if 'yes' in response.lower():
                        terminology.append((concept_terms[i], concept_terms[j]))

        return terminology


    def build_KG(self) -> nx.Graph:
        '''
        Builds a knowledge using learning concepts, outcomes, and key terms

        Args:
            concepts (): learning concepts
            outcomes (): learning outcomes
            key_terms (): key_terms for chapters
        
        Returns:
            Graph: networkx object 
        '''
        kg = nx.Graph()


        return kg