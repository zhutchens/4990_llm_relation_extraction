from urllib.request import urlopen
from nltk import word_tokenize 
# from nltk.tokenize import sent_tokenize
import pymupdf
from io import BytesIO
# from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings


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


def contains_title(chapters: list, text: str, completed: list, consider: int = None, current: str = None) -> str | None:
    '''
    Find if a page contains a starting point of a chapter. 
    
    Sometimes chapter names will take up multiple lines, making the function fail. So use consider argument\n to limit the string and find it

    Args:
        chapters (list): list of chapters
        text (str): text to search in
        completed (list): list of chapters that already have text built
        consider (int, default None): how many letters of the chapter names to consider
        current (str, default None): chapter currently being built

    Returns:
        str or None: none if not found, chapter name otherwise
    '''
    if consider is None:
        if current is None:
            for chapter in chapters[:len(completed) + 2]:
                if chapter in text and chapter not in completed:
                    return chapter

            return None
        else:
            for chapter in chapters[:len(completed) + 2]:
                if chapter in text and chapter not in completed and chapter != current:
                    return chapter

            return None
    else:
        if current is None:
            for chapter in chapters[:len(completed) + 2]:
                # print(f'Chapters being considered:', chapters[:len(completed) + 1])
                # print('Currently filling chapter:', current)
                if chapter[:consider] in text and chapter not in completed:
                    return chapter

            return None
        else:
            for chapter in chapters[:len(completed) + 2]:
                # print(f'Chapters being considered:', chapters[:len(completed) + 1])
                # print('Currently filling chapter:', current)
                if chapter[:consider] in text and chapter not in completed and chapter != current:
                    return chapter

            return None


def is_substring(substring: str, chapters: list[str]) -> bool:
    '''
    Checks if a string is inside a given chapter

    Args:
        substring (str): the substring to check for
        chapters (list[str]): list of chapters to check in

    Returns:
        bool: True if substring is found, False otherwise
    '''
    for chapter in chapters:
        if substring in chapter:
            return True

    return False


# NOTE: this function splits the three textbooks correctly by chapter (hadoop, intro. to algorithms, data structures and algorithms)
# Not sure if it generalizes well to other types of docs (assignments, other textbooks, etc)
def split(link: str, chapters: list[str], stopword: str) -> dict[str, str]:
    '''
    Splits a web link into chapter/section chunks.

    Args:
        link (str): web link 
        chapters (list[str]): list of chapters in the web link
        stopword (str): first word where the last chapter ends (usually appendix or bibiliography)

    Returns:
        dict: key as chapter name, value as chapter content
    '''
    content = urlopen(link).read()
    doc = pymupdf.open('pdf', BytesIO(content))

    iterable = iter(doc) 
    next(iterable) # skip title page

    strings = {}
    chap_text = ''
    prev_found = ''
    chp = None
    break_out = False

    for page in iterable:
        data = page.get_text('dict', sort = True)
        full_t = page.get_text()


        # NOTE
        # maybe added paramter for these
        # or, add a parameter for starting page
        if 'Table of Contents' in full_t or 'Contents' in full_t or 'Preface' in full_t: 
            continue
    
        for block in data['blocks']:
            if 'lines' in block.keys():

                for line in block['lines']:
                    span = line['spans']
                    font = span[0]['font']
                    f_size = span[0]['size']
                    span_t = span[0]['text']

                    if chp is None:
                        chp = contains_title(chapters, span_t, list(strings.keys()), 20, prev_found)
                    else:
                        chp = contains_title(chapters, span_t, list(strings.keys()), 20, chp)

                    if stopword in span_t.lower() and len(strings) == len(chapters) - 1: 
                        strings[chapters[-1]] = chap_text
                        break_out = True
                        break # exit loop after ending of last chapter
                
                    if chp == chapters[0] and chp not in strings.keys(): 
                        prev_found = chp
                        chap_text += ' ' + span_t

                    if ('bold' in font.lower() or 'BX' in font) and chp is not None:
                        if f_size > 17 and span_t not in strings.keys() and (span_t in chapters or is_substring(span_t, chapters)):
                            strings[prev_found] = chap_text
                            chap_text = span_t
                            prev_found = chp
                    else:
                        chap_text += ' ' + span_t

        if break_out:
            break

    return strings


def lc_split(link: str) -> list[str]:
    '''
    Splits the web link content using langchains recursive character text splitter

    Args:
        link (str): web link

    Returns:
        list[str]: web content strings
    '''
    # docs = []
    # # print(f'Inside lc_split: chapter texts is {chapter_texts}')
    # for chp in chapter_texts.keys():
    #     chap_text = chapter_texts[chp] 
    #     paragraph_str = ''

    #     # for sentence in sent_tokenize(chap_text): # add individual sentences as docs
    #         # docs.append(Document(page_content = sentence.strip(), metadata = {'chapter': chp}))
    #     for sentence in re.split(r'[!?.]', chap_text):
    #         print('sentence:', sentence)
    #         if sentence.startswith('\n'): # new paragraph
    #             docs.append(Document(page_content = paragraph_str, metadata = {'chapter': chp}))
    #         else:
    #             paragraph_str += ' ' + sentence
    text = ''
    content = urlopen(link).read()
    doc = pymupdf.open('pdf', BytesIO(content))

    text = ' '.join([page.get_text() for page in doc])

    splitter = RecursiveCharacterTextSplitter(chunk_size = 3000, chunk_overlap = 100)

    return splitter.split_text(text = text)


# NOTE: similarity search can filter metadata of documents, but not sure how to add it in while splitting
# ex: document has 'Internal Sorting' as chapter_name, how do I tell that to the splitter?
def invoke_retriever(query: str) -> list[Document]:
    '''
    Invoke the retriever with a query or prompt. Must run create_retriever() first.

    Args:
        query (str): question or statement to give to the retriever

    Returns:
        list[Document]: relevant pages of documents
    '''

    # this should never happen but just in case!
    if not retriever:
        raise ValueError(f'Please run create_retriever() first before invoking it.')
    
    # return retriever.invoke(input = query)
    # return retriever.similarity_search(query = query)
    return retriever.as_retriever(search_type = 'similarity').invoke(query)


def create_retriever(link: str) -> None:
    '''
    Create retriever from the provided link

    Args:
        link (str): the website link to create a retriever form
        chapters (list): list of chapters in link
        stopword (str): first word (usually appendix or bibliography)
    
    Returns:
        None
    '''
    
    # texts = split(link, chapters, stopword)
    strings = lc_split(link)

    # print(f'Creating retriever with docs: {docs}')

    docs = []
    global retriever

    for string in strings:
        docs.append(Document(page_content = string))

    # retriever = BM25Retriever.from_documents(docs)
    retriever = InMemoryVectorStore(embedding = OpenAIEmbeddings())
    retriever.add_documents(documents = docs)