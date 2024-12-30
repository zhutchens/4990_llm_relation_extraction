

# def invoke_retriever(query: str) -> list[Document]:
#     '''
#     Invoke the retriever with a query or prompt. Must run create_retriever() first.

#     Args:
#         query (str): question or statement to give to the retriever

#     Returns:
#         list[Document]: relevant pages of documents
#     '''

#     # this should never happen but just in case!
#     if not retriever:
#         raise ValueError(f'Run create_retriever() first before invoking it.')
    
#     # return retriever.as_retriever(search_type = 'similarity').invoke(query)
#     return retriever.similarity_search(query)


# def create_retriever(link: str) -> None:
#     '''
#     Create retriever from the provided link

#     Args:
#         link (str): the website link to create a retriever form
#         chapters (list): list of chapters in link
#         stopword (str): first word (usually appendix or bibliography)
    
#     Returns:
#         None
#     '''
    
#     strings = lc_split(link)

#     docs = []
#     global retriever

#     for string in strings:
#         docs.append(Document(page_content = string))

#     retriever = InMemoryVectorStore(embedding = OpenAIEmbeddings())
#     retriever.add_documents(documents = docs)

class RetrievalSystem():
    def __init__():
        pass

    def retrieve():
        pass

    def invoke():
        pass

