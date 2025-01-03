from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

class TransformerEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L12-v2')


    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        '''
        Embed documents 

        Args:
            texts (list[str]): documents to embed

        Returns:
            np.ndarray: embeddings
        '''
        return [self.model.encode(text).flatten().tolist() for text in texts]


    def embed_query(self, text: str) -> list[float]:
        '''
        Embed query

        Args:
            text (str): query to embed

        Returns:
            np.ndarray: embedded query
        '''
        return self.model.encode(sentences = [text]).flatten().tolist()