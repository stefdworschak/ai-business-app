import os
from uuid import uuid4

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pinecone import Pinecone, ServerlessSpec


try:
    from google.colab import userdata
    PINECONE_API_KEY = userdata.get('PINECONE_API_KEY')
    OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
    OPENAI_BASE = userdata.get('OPENAI_BASE', 'https://47v4us7kyypinfb5lcligtc3x40ygqbs.lambda-url.us-east-1.on.aws/v1/')
except:
    from dotenv import load_dotenv
    load_dotenv('.env')
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_BASE = os.getenv('OPENAI_BASE', 'https://47v4us7kyypinfb5lcligtc3x40ygqbs.lambda-url.us-east-1.on.aws/v1/')


class VectorstoreFactory:
    def __init__(self, pinecone):
        self.pinecone = pinecone
        self.index = None
        self.embeddings = None
        self.vector_store = None
    
    def get_or_create_index(self, index_name, dimension=768):
        if not self.pinecone.has_index(index_name):
            self.pinecone.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        self.index = self.pinecone.Index(index_name)
        return self.index

    def get_embeddings(self):
        self.embeddings = OpenAIEmbeddings(
            model='text-embedding-3-small',
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_BASE,
        )
        return self.embeddings

    def get_vectorstore(self, index_name):
        self.vector_store =  PineconeVectorStore(
            index=self.get_or_create_index(index_name),
            embedding=self.get_embeddings())
        return self.vector_store

    def create_document(self, page_content, metadata={}):
        return Document(
            page_content=page_content,
            metadata=metadata
        )
    
    def chunk_text(self, text, chunk_size=500, chunk_overlap=50):
        """Chunk text into smaller pieces for better embedding performance."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return text_splitter.split_text(text)
    
    def create_chunked_documents(self, text, metadata={}, chunk_size=500, chunk_overlap=50):
        """Create multiple documents from chunked text."""
        chunks = self.chunk_text(text, chunk_size, chunk_overlap)
        documents = []
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_id'] = i
            chunk_metadata['total_chunks'] = len(chunks)
            documents.append(self.create_document(chunk, chunk_metadata))
            
        return documents
 
    def add_documents(self, documents, index_name):
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vector_store.add_documents(documents=documents, ids=uuids)

if __name__ == '__main__':
    pinecone = Pinecone(api_key=PINECONE_API_KEY)
    v = VectorstoreFactory(pinecone)
    vector_store = v.get_vectorstore('products')
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    import ipdb
    ipdb.set_trace()

    #doc = v.create_document("This is a test document", {"source": "test"})
    #v.add_documents([doc], 'test-index')