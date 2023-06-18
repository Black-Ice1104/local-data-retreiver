# Local Data Retriever
The local data retriever uses OpenAI, LangChain and Chroma to achieve a search engine based on imported local data.

## Operating Guide
### 1.  Config the environment parameters
Create the "config.env" file. Several things need to be specified:
- PERSIST_DIRECTORY
    
    *the place to put the database*
- EMBEDDINGS_MODEL_NAME
    
    *can either be a model from hugging face or openai*
- TARGET_SOURCE_CHUNKS

    *the number of chunks that'd be chosen as refrences to the answer*
- SOURCE_DIRECTORY

    *the source of local data*
- OPENAI_API_KEY
### 2. Install the needed sdk and packages
- Python: ≥ 3.9
- LangChain: 
    
    pip install langchain
- Chroma:
    
    pip install chromadb

    *Chroma is a vector database that stores the embeddings of data*

We use docIndex.py to 