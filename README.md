# Local Data Retriever
## What is it?
Utilizes OpenAI, LangChain and Chroma to achieve a search engine based on imported local data.

## Operating Guide
### 1. Install the required sdk and packages
- Python: ≥ 3.9
- LangChain - *a great tool that helps LLMs function better
    
        pip3 install langchain
- Chroma - *a vector database that stores the embeddings of data*

        pip3 install chromadb
- Other requirements

        pip3 install chromadb openai unstructured tiktoken
### 2. Prepare local data
- put data into a directory
- supported formats: *csv, doc, docx, epub, html, md, pdf, ppt, pptx, txt, etc.*
### 3.  Configure the environment parameters
Create the "config.env" file. Several things need to be specified:
- PERSIST_DIRECTORY - *the place to put the database*
- EMBEDDINGS_MODEL_NAME - *can either be a model from hugging face or openai*
- TARGET_SOURCE_CHUNKS - *the number of chunks that'd be chosen as refrences to the answer*
- SOURCE_DIRECTORY - *the source of local data*
- OPENAI_API_KEY
We use docIndex.py to 
### 4. Vectorize the data and store in database
    python3 docRetriever.py
- The **main()** in the **docVectorize.py** slices the data and converts it to vectors
- It could take several minutes if the amount of data is huge
### 5. Run the retriever in the form of Q&A
    python3 main.py
Use **exit** to terminate the program
    
    exit

[Reference](https://github.com/zhaoqingpu/LangChainTest)