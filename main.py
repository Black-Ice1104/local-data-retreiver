import argparse
import os
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI, openai
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

load_dotenv("config.env")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
from constants import CHROMA_SETTINGS

# Chinese version of prompt
def prompt_chinese(retriever, llm):
    prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。尽可能发掘已知信息中有用的部分。
    若无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"。
    不在答案中添加编造成分。答案使用中文。
    已知:
    {context}
    问题:
    {question}"""

    promptA = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": promptA}
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff",
                                     chain_type_kwargs=chain_type_kwargs, return_source_documents=True)
    while True:
        query = input("\n请输入问题: ")
        if query == "exit":
            break
        res = qa(query)
        answer, docs = res['result'], res['source_documents']
        print("\n\n> 问题:")
        print(query)
        print("\n> 回答:")
        print(answer)
        print("来源:\n")
        for document in docs:
            print("\n> " + document.metadata["source"])


# English version of prompt
def prompt_english(retriever, llm):
    prompt_template = """Answer the questions in a concise and professional manner based on the following known information.
    If unable to obtain the answer, 
    Say "the question cannot be answered according to the known information" 
    or "not enough relevant information is provided". 
    Never add fabrications in the answer. Use the answer in English.
    What is known:
    {context}
    Question:
    {question}"""

    promptA = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": promptA}
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff",
                                     chain_type_kwargs=chain_type_kwargs, return_source_documents=True)
    while True:
        query = input("\nPlease enter the question: ")
        if query == "exit":
            break
        res = qa(query)
        answer, docs = res['result'], res['source_documents']
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)
        print("Source:\n")
        for document in docs:
            print("\n> " + document.metadata["source"])



if __name__ == '__main__':
    embeddings = OpenAIEmbeddings(model=embeddings_model_name, openai_api_key=OPENAI_API_KEY)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    llm = OpenAI(model_name="text-davinci-003", 
                max_tokens=200,
                n=3, 
                best_of=3)
    
    prompt_chinese(retriever, llm)
    # prompt_english(retriever, llm):
