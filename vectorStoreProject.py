from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI



"""
chatgpt ile bir llm kendi verdigimiz dokumanlar vektora cevirdik , similrity search yaptik istedigimiz cevap ne oldugunu anladi sonra bize verdi so when i ask a specific question it only returns an answer related to the doc

"""
load_dotenv()

# chroma is a vector database
# check out the documentation of chroma
documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

vectorStore = Chroma.from_documents(
    documents=documents,
    embedding=OpenAIEmbeddings()
)

retriever = RunnableLambda(vectorStore.similarity_search).bind(k=1)
print('here:', retriever)

llm = ChatOpenAI(model='gpt-3.5-turbo')

message = """
Answer the following question using the provided context only
{question}

Contex: {context}
"""

prompt = ChatPromptTemplate.from_messages([("human",message)])


chain = {"context" : retriever , "question": RunnablePassthrough()} | prompt | llm


if __name__ == "__main__":
    response = chain.invoke("tell me about cats")
    print(response.content)

    # embedding = OpenAIEmbeddings().embed_query("dog") # here i transform the query to vector like this
    # print(vectorStore.similarity_search_by_vector(embedding))

# RAG