import bs4
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

"""
internetten veri cekecegiz
veriyi bolecegiz
embedding ve vectorize
store it in db
retriever olustur
bu r ile generation edip bilgi verecek
"""

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo")

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/#agent-system-overview",),
    bs_kwargs=dict(
        # bu kutup hane to get data from web and look into the important parts
        parse_only=bs4.SoupStrainer(
            # here a specify which parts i need to get from the page
            class_=("post-content", "post-title", "post-header")
        )
    )
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
# we can use any models vectoring sinifi

# datayi sakladigimiz zaman datadan bisey almak istiyorsak retriver kullaniriz

retriver = vectorstore.as_retriever()

#### RAG prompt and llm'ye vermek  ####
prompt = hub.pull("rlm/rag-prompt")


# hub, gitlab gibi bisey promptlar yazildi ve boyle ise biz birinin yazdegi bir prompt cekiyoruz

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
        {"context": retriver | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

# In this RAG workflow, the prompt is a crucial component that helps structure the input for the LLM. It ensures that the retrieved context (relevant information from the document embeddings) is effectively combined with the user's question. The prompt template guides how this combination should be presented to the LLM, ensuring a coherent and contextually rich response.
# Unlike a simple prompt where you directly ask the LLM a question, in RAG, the prompt is more complex and is used to integrate additional retrieved context before passing it to the LLM. This enhances the quality and relevance of the generated response.
# User's Question → Retrieve Relevant Information → Combine with Original Question Using a Prompt Template → Send to LLM → Generate Answer.

if __name__ == "__main__":
    for chunk in rag_chain.stream("explain me the concept behind the article as if i am 12 yo?"):
        print(chunk, end="", flush=True)
