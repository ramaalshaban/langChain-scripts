from dotenv import load_dotenv, dotenv_values
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


load_dotenv()
API_KEY = dotenv_values(".env").get("OPENAI_API_KEY")
model = ChatOpenAI(api_key=API_KEY, temperature=0.1, model="gpt-3.5-turbo")
# sessoinIds tutuyoruz. belki bir gun databasete yada farkli yerlerde kullanabiliriz

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helpful assistant. Answer all questions to the best of your ability"),
        MessagesPlaceholder(variable_name="messages")
    ]
)

chain = prompt | model
config = {"configurable": {"session_id": "session id"}}
with_message_history = RunnableWithMessageHistory(chain, get_session_history)
# chain ve hangi fojksyonla session id alabiliyom




if __name__ == "__main__":
    while True:
        user_input = input("> ")
        for r in with_message_history.stream(
            [
                HumanMessage(content=user_input)
            ],
            config=config,
        ):
            print(r.content, end=" ")

