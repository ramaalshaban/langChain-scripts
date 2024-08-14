from dotenv import load_dotenv, dotenv_values
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from langserve import add_routes


load_dotenv()


API_KEY = dotenv_values(".env").get("OPENAI_API_KEY")


model = ChatOpenAI(api_key=API_KEY, temperature=0.1, model="gpt-3.5-turbo")
# message =[HumanMessage(content="Translate the following from English to Arabic"),
#            SystemMessage(content="Fuck off")]

system_prompt = "Translate the following into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [
       ("system", system_prompt), ("user", "{text}")
    ]
)

parser = StrOutputParser()
chain = prompt_template | model | parser



app = FastAPI(
    title="Translator App! ",
    version="1.0.0",
    description="Translator App description! ",
)
add_routes(
    app,
    chain,
    path="/chain"
 )



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)