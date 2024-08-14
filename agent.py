from dotenv import load_dotenv, dotenv_values
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

#SqliteSaver ile basit bir hafiza hazirlayabiliriz

load_dotenv()

search = TavilySearchResults(max_results=2)

model = ChatOpenAI(model="gpt-4")

memory = SqliteSaver.from_conn_string(":memory")
# Cok araclar var! search about it : D
tools = [search]

# model_with_tool = model.bind_tools(tools)

agent_executor = create_react_agent(model, tools, checkpointer=memory)


config = {"configurable": {"thread_id" : "some_idhe"}}

if __name__ == '__main__':
    while True:
        user_input = input(">")
        for chunk in agent_executor.stream(
                {"messages": [HumanMessage(content=user_input)]}, config
        ):
            print(chunk)
            print("----")


    # search_result = agent_executor.invoke(
    #     {"messages": [HumanMessage(content="what is the current weather in istanbul now?")]}
    #
    # )
    # for r in search_result['messages']:
    #     print(r.content)
    # print(search_result.tool_calls)

# Here the response is coming back empty means using tools directly with open ai is not the best idea and actually we need the Agents.
# katakulle
