from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from prompts import *
from states import *
from langgraph.constants import END
from langgraph.graph import StateGraph

load_dotenv()

OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=OPEN_AI_API_KEY)


def planner_agent(state: dict) -> dict:
    user_prompt = state["user_prompt"]
    response = llm.with_structured_output(Plan).invoke(planner_prompt(user_prompt))
    return {"plan": response}


graph = StateGraph(dict)
graph.add_node("planner", planner_agent)
graph.set_entry_point("planner")

agent = graph.compile()

user_prompt = "Create a simple calculator web app"

result = agent.invoke({"user_prompt": user_prompt})

print(result)