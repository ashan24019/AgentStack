from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from prompts import *
from states import *
from langgraph.constants import END
from langgraph.graph import StateGraph
from langchain_core.globals import set_verbose, set_debug
from tools import *

load_dotenv()

set_verbose(True)
set_debug(True)

OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=OPEN_AI_API_KEY)


def planner_agent(state: dict) -> dict:
    user_prompt = state["user_prompt"]
    response = llm.with_structured_output(Plan).invoke(planner_prompt(user_prompt))

    if response is None:
        raise ValueError("Planner agent failed to generate a project plan.")
    
    return {"plan": response}


def architect_agent(state: dict) -> dict:
    plan: Plan = state["plan"]
    response = llm.with_structured_output(TaskPlan).invoke(architect_prompt(plan))

    if response is None:
        raise ValueError("Architect agent failed to generate a task plan.")
    
    return {"task_plan": response, "plan": plan}

def coder_agent(state: dict) -> dict:
    steps = state['task_plan'].implementation_steps
    current_steps_idx = 0
    current_task = steps[current_steps_idx]

    existing_content = read_file.run(current_task.filepath)

    user_prompt = (
        f"Task: {current_task.task_description}\n"
        f"File: {current_task.filepath}\n"
        f"Existing content:\n{existing_content}\n"
        "Use write_file(path, content) to save your changes."
    )

    system_promp = coder_system_prompt()

    coder_tools = [read_file, write_file, list_files, get_current_directory]
    react_agent = create_react_agent(llm, coder_tools)
    react_agent.invoke(
        {"messages": [
            {"role": "system", "content": system_promp},
            {"role": "user", "content": user_prompt}
        ]}
    )

    return {}


graph = StateGraph(dict)
graph.add_node("planner", planner_agent)
graph.add_node("architect", architect_agent)
graph.add_node("coder", coder_agent)
graph.add_edge("planner", "architect")
graph.add_edge("architect", "coder")
graph.set_entry_point("planner")

agent = graph.compile()


if __name__ == "__main__":

    user_prompt = "Create a simple calculator web app"

    result = agent.invoke({"user_prompt": user_prompt})

    print(result)