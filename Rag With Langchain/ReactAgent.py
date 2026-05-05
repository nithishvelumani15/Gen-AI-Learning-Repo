import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
 
load_dotenv()
 
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.0
)
 
@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together. Use this when the user wants to add or sum numbers."""
    return a + b
@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together. Use this when the user wants to multiply numbers."""
    return a * b
@tool
def get_company_info(topic: str) -> str:
    """Get general information about the company.
    Use this when asked about company name, location, or founding year.
    Input should be a topic like 'name', 'location', or 'founded'."""
    info = {
        "name": "ABC Corporation",
        "location": "Chennai, Tamil Nadu",
        "founded": "2005"
    }
    return info.get(topic.lower(), "Information not available")
 
web_search = TavilySearch(
    max_results=3,
    topic="general",
    include_answer=True,
    search_depth="basic",
)
tools = [add_numbers, multiply_numbers, get_company_info,web_search]

agent = create_react_agent(
    model=llm,
    tools=tools,
)
 

result = agent.invoke({"messages": [{"role": "user", "content": "What is 15 multiplied by 7?"}]})
print(f"\nFull message trace:")
for msg in result["messages"]:
    print(f"{type(msg).__name__}: {msg.content}")
 

result = agent.invoke({"messages": [{"role": "user", "content": "What is the capital of India?"}]})
print(f"\nFull message trace:")
for msg in result["messages"]:
    print(f"{type(msg).__name__}: {msg.content}")

result = agent.invoke({"messages": [{"role": "user", "content": "What is the location of ABC Corporation"}]})
print(f"\nFull message trace:")
for msg in result["messages"]:
    print(f"{type(msg).__name__}: {msg.content}")
