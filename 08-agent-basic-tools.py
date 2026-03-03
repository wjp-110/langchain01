from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

def get_weather(city: str) -> str:
    """get the weather given city"""
    return f"it is always sunny in {city}"


agent = create_agent(
    model="deepseek:deepseek-chat",
    tools=[get_weather]
)

results = agent.invoke({"messages": [{"role": "user", "content": "waht is the weather in SF？"}] })

messages = results["messages"]

print(f"历史消息: {len(messages)}")

for message in messages:
    message.pretty_print()