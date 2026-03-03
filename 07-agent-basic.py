from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()


agent = create_agent(
    model="deepseek:deepseek-chat",
)

results = agent.invoke({"messages": [{"role": "user", "content": "waht is the weather in SF？"}] })

messages = results["messages"]

print(f"历史消息: {len(messages)}")

for message in messages:
    message.pretty_print()