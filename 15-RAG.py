from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

agent = create_agent(
    model="deepseek:deepseek-chat"
)

results = agent.invoke({"messages": [{"role": "user", "content": "讲一下 3i/Atlas"}] })

messages = results["messages"]
for message in messages:
    message.pretty_print()