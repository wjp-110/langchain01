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

# for event in agent.stream(
#     {"messages": [{"role": "user", "content": "waht is the weather in SF？"}] },
#     stream_mode="values" # message by message
# ):
#     print(event)
#     messages = event["messages"]
#     print(f"历史消息: {len(messages)}")
#
#     # for message in messages:
#     #     message.pretty_print()
#     messages[-1].pretty_print()

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "waht is the weather in SF？"}] },
    stream_mode="messages" # token by token
):
    print(chunk[0].content, end='')
    # print(chunk)



