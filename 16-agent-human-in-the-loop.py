from dataclasses import dataclass
from pydoc import describe

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langchain.agents.middleware import human_in_the_loop, HumanInTheLoopMiddleware
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from dotenv import load_dotenv

load_dotenv()

# Define system prompt
SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location.
用中文回答
"""


# Define context schema
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str


# Define tools
@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"{city}的天气很棒！阳光明媚！"


@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "旧金山"


# Configure model - 使用DeepSeek
model = init_chat_model(
    "deepseek-chat",
    model_provider="deepseek",
    temperature=0
)


# Define response format
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None


# Set up memory
checkpointer = InMemorySaver()

# Create agent
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "get_user_location": True,
                "get_weather_for_location": {
                    "allowed_decisions": ["approve", "reject"]
                }
            },
            description_prefix="等待决策，挂起"
        )
    ],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)

# Run agent
# `thread_id` is a unique identifier for a given conversation.
config = {"configurable": {"thread_id": "1"}}

try:
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "外面天气怎么样？"}]},
        config=config,
        context=Context(user_id="1")
    )

    messages = response["messages"]
    print(f"历史消息：{len(messages)}")
    for message in messages:
        if hasattr(message, 'pretty_print'):
            message.pretty_print()
        else:
            print(f"{message.role}: {message.content}")

    if "__interrupt__" in response:
        print("INTERRUPT")
        interrupt = response["__interrupt__"][0]
        for command in interrupt.value["action_requests"]:
            print(command["description"])

    # 指令
    response = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {"type": "approve"}
                ]
            }
        ),
        config=config,
        context=Context(user_id="1")
    )

    # Note that we can continue the conversation using the same `thread_id`.
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "谢谢！"}]},
        config=config,
        context=Context(user_id="1")
    )

    if "__interrupt__" in response:
        print("INTERRUPT")
        interrupt = response["__interrupt__"][0]
        for command in interrupt.value["action_requests"]:
            print(command["description"])

    # 指令
    response = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {"type": "approve"}
                ]
            }
        ),
        config=config,
        context=Context(user_id="1")
    )

    messages = response["messages"]
    print(f"历史消息：{len(messages)}")
    for message in messages:
        if hasattr(message, 'pretty_print'):
            message.pretty_print()
        else:
            print(f"{message.role}: {message.content}")

except Exception as e:
    print(f"发生错误: {e}")
    import traceback

    traceback.print_exc()
