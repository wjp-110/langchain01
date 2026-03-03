from anyio.lowlevel import checkpoint
from langchain.agents import create_agent
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()

DB_URL = ""
with PostgresSaver.from_conn_string(DB_URL) as checkpointer:
    checkpointer.setup()

    agent = create_agent(
        model="deepseek:deepseek-chat",
        checkpointer=checkpointer
    )

    config = {"configurable": {"thread_id": "1"}}


    ## 第一轮消息
    results = agent.invoke(
        {"messages": [{"role": "user", "content": "来一首宋词"}] },
        config=config
    )

    messages = results["messages"]

    print(f"历史消息: {len(messages)}")

    for message in messages:
        message.pretty_print()

    ## 第二轮消息

    results = agent.invoke(
        {"messages": [{"role": "user", "content": "再来"}]},
        config=config
    )

    messages = results["messages"]

    print(f"历史消息: {len(messages)}")

    for message in messages:
        message.pretty_print()

