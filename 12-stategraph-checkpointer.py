from anyio.lowlevel import checkpoint
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from langchain_core.runnables import RunnableConfig
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

# 表达状态
class State(TypedDict):
    foo : str
    bar : Annotated[list[str], add]

def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}

def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}

# 定义状态图
workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

# 检查点管理器
checkpointer = InMemorySaver()

# 图编译
graph = workflow.compile(checkpointer)

# 配置
config : RunnableConfig = {
    "configurable": {"thread_id": "1"}
}

# 调用
results = graph.invoke({"foo": ""}, config)
print(results)

# 状态查看
# print(graph.get_state(config))
# StateSnapshot(
#   values={'foo': 'b', 'bar': ['a', 'b']},
#   next=(), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f111f79-e7f3-64e8-8002-7a0d0dee8aae'}}, metadata={'source': 'loop', 'step': 2, 'parents': {}}, created_at='2026-02-25T03:11:03.892705+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f111f79-e7f2-6af2-8001-0aab4e8fbd1f'}}, tasks=(), interrupts=())

for checkpointer_tuple in checkpointer.list(config):
    print(checkpointer_tuple)
# CheckpointTuple(
#   config={
#       'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f111f7d-36f1-68f8-bfff-aeedb1124841'}},
#       checkpoint={'v': 4, 'ts': '2026-02-25T03:12:32.706377+00:00', 'id': '1f111f7d-36f1-68f8-bfff-aeedb1124841',
#           'channel_versions': {'__start__': '00000000000000000000000000000001.0.49715785536142754'}, 'versions_seen': {'__input__': {}}, 'updated_channels': ['__start__'],
#           'channel_values': {'__start__': {'foo': ''}}},
#       metadata={'source': 'input', 'step': -1, 'parents': {}}, parent_config=None, pending_writes=[('f7135823-40ec-e8fc-830a-8e724438f3fe', 'foo', ''), ('f7135823-40ec-e8fc-830a-8e724438f3fe', 'branch:to:node_a', None)])
