# from langchain_deepseek import ChatDeepSeek

from dotenv import load_dotenv
load_dotenv()

# model = ChatDeepSeek (
#     model="deepseek-chat",
#     temperature=0.1,
#     max_tokens=2000,
#     timeout=None,
#     max_retries=2
# )


from langchain.chat_models import init_chat_model
model = init_chat_model (
    model="deepseek:deepseek-chat", 
    temperature=0.1, # q: biao示模型生成内容时，生成的内容越长，越相似
    max_tokens=2000,
    timeout=None,
    max_retries=2
)

for chunk in model.stream("来段宋词吧"):
    print(chunk.content, end="", flush=True)