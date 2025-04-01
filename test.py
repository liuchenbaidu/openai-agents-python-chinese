from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings,set_default_openai_client
from openai import AsyncOpenAI
import os
import logging

# Disable OpenAI telemetry/tracing completely
os.environ["OPENAI_TELEMETRY"] = "false"
os.environ["OPENAI_TRACING_ENABLED"] = "false"
os.environ["OPENAI_LOG_LEVEL"] = "ERROR"  # Only show ERROR level logs

# Disable warning outputs
logging.getLogger("openai").setLevel(logging.ERROR)
from openai import AsyncOpenAI
import configparser 
# 读取配置文件     
config = configparser.ConfigParser() 
config.read('project/openai-agents-python-chinese/config.ini')


external_client = AsyncOpenAI(
    api_key=config['api']['api_key'],
    base_url=config['api']['base_url'],
    # Additional parameter to disable telemetry
    default_headers={"x-stainless-telemetry": "false"}
)

from agents.tool import function_tool

@function_tool
def search_web(query: str) -> str:
    """
    使用给定查询在网络上搜索信息。

    Args:
        query: 要搜索的查询字符串

    Returns:
        搜索结果的摘要
    """
    # 添加调试输出
    print(f"*** andy #### search_web 函数被调用！查询: '{query}' ***")

    
    # 实际搜索逻辑
    return f"关于'{query}'的搜索结果：这是一些有关{query}的信息，包括最新的AI模型、应用和研究突破。"


agent = Agent(
    name="Assistant", 
    instructions="""你是一个搜索助手。当用户有问题时，你应该始终使用search_web工具来查找信息。
不要自己编造答案，而是依赖搜索结果。使用search_web工具。
步骤：
1. 理解用户的查询
2. 使用search_web工具搜索相关信息
3. 基于搜索结果回答""",
    model=OpenAIChatCompletionsModel(
        model="qwen-max",
        openai_client=external_client,
    ),
    model_settings=ModelSettings(temperature=0.5),
    tools=[search_web],  # Add the search_web tool
)

# Debug check to verify tools are available
print(f"Agent tools: {agent.tools}")

# 用户查询
user_query = "搜索关于人工智能的最新发展"
print(f"用户查询: {user_query}")

# 尝试正常运行
print("尝试正常运行...")
result = Runner.run_sync(agent, user_query)
print(f"模型返回: {result.final_output}")



# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.