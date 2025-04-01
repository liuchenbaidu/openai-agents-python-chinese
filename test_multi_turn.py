from openai import AsyncOpenAI
from agents import set_default_openai_client
from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings

import os
import logging
import configparser
import sys

# Disable OpenAI telemetry/tracing completely
os.environ["OPENAI_TELEMETRY"] = "false"
os.environ["OPENAI_TRACING_ENABLED"] = "false"
os.environ["OPENAI_LOG_LEVEL"] = "ERROR"  # Only show ERROR level logs

# Disable warning outputs
logging.getLogger("openai").setLevel(logging.ERROR)

from agents import set_tracing_disabled

set_tracing_disabled(True)

# Read configuration from config.ini
config = configparser.ConfigParser()

# Try different config file paths (current directory, script directory, absolute path)
config_paths = [
    'config.ini',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini'),
    'openai-agents-python-chinese/config.ini'
]

config_found = False
for config_path in config_paths:
    if os.path.exists(config_path):
        config.read(config_path)
        config_found = True
        print(f"Using configuration from: {config_path}")
        break

if not config_found:
    print("Error: Configuration file 'config.ini' not found.")
    print("Please create a config.ini file with [api] section containing base_url and api_key.")
    sys.exit(1)

try:
    # Create client using configuration values
    external_client = AsyncOpenAI(
        base_url=config['api']['base_url'],
        api_key=config['api']['api_key'],
        default_headers={"x-stainless-telemetry": "false"}
    )
except KeyError as e:
    print(f"Error: Missing configuration key: {e}")
    print("Please ensure config.ini contains [api] section with base_url and api_key.")
    sys.exit(1)

# set_default_openai_client(external_client)

# external_client = AsyncOpenAI(
#     api_key="sk-f1d7770d8f5249049a8c0f0754e27e32",
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     # Additional parameter to disable telemetry
#     default_headers={"x-stainless-telemetry": "false"}
# )

import asyncio

from agents import Agent, Runner, function_tool


@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."

@function_tool
# 评测皮肤
def evaluate_skin(skin: str) -> str:
    """调用皮肤检测模型，返回皮肤状态"""
    print(f"调用皮肤检测模型，返回皮肤状态: {skin}")
    return f"The skin is {skin}."
agent = Agent(
    name="Hello world",
    instructions="You are a helpful agent.",
    model=OpenAIChatCompletionsModel(
        model="qwen-max",
        openai_client=external_client,
    ),
    model_settings=ModelSettings(temperature=0.5),
    tools=[get_weather, evaluate_skin],
)


async def main():
    # Initial user query
    user_input = "天气如何？"
    print(f"User: {user_input}")
    
    # Start conversation
    result = await Runner.run(agent, input=user_input)
    print(f"Agent: {result.final_output}")
    
    # The agent remembers context automatically in subsequent runs
    # Continue conversation for multiple turns
    while True:
        # Get next user input
        user_input = input("\nUser (type 'exit' to end): ")
        if user_input.lower() == 'exit':
            break
            
        # Continue the conversation with only the latest user input
        # The agent should maintain context internally
        result = await Runner.run(
            agent,
            input=user_input
        )
        print(f"Agent: {result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())