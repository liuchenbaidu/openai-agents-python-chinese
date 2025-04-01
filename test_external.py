from openai import AsyncOpenAI
from agents import set_default_openai_client
from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings

import os
import logging

# Disable OpenAI telemetry/tracing completely
os.environ["OPENAI_TELEMETRY"] = "false"
os.environ["OPENAI_TRACING_ENABLED"] = "false"
os.environ["OPENAI_LOG_LEVEL"] = "ERROR"  # Only show ERROR level logs

# Disable warning outputs
logging.getLogger("openai").setLevel(logging.ERROR)

from agents import set_tracing_disabled
import configparser

config = configparser.ConfigParser()
config.read('project/openai-agents-python-chinese/config.ini')

set_tracing_disabled(True)

external_client = AsyncOpenAI(base_url=config['api']['base_url'],
                             api_key = config['api']['api_key'] , 
                             default_headers={"x-stainless-telemetry": "false"})
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


agent = Agent(
    name="Hello world",
    instructions="You are a helpful agent.",
    model=OpenAIChatCompletionsModel(
        model="qwen-max",
        openai_client=external_client,
    ),
    model_settings=ModelSettings(temperature=0.5),
    tools=[get_weather],
)


async def main():
    result = await Runner.run(agent, input="天气如何？")
    print(result.final_output)
    # The weather in Tokyo is sunny.


if __name__ == "__main__":
    asyncio.run(main())