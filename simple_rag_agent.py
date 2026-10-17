import os
from PIL import Image

from smolagents import load_tool, CodeAgent, InferenceClientModel, DuckDuckGoSearchTool

from huggingface_hub import login
from dotenv import load_dotenv



load_dotenv()
hf_token = os.getenv("HF_TOKEN")


login(token=hf_token)

image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)

search_tool = DuckDuckGoSearchTool()

model = InferenceClientModel("deepseek-ai/DeepSeek-V4.1-Flash")

agent = CodeAgent(tools=[image_generation_tool, search_tool], model=model)

result = agent.run(
    "Generate me a photo of the car that James bond drove in the latest movie.",
)
result