"""
This endpoint is now working
"""
from openai import OpenAI
import requests

client = OpenAI(base_url="http://127.0.0.1:5500", api_key="__UNKNOWN__")

result = client.images.generate(
    model="stabilityai/stable-diffusion-3.5-medium",
    prompt="a white siamese cat",
    size="1024x1024"
)

print(f"URL of the generated image: {result.data[0].url}\n")

print(f"Downloading image\n")
image_url = result.data[0].url
response = requests.get(image_url)

with open("image.png", "wb") as f:
    f.write(response.content)

print(f"Image downloaded successfully\n")