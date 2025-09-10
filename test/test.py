from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:5500", api_key="__UNKNOWN__")

result = client.images.generate(
    model="stabilityai/stable-diffusion-3.5-medium",
    prompt="a white siamese cat",
    size="1024x1024"
)

print(result.data[0].url)