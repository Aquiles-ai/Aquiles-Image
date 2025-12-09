from openai import OpenAI
import base64

client = OpenAI(base_url="https://f4k3r22--aquiles-image-server-serve.modal.run", api_key="dummy-api-key")

prompt = "A vast futuristic city curving upward into the sky, its buildings bending and connecting overhead in a continuous loop. Gravity shifts seamlessly along the curve, with sunlight streaming across inverted skyscrapers. The scene feels serene and awe-inspiring—earthlike fields and rivers running along the inner surface of a colossal rotating structure."

result = client.images.generate(
    model="black-forest-labs/FLUX.1-Krea-dev",
    prompt=prompt,
    size="1024x1024",
    response_format="b64_json"
)

print(f"Downloading image\n")

image_bytes = base64.b64decode(result.data[0].b64_json)
with open("output.png", "wb") as f:
    f.write(image_bytes)

print(f"Image downloaded successfully\n")