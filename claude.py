
import anthropic
import base64

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

with open("receipt1.jpg", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

questions = [
    "What is the vendor name?",
    "What is the total amount?",
    "What is the date?",
    "What items were purchased?",
    "What is the amount due?",
]

for q in questions:
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": q},
                ],
            }
        ],
    )
    print(f"\nQ: {q}")
    print(f"A: {message.content[0].text}")
