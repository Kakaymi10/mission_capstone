import ollama

res = ollama.chat(
    model="llava:7b",
    messages=[
        {
            'role': 'user',
            'content': 'Describe this image:',
            'images': ['./sr_image.png']
        }
    ]
)

print(res['message']['content'])
