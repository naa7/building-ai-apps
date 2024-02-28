from transformers import pipeline

generator = pipeline('text-generation', model="gpt2")

result = generator("Hello, I am", truncation=True, max_length=30, num_return_sequences=5)

print(result)