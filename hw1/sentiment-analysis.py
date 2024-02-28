from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
# what is the difference between the two?
# The pipeline function is a high-level function that 
# will handle all the tokenization, model prediction, 
# and decoding for you. 

############################################
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
sequence = "I am excited about this course"

classifier = pipeline('sentiment-analysis', model=model_name)

result1 = classifier(sequence)

print(result1)

############################################
# AutoModelForSequenceClassification and AutoTokenizer 
# classes to explicitly load the model and its associated 
# tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

result2 = classifier(sequence)

print(result2)

############################################
result3 = tokenizer(sequence)
print(result3)

tokens = tokenizer.tokenize(sequence)
print(tokens)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(token_ids)
decoded_string = tokenizer.decode(token_ids)
print(decoded_string)

############################################
# This is what the pipeline function does under the hood
X_train = ["I am excited about this course", "I am not happy with this course"]
inputs = tokenizer(X_train, padding=True, truncation=True, return_tensors="pt")
print(inputs)

with torch.no_grad():
    outputs = model(**inputs)
    print(outputs)
    predictions = F.softmax(outputs.logits, dim=-1)
    print(predictions)
    labels = torch.argmax(predictions, dim=-1)
    print(labels)