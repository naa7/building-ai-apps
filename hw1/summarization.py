from transformers import pipeline

summarizer = pipeline('summarization', model="facebook/bart-large-cnn")
text = """The Apollo program, also known as Project Apollo, was the third 
        United States human spaceflight program carried out by the National 
        Aeronautics and Space Administration (NASA), which accomplished 
        landing the first humans on the Moon from 1969 to 1972. First 
        conceived during Dwight D. Eisenhower's administration as"""
result = summarizer(text, max_length=30, min_length=15, do_sample=False)
print(result)