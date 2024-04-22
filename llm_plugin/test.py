# import sys
# from transformers import pipeline

# # Charger le modèle
# generator = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")

# prompt_command = sys.argv[1] if sys.argv[1] is not None else "What is a castle?"

# # Générer une réponse en utilisant le prompt
# response = generator(prompt_command, max_length=400)

# # Afficher la réponse générée
# print(response)

# print("="*30)
# print(response[0]['generated_text'])

from transformers import AutoTokenizer, OpenAIGPTModel
import torch

tokenizer = AutoTokenizer.from_pretrained("openai-community/openai-gpt")
model = OpenAIGPTModel.from_pretrained("openai-community/openai-gpt")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state

print(outputs)
