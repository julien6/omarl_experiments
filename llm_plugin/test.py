from transformers import pipeline

# Charger le modèle
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")

prompt_command = "Could you write an historical overview of the Americas's colonization"

# Générer une réponse en utilisant le prompt
response = generator(prompt_command, max_length=400)

# Afficher la réponse générée
print(response)

print("="*30)
print(response[0]['generated_text'])
