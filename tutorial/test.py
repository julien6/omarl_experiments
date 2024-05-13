from transformers import AutoTokenizer, AutoModelForCausalLM

# Charger le tokenizer et le modèle pré-entraîné
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Définir le texte d'entrée
input_text = "Once upon a time, "

# Tokeniser le texte d'entrée
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Générer du texte avec le modèle
output = model.generate(input_ids, max_length=100, num_return_sequences=5, temperature=0.7)

# Décoder et afficher les séquences générées
for i, sample_output in enumerate(output):
    print(f"Generated sequence {i+1}: {tokenizer.decode(sample_output, skip_special_tokens=True)}\n")
