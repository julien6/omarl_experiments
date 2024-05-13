from transformers import pipeline, AutoTokenizer
import torch

torch.manual_seed(0)
model = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

torch.manual_seed(4)
prompt = """Answer the question using the context below.
Context: Moving Company is a two-dimensional grid game representing two towers as a few vertical air cells separated by a few horizontal air cells at the bottom. Indeed air cells look like a big U. There are drop zones in the corner of the U. Mover employees have to bring a package from the cell located at the top of the first tower (the left one) to a cell located at the top of the second tower (the right one). So, the package must be brought down from the first tower's top by the first agent (so from top to bottom), then a second agent must bring it from the first tower's bottom the second tower's bottom (so from left to right), finally a third agent must bring it from the second tower's bottom to the top (so from bottom to top). In order to do that are free to move up, left, down, and right in the white cells. They can pick up or drop down the package in the drop zone but they can not be on the drop zones. The white cells are air and the grey cells represent walls. The game ends when the package is dropped in the final cell. The environment is fully discrete, vectorized. Agents' observations are the 3x3 grid cells surrounding an agent. 'Agent_0' takes the package, then goes down, then goes down, then drop off the package. 'Agent_1' takes the package, then goes right, then goes right, then drops off the package. 'Agent_2' takes the package, then goes up, then goes up, then drops off the package.
Question: What are the three keywords that better describe the roles respectively played by 'Agent_0', 'Agent_1', and 'Agent_2'?
Agents roles:
"""

sequences = pipe(
    prompt,
    max_new_tokens=100,
    do_sample=True,
    top_k=10,
    return_full_text = False,
)

for seq in sequences:
    print(f"Result: {seq['generated_text']}")