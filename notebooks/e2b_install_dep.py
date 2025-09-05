
from e2b_code_interpreter import Sandbox

from openevals.code.e2b.execution import create_e2b_execution_evaluator
#export E2B_API_KEY="e2b_xxxxxxxxxxxxxxxxxxxxx"

# Your template ID from the previous step
#template_id = 'k0wmnzir0zuzye6dndlw' 
# Pass the template ID to the `Sandbox.create` method
#sbx = Sandbox.create(template_id = template_id) 
sbx = Sandbox.create()

sbx.commands.run("python -m pip install --upgrade pip", timeout=0)

# Install torch with no timeout
sbx.commands.run("pip install torch --index-url https://download.pytorch.org/whl/cpu", timeout=0)

evaluator = create_e2b_execution_evaluator(sandbox=sbx)

CODE = """
import torch
import torch.nn as nn

# Define the generated model (bad style originally, fixed as a proper class)
class GeneratedModel(nn.Module):
    def __init__(self):
        super(GeneratedModel, self).__init__()
        self.hidden1 = nn.Linear(784, 100)
        self.hidden2 = nn.Linear(100, 100)
        self.output_layer = nn.Linear(100, 10)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = self.output_layer(x)
        return x

# Define the reference model (already good, just moved outside function)
class ReferenceModel(nn.Module):
    def __init__(self):
        super(ReferenceModel, self).__init__()
        self.hidden1 = nn.Linear(784, 100)
        self.hidden2 = nn.Linear(100, 100)
        self.output_layer = nn.Linear(100, 10)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = self.output_layer(x)
        return x

# Instantiate both models
generated_model = GeneratedModel()
reference_model = ReferenceModel()

# Run test inputs
toy_input = torch.randn(1, 784)

print("Generated Model Output:")
print(generated_model(toy_input).tolist())

print("Reference Model Output:")
print(reference_model(toy_input).tolist())


"""

eval_result = evaluator(outputs=CODE)
print(eval_result)

# Direct execution in sandbox (outside evaluator)
run_result = sbx.run_code(CODE)

print("=== STDOUT ===")
print(run_result)

print(run_result.__dict__)  # Or vars(result) — shows what's available


