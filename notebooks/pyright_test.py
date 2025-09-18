from openevals.code.pyright import create_pyright_evaluator

evaluator = create_pyright_evaluator()

CODE = """
import torch
import torch.nn as nn

# Generated model
generated_net = nn.Sequential(nn.Linear(3, 1)).double()
print("Generated model created:\\n", generated_net)


# Reference model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(3, 1).double()

    def forward(self, x):
        return self.fc(x)

reference_net = SimpleNet()

# Define the constant TOY_INPUT
TOY_INPUT = torch.tensor([2.0, 3.0, 4.0]).view(1, -1)

# Generate outputs for both models
generated_output = generated_net(TOY_INPUT)
reference_output = reference_net(TOY_INPUT)

print("\\nTOY_INPUT:")
print(TOY_INPUT)
print("\\nGenerated model output:")
print(generated_output)
print("\\nReference model output:")
print(reference_output)
"""

result = evaluator(outputs=CODE)

print(result)