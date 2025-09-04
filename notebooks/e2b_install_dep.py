API_KEY="e2b_a913ba6f51c06f526eeef2de8b5b5a40f6372f9c"

from e2b_code_interpreter import Sandbox

from openevals.code.e2b.execution import create_e2b_execution_evaluator


# Your template ID from the previous step
template_id = 'k0wmnzir0zuzye6dndlw' 
# Pass the template ID to the `Sandbox.create` method
sbx = Sandbox(template_id) 
#sbx = Sandbox(api_key=API_KEY)

#sbx.commands.run("python -m pip install --upgrade pip", timeout=0)

# Install torch with no timeout
sbx.commands.run("pip install torch", timeout=0)


evaluator = create_e2b_execution_evaluator(sandbox=sbx)

CODE = """
import torch
import torch.nn as nn
import snntorch as snn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.lif1 = snn.Leaky(beta=0.95)  # just beta param, no spike_fn
        self.fc2 = nn.Linear(128, 10)
        self.lif2 = snn.Leaky(beta=0.95)

    def forward(self, x):
        x = self.fc1(x)
        x, _ = self.lif1(x)
        x = self.fc2(x)
        x, _ = self.lif2(x)
        return x

model = Net()
input_tensor = torch.rand((1, 784))
output_tensor = model(input_tensor)
print(output_tensor)


"""

eval_result = evaluator(outputs=CODE)
print(eval_result)

# Direct execution in sandbox (outside evaluator)
run_result = sbx.run_code(CODE)

print("=== STDOUT ===")
print(run_result)

print(run_result.__dict__)  # Or vars(result) — shows what's available


