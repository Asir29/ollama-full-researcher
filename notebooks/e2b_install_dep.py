#export E2B_API_KEY="e2b_4590f83de28112a9bd68eb920f368104fff15706"


from e2b_code_interpreter import Sandbox
from openevals.code.e2b.execution import create_e2b_execution_evaluator

sbx = Sandbox()

# Install torch with no timeout
sbx.commands.run("pip install torch", timeout=0)

evaluator = create_e2b_execution_evaluator(sandbox=sbx)

CODE = """
import torch
x = torch.tensor([1.0, 2.0, 3.0])
print("Sum:", torch.sum(x))
"""

eval_result = evaluator(outputs=CODE)
print(eval_result)

# Direct execution in sandbox (outside evaluator)
run_result = sbx.run_code(CODE)

print("=== STDOUT ===")
print(run_result)

print(run_result.__dict__)  # Or vars(result) — shows what's available


