Installation:

- with conda
1. conda env create -f environment.yaml
2. conda activate ollama_full_research
3. langgraph dev --no-reload (beeing in the project directory)



If you want to install a new module:

1. find the env path (use conda env list)
2. path/to/your/env/bin/python -m pip install ...
3. check if has been installed (conda list)