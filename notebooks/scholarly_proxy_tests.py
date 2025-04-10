import time
import random
import asyncio
from scholarly import scholarly, ProxyGenerator
from fake_useragent import UserAgent
import os
import json


async def academic_research(state, config):
    """Perform academic research using scholarly and extract only the abstract with fake-useragent"""
    abstract = "No Abstract"
    random_user_agent = "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0"

    try:
        print("Loading user agents from browsers.jsonl...")

        fake_useragent_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'data', 'browsers.jsonl')
        )

        if os.path.exists(fake_useragent_path):
            print("File exists at:", fake_useragent_path)
        else:
            print("File does not exist at:", fake_useragent_path)

        user_agents = []
        try:
            with open(fake_useragent_path, 'r') as file:
                for line in file:
                    try:
                        user_agent = json.loads(line.strip())["useragent"]
                        user_agents.append(user_agent)
                    except json.JSONDecodeError:
                        print("Skipping invalid line in browsers.jsonl")
        except Exception as e:
            print(f"Error reading browsers.jsonl: {e}")
        else:
            if user_agents:
                random_user_agent = random.choice(user_agents)
            else:
                print("No valid user agents found in the file.")

        print(f"Using random user agent: {random_user_agent}")

        # Set up ProxyGenerator with the random user agent
        pg = ProxyGenerator()
        pg.SingleProxy(http=random_user_agent)
        scholarly.use_proxy(pg)

        # Perform the search
        search_query = state.search_query
        print(f"Searching for publications with query: {search_query}")
        search_results = scholarly.search_pubs(search_query)

        # Extract the abstract from the first result
        first_result = next(search_results, None)
        if first_result:
            abstract = first_result.get("bib", {}).get("abstract", "No Abstract Found")
            print(f"Abstract found: {abstract}")
        else:
            print("No search results found.")
        


    except Exception as e:
        print(f"General error in academic_research: {e}")
        return {"raw_search_result": {"abstract": abstract}}

    finally:
        print("Exiting academic_research function.")


# Example state and config
state = type('obj', (object,), {'search_query': 'Spike Neural Networks'})
config = {}

# Run the academic research function
result = asyncio.run(academic_research(state, config))
print(result)
