#from scholarly import scholarly

#search_query = scholarly.search_pubs('deep learning')
#scholarly.pprint(next(search_query))


from semanticscholar import SemanticScholar

sch = SemanticScholar()
results = sch.search_paper("transformers in NLP", limit=1)

print("Search Results:")
for paper in results:
    print(paper.title, "-", paper.abstract or "No abstract")
