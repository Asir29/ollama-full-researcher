from scholarly import scholarly

search_query = scholarly.search_pubs('deep learning')
scholarly.pprint(next(search_query))



