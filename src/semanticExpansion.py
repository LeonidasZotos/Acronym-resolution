import re
import sys
import requests
import urllib3

## This function sets up the queries, using the input data.
def create_query(X):
    query = '''
        ?''' + X + ''' schema:description ?itemdesc.
        FILTER(LANG(?itemdesc) = "en")
    '''
    # '''SELECT ?answerLabel WHERE {
    #     wd:''' + X + ''' wdt:''' + Y + ''' ?answer.
    #     SERVICE wikibase:label {
    #         bd:serviceParam wikibase:language "en" .
    #     }
    # }'''
    return query

## This function will execute the query and print the 'answer' the data of the query links to.
def run_query(query):
	url = 'https://query.wikidata.org/sparql'
     # TODO: it breaks here because of the query not being JSON
	data = requests.get(url, params={'query': query, 'format': 'json'}).json()
	final_results = []
	for item in data['results']['bindings']:
		final_results.append(item['answerLabel']['value'])
	return final_results

## This function finds the wd SPARQL code for a natural word. 
def find_entity(line):
    url = 'https://www.wikidata.org/w/api.php'
    params = {'action':'wbsearchentities',
          'limit':10,
          'language':'en',
          'format':'json'}
    params['search'] = line.rstrip()
    json = requests.get(url,params).json() 
    answerList = []
    for result in json['search']:
        answerList.append(result['id'])
    return answerList                           

def expandSemantically(acronym):
    expansion = "TEMP EXPANSION" # Placeholder added by Leo to make things work for now
    entity =  find_entity(acronym)
    query = create_query(entity[0])
    result = run_query(query)
    
    # TODO: Add result to text function
    print(result)
    # return text
    
    return expansion