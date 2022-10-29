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

## This function scrapes wikidata for the description of the entity
def scrape_for_description(query):

    url='https://www.wikidata.org/wiki/' + query
    req=requests.get(url)
    content=req.text
    description = re.findall('<div class="wikibase-entitytermsview-heading-description ">.*?<\/div>', str(content))
    description = re.sub('<div class="wikibase-entitytermsview-heading-description ">', '', str(description)) 
    description = re.sub('<\/div>', '', str(description))

    return description                 

def expandSemantically(acronym):
    
    entity =  find_entity(acronym)
    description = scrape_for_description(entity[0])
    # query = create_query(entity[0])
    # result = run_query(query)
    
    # # return text
    expansion = str(description) # Placeholder added by Leo to make things work for now
    
    return expansion