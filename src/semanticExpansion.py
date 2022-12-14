import re
import sys
import requests
import urllib3

## This function sets up the queries, using the input data. 
# it is a simple query that checks whether an object has a certain property
def create_query(X, Y):
    query = '''
    SELECT ?answerLabel WHERE {
        wd:''' + X + ''' wdt:''' + Y + ''' ?answer.
        SERVICE wikibase:label {
            bd:serviceParam wikibase:language "en" .
        }
    }'''
    return query

## This function will execute the query and print the 'answer' the data of the query links to.
def run_query(query):
	url = 'https://query.wikidata.org/sparql'
	data = requests.get(url, params={'query': query, 'format': 'json'}).json()
	final_results = []
	for item in data['results']['bindings']:
		final_results.append(item['answerLabel']['value'])
	return final_results

## This function finds the wd SPARQL code for a natural word. 
def find_entity(line):
    url = 'https://www.wikidata.org/w/api.php'
    params = {'action':'wbsearchentities',
          'limit':1,
          'language':'en',
          'format':'json'}
    params['search'] = line.rstrip()
    json = requests.get(url,params).json() 
    answerList = []
    for result in json['search']:
        answerList.append(result['id'])
    return answerList          

## This function was created and used to find what properties fit 
# the natural language prompts of what property we wanted to query for
# It is not used while testing the program
def find_property(line):
    url = 'https://www.wikidata.org/w/api.php'
    params = {'action':'wbsearchentities',
          'limit':10,
          'type':'property',
          'language':'en',
          'format':'json'}
    params['search'] = line.rstrip()
    json = requests.get(url,params).json() 
    answerList = []
    for result in json['search']:
        answerList.append(result['id'])
    return answerList

## This function scrapes wikidata for the description of the entity, or the name of the property
def scrape_for_information(input, url, htmlstart, htmlend):
  custom_user_agent = "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:47.0) Gecko/20100101 Firefox/47.0"
  input_url = url + input
  req=requests.get(input_url, headers={'User-Agent': custom_user_agent})
  soup=req.text
  output = re.findall(htmlstart + '.*?' + htmlend, str(soup))
  output = re.sub(htmlstart, '', str(output))
  output = re.sub(htmlend, '', str(output))

  return output

## This is the main function of this part of the pipeline. 
#  It takes an expanded acronym, checks whether there is a description, or properties.
#  It will then return this additional information which will be placed in the sentence.        
def expandSemantically(acronym):
    properties = {'P31' : 'is an instance of ', 'P361': 'is a part of ', 'P366': 'has use ', 'P1889': 'is different from '}
    results = []
    entities =  find_entity(acronym)
    wasExpanded = False
    if entities:
        wasExpanded = True
        description = scrape_for_information(entities[0], 'https://www.wikidata.org/wiki/', '<div class="wikibase-entitytermsview-heading-description ">', '<\/div>')
        # Checks for thee 4 properties whether the input word has these properties
        for q_property in properties:
            query = create_query(entities[0], q_property)
            result = run_query(query)
            if result:
                results.append(properties[q_property] + str(result[0]))
        if results:
            expansion = str(description) + ', ' + acronym + ' has the following properties: ' + str(results)
        else:
            expansion = str(description)
    else:
        expansion = "No additional information found for " + acronym

    return str(expansion)