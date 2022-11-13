import pandas as pd
import json

if __name__ == "__main__":
    # if allData.csv doesn't exist, create it
    try:
        uniqueAllData = pd.read_csv("allData.csv")
    except:
        dev = pd.read_csv('dev.csv')
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')
        
        allData = pd.concat([dev, train, test])
        uniqueAllData = allData.drop_duplicates(keep='last')
        
        # drop first column of uniqueAllData
        uniqueAllData = uniqueAllData.iloc[:, 1:]
        # drop acronym column
        uniqueAllData = uniqueAllData.drop('acronym', axis=1)
        # add a column of empty strings
        uniqueAllData['acronym_'] = ''
        
        # iterate over the rows of uniqueAllData
        for i in range(len(uniqueAllData)):
            # get the row
            row = uniqueAllData.iloc[i]
            listOfTokens = row['tokens'].replace('[', '').replace(']', '').replace('\'', '').split(', ')
            acronym  = listOfTokens[row['location']]
            uniqueAllData.iloc[i, 5] = acronym
            uniqueAllData.iloc[i, 4] = ' '.join(listOfTokens)

        # move acronym column to the front
        cols = uniqueAllData.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        uniqueAllData = uniqueAllData[cols]

        # drop unnamed column
        uniqueAllData = uniqueAllData.drop('Unnamed: 0', axis=1)
        # drop location column
        uniqueAllData = uniqueAllData.drop('location', axis=1)

        # rename columns
        uniqueAllData = uniqueAllData.rename(columns={"disambiguation": "expansion", "tokens": "text"})
        uniqueAllData.to_csv('allData.csv', index=False)
   
    # import dict.json and convert it to a dictionary
    dict = json.load(open('dict.json'))
    
    # save all entries to check if they don't exist in train/dev/ext data
    allEntries = dict.keys()
    # convert to list
    allEntries = list(allEntries)
    
    cleanAllData0 = uniqueAllData.copy()
    
    # If a word in the train/dev/test data doesn't exist in dict.json, remove it from the data
    for i in range(len(uniqueAllData)):
        if uniqueAllData.iloc[i, 0] not in allEntries:
            cleanAllData0 = cleanAllData0.drop(i)
            
    cleanAllData = cleanAllData0.copy()
    
    allDictValues = list(dict.values())
    # combine lists of values
    allDictValues = [item for sublist in allDictValues for item in sublist]
    numberOfValuesRemoved = 0
    # if an expansion is not in dict.json, remove it from the cleanAllData
    for i in range(len(cleanAllData0)):
        if cleanAllData0.iloc[i, 1] not in allDictValues:
            # remove row where the expansion is cleanAllData0.iloc[i, 1]
            cleanAllData = cleanAllData[cleanAllData.expansion != cleanAllData0.iloc[i, 1]]
            numberOfValuesRemoved = numberOfValuesRemoved + 1
    
    print("Removed " + str(numberOfValuesRemoved) + " values from cleanAllData, from the original " + str(len(cleanAllData0)) + " values.")

    # create test file with x percent of cleanAllData
    testData = cleanAllData.sample(frac = 0.05)
    cleanAllData = cleanAllData.drop(testData.index)
    # drop expansion column
    # testData = testData.drop('expansion', axis=1)
    
    # items that contain dev in the id, are dev items, and remove them from cleanAllData
    devData = cleanAllData[cleanAllData['id'].str.contains('DEV')]
    cleanAllData = cleanAllData.drop(devData.index)
    
    # create train file with x percent of cleanAllData
    externalData = cleanAllData.sample(frac = 0.1)
    cleanAllData = cleanAllData.drop(externalData.index)

    
    # items left are train items
    trainData = cleanAllData
    
    ###Name CSVs' ids appropriately
    for i in range(len(testData)):
        testData.iloc[i, 2] = 'TS-' + str(i)
        
    for i in range(len(devData)):
        devData.iloc[i, 2] = 'DEV-' + str(i)
              
    for i in range(len(externalData)):
        externalData.iloc[i, 2] = 'EXT'
        
    for i in range(len(trainData)):
        trainData.iloc[i, 2] = 'TR-' + str(i)

    testData.to_csv('test.csv', index=False)
    devData.to_csv('dev.csv', index=False)
    trainData.to_csv('train.csv', index=False)
    externalData.to_csv('external_data.csv', index=False)
