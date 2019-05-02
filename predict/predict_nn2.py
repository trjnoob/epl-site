import numpy as np
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import random

#print("Enter start year(YY): ")
#YEAR_START = int(input())
#print("Enter end year(YY): ")
#YEAR_END = int(input())

#YEAR_END = 18
#YEAR_START = YEAR_END - 2

def getTeamList(YEAR_END):
    YEAR_START = YEAR_END - 2
    infileList = []
    for i in range(YEAR_START, YEAR_END):
        #infileName = "epl_" + f"{i:02d}" + "" + f"{i+1:02d}" + "_data.json"
        infileName = "epl_" + str(i) + "" + str(i+1) + "_data.json
        infileList.append(infileName)

    #Only the latest season data is loaded for now 
    with open(infileList[1], "r") as infile:
        data = json.load(infile)

    teamList = []
    examples = []

    for datarow in data:
        examples.append(datarow)
        if datarow["HomeTeam"] not in teamList:
            teamList.append(datarow["HomeTeam"])
    return teamList

def printTeamList(YEAR_END):
    teamList = getTeamList(YEAR_END)
    print('\nTeams:\n')   
    for teamName in teamList:
        print(' ' + str(teamList.index(teamName)) + '. ' + teamName)
    

def predict_match(YEAR_END, hteam, ateam):
    YEAR_START = YEAR_END - 2
    infileList = []
    for i in range(YEAR_START, YEAR_END):
        #infileName = "epl_" + f"{i:02d}" + "" + f"{i+1:02d}" + "_data.json"
        infileName = "epl_" + str(i) + "" + str(i+1) + "_data.json
        infileList.append(infileName)

    nnData = []

    #Only the latest season data is loaded for now 
    with open(infileList[1], "r") as infile:
        data = json.load(infile)

    teamList = []
    examples = []

    for datarow in data:
        examples.append(datarow)
        if datarow["HomeTeam"] not in teamList:
            teamList.append(datarow["HomeTeam"])

    #print('\nTeams:\n')
    '''
    for teamName in teamList:
        print(' ' + str(teamList.index(teamName)) + '. ' + teamName)
    '''
    #print('\nPlease choose Home Team:')
    #hteam = int(input())
    #print('\nPlease choose Away Team:')
    #ateam = int(input())
    #print('\nEnter match date(YYYY-MM-DD):')
    date = '2019-06-06'

    #print('\n' + teamList[hteam] + ' vs ' + teamList[ateam])
    htName = teamList[hteam]
    atName = teamList[ateam]    

    tempRow = {"AC": 0, "AF": 0, "AR": 0, "AS": 0, "AST": 0, "AY": 0, "AwayTeam": atName,
                "Date": date, "FTAG": 0, "FTHG": 0,
                "FTR": "D", "HC": 0, "HF": 0, "HR": 0, "HS": 0, "HST": 0, "HTAG": 0, "HTHG": 0,
                "HTR": "D", "HY": 0, "HomeTeam": htName, "Referee": "Nobody"}
    examples.append(tempRow)
    data.append(tempRow)

    teamResults = defaultdict(list)

    for teamName in teamList:
        #print("17/18 " + teamName + "\n")
        for datarow in data:
            if (teamName == datarow["HomeTeam"]) or (teamName == datarow["AwayTeam"]):
                teamResults[teamName].append(datarow)
                #Track the match sequence number for the specific team in the season
                teamResults[teamName][-1]['MatchNum'] = len(teamResults[teamName])

        #print(teamResults[teamName])
        #print("\n")

    #print("Teams Game-by-Game Stats:\n")
    teamStats = {}
    for teamName in teamList:
        #print(teamName + " Game-by-Game Stats:\n")
        teamStats[teamName] = []
        for match in teamResults[teamName]:
            matchDict = {}
            matchNum = match['MatchNum']
            matchDict['MatchNum'] = matchNum
            if match["HomeTeam"] == teamName:
                matchDict['HomeOrAway'] = 'Home'
                matchDict['Opponent'] = match['AwayTeam']
                if match['FTR'] == 'H':
                    matchDict['Result'] = 'Win'
                elif match['FTR'] == 'A':
                    matchDict['Result'] = 'Lose'
                else:
                    matchDict['Result'] = 'Draw'
                matchDict['CurrentGF'] = match['FTHG']
                matchDict['CurrentGA'] = match['FTAG']
            else:
                matchDict['HomeOrAway'] = 'Away'
                matchDict['Opponent'] = match['HomeTeam']
                if match['FTR'] == 'H':
                    matchDict['Result'] = 'Lose'
                elif match['FTR'] == 'A':
                    matchDict['Result'] = 'Win'
                else:
                    matchDict['Result'] = 'Draw'
                matchDict['CurrentGF'] = match['FTAG']
                matchDict['CurrentGA'] = match['FTHG']
            matchDict['CurrentGD'] = matchDict['CurrentGF'] - matchDict['CurrentGA']
            if matchDict['Result'] == 'Win':
                matchDict['CurrentPoints'] = 3
            elif matchDict['Result'] == 'Draw':
                matchDict['CurrentPoints'] = 1
            else:
                matchDict['CurrentPoints'] = 0
            matchDict['PreMatchGF'] = 0
            matchDict['PreMatchGA'] = 0
            matchDict['PreMatchGD'] = 0
            matchDict['PreMatchPoints'] = 0
            if matchNum != 1:
                matchDict['PreMatchGF'] = teamStats[teamName][-1]['CurrentGF']
                matchDict['PreMatchGA'] = teamStats[teamName][-1]['CurrentGA']
                matchDict['PreMatchGD'] = teamStats[teamName][-1]['CurrentGD']
                matchDict['PreMatchPoints'] = teamStats[teamName][-1]['CurrentPoints']
                matchDict['CurrentGF'] += teamStats[teamName][-1]['CurrentGF']
                matchDict['CurrentGA'] += teamStats[teamName][-1]['CurrentGA']
                matchDict['CurrentGD'] += teamStats[teamName][-1]['CurrentGD']
                matchDict['CurrentPoints'] += teamStats[teamName][-1]['CurrentPoints']
            matchDict['Past5AvgPoints'] = 0
            matchDict['Past5GD'] = 0
            matchDict['Past5AvgHome'] = 0
            matchDict['Past5AvgAway'] = 0
            
            tempMatchNum = tempPoints = 0
            for tempMatch in reversed(teamStats[teamName]):
                tempMatchNum += 1
                if tempMatch['Result'] == 'Win':
                    tempPoints += 3
                elif tempMatch['Result'] == 'Lose':
                    tempPoints += 0
                else:
                    tempPoints += 1
                if tempMatchNum >= 5:
                    break
            if matchNum != 1:
                matchDict['Past5AvgPoints'] = tempPoints / tempMatchNum
                matchDict['Past5GD'] = teamStats[teamName][-1]['CurrentGD']
            if matchNum > 6:
                matchDict['Past5GD'] -= teamStats[teamName][-6]['CurrentGD']
                
            tempMatchNum = tempPoints = 0
            for tempMatch in reversed(teamStats[teamName]):
                if tempMatch['HomeOrAway'] == 'Home':
                    tempMatchNum += 1
                    if tempMatch['Result'] == 'Win':
                        tempPoints += 3
                    elif tempMatch['Result'] == 'Lose':
                        tempPoints += 0
                    else:
                        tempPoints += 1
                    if tempMatchNum >= 5:
                        break
            if tempMatchNum != 0:
                matchDict['Past5AvgHome'] = tempPoints / tempMatchNum

            tempMatchNum = tempPoints = 0
            for tempMatch in reversed(teamStats[teamName]):
                if tempMatch['HomeOrAway'] == 'Away':
                    tempMatchNum += 1
                    if tempMatch['Result'] == 'Win':
                        tempPoints += 3
                    elif tempMatch['Result'] == 'Lose':
                        tempPoints += 0
                    else:
                        tempPoints += 1
                    if tempMatchNum >= 5:
                        break
            if tempMatchNum != 0:
                matchDict['Past5AvgAway'] = tempPoints / tempMatchNum        

            teamStats[teamName].append(matchDict) 
            #print(matchDict)
            #print("")
                    
    teamSeason = {}
    for teamName in teamList:
        newDict = {}
        newDict['TeamName'] = teamName
        teamSeason[teamName] = newDict

    #print(teamSeason)

    #Only the previous season data is loaded for now
    #Need to edit this file name to be variable later
    with open(infileList[0], "r") as infile:
        prevData = json.load(infile)

    prevTeamList = []

    for datarow in prevData:
        if datarow["HomeTeam"] not in prevTeamList:
            prevTeamList.append(datarow["HomeTeam"])

    prevTeamResults = defaultdict(list)

    for teamName in prevTeamList:
        #print("16/17 " + teamName + "\n")
        for datarow in prevData:
            if (teamName == datarow["HomeTeam"]) or (teamName == datarow["AwayTeam"]):
                prevTeamResults[teamName].append(datarow)
        #print(prevTeamResults[teamName])
        #print("\n")

    teamPreSeason = {}
    for teamName in prevTeamList:
        newDict = {}
        newDict['TeamName'] = teamName
        newDict['TotalHomeWin'] = 0
        newDict['TotalAwayWin'] = 0
        newDict['TotalHomeDraw'] = 0
        newDict['TotalAwayDraw'] = 0
        newDict['TotalHomeLoss'] = 0
        newDict['TotalAwayLoss'] = 0
        newDict['TotalPoints'] = 0
        newDict['GoalDifference'] = 0
        for datarow in prevTeamResults[teamName]:
            if (datarow['HomeTeam'] == teamName):
                newDict['GoalDifference'] += int(datarow['FTHG'])
                newDict['GoalDifference'] -= int(datarow['FTAG'])
                if (datarow['FTR'] == 'H'):
                    newDict['TotalHomeWin'] += 1
                    newDict['TotalPoints'] += 3
                if (datarow['FTR'] == 'D'):
                    newDict['TotalHomeDraw'] += 1
                    newDict['TotalPoints'] += 1
                if (datarow['FTR'] == 'A'):
                    newDict['TotalHomeLoss'] += 1
            else:
                newDict['GoalDifference'] += int(datarow['FTAG'])
                newDict['GoalDifference'] -= int(datarow['FTHG'])
                if (datarow['FTR'] == 'A'):
                    newDict['TotalAwayWin'] += 1
                    newDict['TotalPoints'] += 3
                if (datarow['FTR'] == 'D'):
                    newDict['TotalAwayDraw'] += 1
                    newDict['TotalPoints'] += 1
                if (datarow['FTR'] == 'H'):
                    newDict['TotalAwayLoss'] += 1
            
        teamPreSeason[teamName] = newDict

    '''
    print("                Team\tHW\tAW\tHD\tAD\tHL\tAL\tGD\tPoints")
    for teamName in prevTeamList:    
        print('{0: >20}'.format(teamPreSeason[teamName]['TeamName']), end = '\t')
        print(teamPreSeason[teamName]['TotalHomeWin'], end = '\t')
        print(teamPreSeason[teamName]['TotalAwayWin'], end = '\t')
        print(teamPreSeason[teamName]['TotalHomeDraw'], end = '\t')
        print(teamPreSeason[teamName]['TotalAwayDraw'], end = '\t')
        print(teamPreSeason[teamName]['TotalHomeLoss'], end = '\t')
        print(teamPreSeason[teamName]['TotalAwayLoss'], end = '\t')
        print(teamPreSeason[teamName]['GoalDifference'], end = '\t')
        print(teamPreSeason[teamName]['TotalPoints'])
    '''


    #np.random.seed(1)
    #np.random.shuffle(examples)

    #For each match, processed data to be input for
    #logistic regression/NN will be stored here

    matchData = []
    defaultPoints = 50
    defaultGD = 0
    #for example in examples:
    inputDict = {}
    exampleHT = examples[-1]['HomeTeam']
    exampleAT = examples[-1]['AwayTeam']
    inputDict['HomeTeam'] = exampleHT
    inputDict['AwayTeam'] = exampleAT

    if exampleHT in teamPreSeason.keys():
        inputDict['HTPrevSeasonPoints'] = teamPreSeason[exampleHT]['TotalPoints']
        inputDict['HTPrevSeasonGD'] = teamPreSeason[exampleHT]['GoalDifference']
        if (defaultPoints > inputDict['HTPrevSeasonPoints']) and (inputDict['HTPrevSeasonPoints'] != 0):
            defaultPoints = inputDict['HTPrevSeasonPoints']
            defaultGD = inputDict['HTPrevSeasonGD']
    else:
        inputDict['HTPrevSeasonPoints'] = 0
        inputDict['HTPrevSeasonGD'] = 0
    if exampleAT in teamPreSeason.keys():
        inputDict['ATPrevSeasonPoints'] = teamPreSeason[exampleAT]['TotalPoints']
        inputDict['ATPrevSeasonGD'] = teamPreSeason[exampleAT]['GoalDifference']
        if (defaultPoints > inputDict['HTPrevSeasonPoints']) and (inputDict['ATPrevSeasonPoints'] != 0):
            defaultPoints = inputDict['ATPrevSeasonPoints']
            defaultGD = inputDict['ATPrevSeasonGD']
    else:
        inputDict['ATPrevSeasonPoints'] = 0
        inputDict['ATPrevSeasonGD'] = 0

    for match in teamStats[exampleHT]:
        if (match['Opponent'] == exampleAT) and (match['HomeOrAway'] == 'Home'): 
            HTMatchStats = match
    for match in teamStats[exampleAT]:
        if (match['Opponent'] == exampleHT) and (match['HomeOrAway'] == 'Away'): 
            ATMatchStats = match

    inputDict['HTMatchNum'] = HTMatchStats['MatchNum']
    if HTMatchStats['MatchNum'] != 1:
        inputDict['HTPointsPerGame'] = HTMatchStats['PreMatchPoints'] / (HTMatchStats['MatchNum'] - 1)
    else:
        inputDict['HTPointsPerGame'] = 0
    inputDict['HTGD'] = HTMatchStats['PreMatchGD']
    inputDict['HTPast5PPG'] = HTMatchStats['Past5AvgPoints']
    inputDict['HTPast5HomePPG'] = HTMatchStats['Past5AvgHome']
    inputDict['HTPast5GD'] = HTMatchStats['Past5GD']
    inputDict['ATMatchNum'] = ATMatchStats['MatchNum']
    if ATMatchStats['MatchNum'] != 1:
        inputDict['ATPointsPerGame'] = ATMatchStats['PreMatchPoints'] / (ATMatchStats['MatchNum'] - 1)
    else:
        inputDict['ATPointsPerGame'] = 0
    inputDict['ATGD'] = ATMatchStats['PreMatchGD']
    inputDict['ATPast5PPG'] = ATMatchStats['Past5AvgPoints']
    inputDict['ATPast5AwayPPG'] = ATMatchStats['Past5AvgAway']
    inputDict['ATPast5GD'] = ATMatchStats['Past5GD']

    if HTMatchStats['Result'] == 'Win':
        inputDict['Result'] = 'H'
    elif HTMatchStats['Result'] == 'Lose':
        inputDict['Result'] = 'A'
    else:
        inputDict['Result'] = 'D'


    matchData.append(inputDict)
    #print(inputDict)
    #print("")

    #print(defaultPoints)
    #print(defaultGD)


    #Feature balancing to account for very early matches in the season
    for match in matchData:
        if match['HTPrevSeasonPoints'] == 0:
            match['HTPrevSeasonPoints'] = defaultPoints
            match['HTPrevSeasonGD'] = defaultGD
        if match['ATPrevSeasonPoints'] == 0:
            match['ATPrevSeasonPoints'] = defaultPoints
            match['ATPrevSeasonGD'] = defaultGD
        
        #Extra balancing
        HTPrevSeasonPPG = match['HTPrevSeasonPoints'] / 38
        ATPrevSeasonPPG = match['ATPrevSeasonPoints'] / 38
        HTPrevSeasonGDP5G = match['HTPrevSeasonGD']*5 / 38
        ATPrevSeasonGDP5G = match['ATPrevSeasonGD']*5 / 38
        HTMatchesPlayed = match['HTMatchNum'] - 1
        ATMatchesPlayed = match['ATMatchNum'] - 1
        if HTMatchesPlayed < 5:
            match['HTPointsPerGame'] = ((HTMatchesPlayed/5) * match['HTPointsPerGame']) + (((5-HTMatchesPlayed)/5) * HTPrevSeasonPPG)
            match['HTGD'] = match['HTGD'] + (((5-HTMatchesPlayed)/5) * HTPrevSeasonGDP5G)
            match['HTPast5PPG'] = match['HTPointsPerGame']
            match['HTPast5HomePPG'] = ((HTMatchesPlayed/5) * match['HTPast5HomePPG']) + (((5-HTMatchesPlayed)/5) * HTPrevSeasonPPG)
            match['HTPast5GD'] = match['HTGD']
        if ATMatchesPlayed < 5:
            match['ATPointsPerGame'] = ((ATMatchesPlayed/5) * match['ATPointsPerGame']) + (((5-ATMatchesPlayed)/5) * ATPrevSeasonPPG)
            match['ATGD'] = match['ATGD'] + (((5-ATMatchesPlayed)/5) * ATPrevSeasonGDP5G)
            match['ATPast5PPG'] = match['ATPointsPerGame']
            match['ATPast5AwayPPG'] = ((ATMatchesPlayed/5) * match['ATPast5AwayPPG']) + (((5-ATMatchesPlayed)/5) * ATPrevSeasonPPG)
            match['ATPast5GD'] = match['ATGD']        
    '''
    nnData.append(matchData)
    print('length of nnData: ' + str(len(nnData)))
    '''

    '''
    #Put 280 examples in Training Set, 100 examples in Test Set
    #For now, train_set_x should be 16 x 280
    #train_set_y should be 3 x 280, 3 due to win, draw or lose
    n_x = 16
    n_y = 3
    m_total = 380 * (YEAR_END - YEAR_START - 1)
    m_train = 280 * (YEAR_END - YEAR_START - 1)
    m_test = 100 * (YEAR_END - YEAR_START - 1)

    parts_x = []
    parts_y = []
    for j in range(len(nnData)):
        part_train_set_x = np.zeros((n_x, 380))
        part_train_set_y = np.zeros((n_y, 380))
        for i in range(380):
    '''
    predict_x = [
    matchData[0]['HTPrevSeasonPoints'], 
    matchData[0]['HTPrevSeasonGD'], 
    matchData[0]['ATPrevSeasonPoints'], 
    matchData[0]['ATPrevSeasonGD'], 
    matchData[0]['HTMatchNum'], 
    matchData[0]['HTPointsPerGame'], 
    matchData[0]['HTGD'], 
    matchData[0]['HTPast5PPG'], 
    matchData[0]['HTPast5HomePPG'],
    matchData[0]['HTPast5GD'],
    matchData[0]['ATMatchNum'], 
    matchData[0]['ATPointsPerGame'], 
    matchData[0]['ATGD'], 
    matchData[0]['ATPast5PPG'], 
    matchData[0]['ATPast5AwayPPG'], 
    matchData[0]['ATPast5GD']] 

    '''        
            if nnData[j][i]['Result'] == 'H':
                part_train_set_y[0][i] = 1
            elif nnData[j][i]['Result'] == 'A':
                part_train_set_y[2][i] = 1
            else:
                part_train_set_y[1][i] = 1
                
        parts_x.append(part_train_set_x)
        parts_y.append(part_train_set_y)

    for i in range(len(parts_x)):
        print('i is: ' + str(i))
        if i == 0:
            train_set_x = part_train_set_x
            train_set_y = part_train_set_y
        else:
            train_set_x = np.concatenate((train_set_x, part_train_set_x), axis = 1)
            train_set_y = np.concatenate((train_set_y, part_train_set_y), axis = 1)

    np.random.seed(1)
    combined_xy = np.concatenate((train_set_x, train_set_y))
    combined_xy = combined_xy.T
    np.random.shuffle(combined_xy)
    combined_xy = combined_xy.T

    print('combined_xy: ')
    print(combined_xy)

    train_set_x = combined_xy[:16]
    train_set_y = combined_xy[16:]
        
    test_set_x = train_set_x[:, m_train:]
    train_set_x = train_set_x[:, :m_train]
    test_set_y = train_set_y[:, m_train:]
    train_set_y = train_set_y[:, :m_train]


    print(train_set_x.shape)
    print(test_set_x.shape)
    print(train_set_y.shape)
    print(test_set_y.shape)
    print(test_set_y.T)



    print("before normalization:")
    print(train_set_x)
    print("after normalization:")
    '''
    # Doesn't work
    #print(np.linalg.norm(train_set_x, axis = (16, 280)))

    # Normalization
    def normalize(X):
        meanRow = (np.sum(X, axis = 1))/X.shape[1]
        #print(meanRow)
        sigma = np.std(X, axis = 1)
        #print(sigma)
        result = (X - meanRow[:, None]) / sigma[:, None]
        return result, meanRow, sigma
        #print((train_set_x.T - meanRow) / sigma)

    '''
    train_set_x, meanRow, sigma = normalize(train_set_x)
    test_set_x = (test_set_x - meanRow[:, None]) / sigma[:, None]
    '''
    with open("feature_mean.txt", "r") as infile:
        meanRow = np.matrix(infile.read())
    with open("feature_stddev.txt", "r") as infile:
        sigma = np.matrix(infile.read())

    #print(predict_x)

    #Normalize predict_X
    predict_x = (predict_x - meanRow) / sigma

    '''
    w = np.zeros((n_x, n_y))
    b = 0
    '''

    #Neural Network
    def sigmoid (z):
        s = 1 / (1 + np.exp(-z))
        return s

    def layer_sizes(X, Y):
        n_x = X.shape[0] # size of input layer
        n_h = 4
        n_y = 3 # size of output layer
        return (n_x, n_h, n_y)


    def initialize_parameters(n_x, n_h, n_y):
        W1 = np.random.randn(n_h, n_x) * 0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h) * 0.01
        b2 = np.zeros((n_y, 1))
        
        assert (W1.shape == (n_h, n_x))
        assert (b1.shape == (n_h, 1))
        assert (W2.shape == (n_y, n_h))
        assert (b2.shape == (n_y, 1))
        
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}
        
        return parameters

    def forward_propagation(X, parameters):
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)

        #print("Z1 shape")
        #print(Z1.shape)
        #print("A1 shape")
        #print(A1.shape)
        #print("Z2 shape")
        #print(Z2.shape)
        #print("A2 shape")
        #print(A2.shape)

        #assert(A2.shape == (1, X.shape[1]))
        
        cache = {"Z1": Z1,
                 "A1": A1,
                 "Z2": Z2,
                 "A2": A2}
        
        return A2, cache

    def compute_cost(A2, Y, parameters):
        m = Y.shape[1] # number of example

        # Compute the cross-entropy cost
        logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),(1-Y)) 
        cost = -np.sum(logprobs) / m
        
        cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                    # E.g., turns [[17]] into 17 
        assert(isinstance(cost, float))
        
        return cost

    def backward_propagation(parameters, cache, X, Y):
        m = X.shape[1]
        W1 = parameters["W1"]
        W2 = parameters["W2"]
        A1 = cache["A1"]
        A2 = cache["A2"]
        dZ2 = A2 - Y
        dW2 = (np.dot(dZ2, A1.T)) / m
        db2 = (1/m) * np.sum(dZ2, axis = 1, keepdims = True)
        dZ1 = (np.dot(W2.T, dZ2)) * (1 - np.power(A1, 2))
        dW1 = (1/m) * np.dot(dZ1, X.T)
        db1 = (1/m) * np.sum(dZ1, axis = 1, keepdims = True)
        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}
        
        return grads

    def update_parameters(parameters, grads, learning_rate = 2):
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        dW1 = grads["dW1"]
        db1 = grads["db1"]
        dW2 = grads["dW2"]
        db2 = grads["db2"]

        W1 = W1 - (learning_rate * dW1)
        b1 = b1 - (learning_rate * db1)
        W2 = W2 - (learning_rate * dW2)
        b2 = b2 - (learning_rate * db2)
        
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}
        
        return parameters

    def nn_model(X, Y, n_h, num_iterations, print_cost=False):
        np.random.seed(3)
        n_x = layer_sizes(X, Y)[0]
        n_h = layer_sizes(X, Y)[1]
        n_y = layer_sizes(X, Y)[2]
        
        # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
        parameters = initialize_parameters(n_x, n_h, n_y)
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        for i in range(0, num_iterations):

            A2, cache = forward_propagation(X, parameters)

            cost = compute_cost(A2, Y, parameters)

            grads = backward_propagation(parameters, cache, X, Y)
     
            parameters = update_parameters(parameters, grads, learning_rate = 0.1)
          
            # Print the cost every 1000 iterations
            if print_cost and i % 1000 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))

        return parameters

    def predict(parameters, X):
        A2, cache = forward_propagation(X, parameters)
        #predictions = (A2 > 0.5)
        #print(A2)

        #Softmax done here    
        A2 = A2 / np.sum(A2, axis = 0)
        #print(A.shape)
        onehot = np.argmax(A2, axis = 0)
        predictions= np.zeros(A2.shape)
        for i in range(A2.shape[1]):
            predictions[onehot[i]][i] = 1
        #print(onehot)
        #print(predictions)
        
        '''
        for i in range(30):
            print (1 - A2[0][random.randint(0, A2.shape[1])])
        '''
        return A2
    '''
    parameters = nn_model(train_set_x, train_set_y, n_h = 4, num_iterations = 10000, print_cost=True)
    pred_train = predict(parameters, train_set_x)
    pred_test = predict(parameters, test_set_x)
    '''
    parameters = {}
    with open("weights_W1.txt", "r") as infile:
        parameters['W1'] = np.matrix(infile.read())
    with open("weights_b1.txt", "r") as infile:
        parameters['b1'] = np.matrix(infile.read())
    with open("weights_W2.txt", "r") as infile:
        parameters['W2'] = np.matrix(infile.read())
    with open("weights_b2.txt", "r") as infile:
        parameters['b2'] = np.matrix(infile.read())

    n_h = 4
    parameters['W1'] = np.reshape(parameters['W1'], (n_h, 16))
    parameters['b1'] = np.reshape(parameters['b1'], (n_h, 1))
    parameters['W2'] = np.reshape(parameters['W2'], (3, n_h))
    parameters['b2'] = np.reshape(parameters['b2'], (3, 1))

    '''
    print(predict_x.T)
    print(np.shape(parameters['W1']))
    print(np.shape(parameters['b1']))
    print(np.shape(parameters['W2']))
    print(np.shape(parameters['b2']))
    print((parameters['W1']))
    print((parameters['b1']))
    print((parameters['W2']))
    print((parameters['b2']))
    '''

    result = predict(parameters, predict_x.T)
    #print('result:')
    #print(result)

    #print(np.count_nonzero(Y))
    #print(len(Y))
    #print(1 - (np.count_nonzero(Y)/len(Y)))


    #Convert false to 0 and true to 1
    def convert(predictions, n, m):
        pred = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                if predictions[i][j]== False:
                    pred[i][j] = 0
                else:
                    pred[i][j] = 1
        return pred
    '''
    pred_train = convert(pred_train, n_y, m_train)
    pred_test = convert(pred_test, n_y, m_test)

    print("Train Accuracy: {} %".format(np.sum(pred_train * train_set_y) * 100 / m_train))
    print("Test Accuracy: {} %".format(np.sum(pred_test * test_set_y) * 100 / m_test))

    #print (((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.T.size)*100))
    '''
    '''
    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 1, print_cost = True)
    '''

    return result

result = predict_match(19, 0, 18)
print(result)
#printTeamList(19)
   
    



