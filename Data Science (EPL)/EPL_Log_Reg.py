from EPL_Data_Science import *
import EPL_ML
import EPL_Lin_Reg
import numpy as np  
import matplotlib.pyplot as plt
import math


df['Home_Win?'] = np.where(df['Home_Goals'] > df['Away_Goals'], 1, 0)
df['Away_Win?'] = np.where(df['Home_Goals'] < df['Away_Goals'], 1, 0)
df['Draw?'] = np.where(df['Home_Goals'] == df['Away_Goals'], 1, 0)
##df = df[0:300]
df_copy = df.copy()

# the array w denotes parameter vector
# b denotes the constant term

##['Home_Pos', 'Home_Corners', 'Home_Shots', 'Home_Shots_On',
##  'Away_Corners', 'Away_Shots', 'Away_Shots_On']

stat_names = ['Home_Pos', 'Home_Corners', 'Home_Shots', 'Home_Shots_On',
              'Away_Corners', 'Away_Shots', 'Away_Shots_On', 'Shot_Diff',
              'Corner_Diff']
Classes = {'Away_Win?' : -1, 'Draw?' : 0, 'Home_Win?' : 1}

def make_higher_order(df, degree):
    for name in stat_names[:7]:
        df[name + '_degree_' + str(degree)] = df[name]**degree
        stat_names.append(name + '_degree_' + str(degree))
    return stat_names

def get_x1_times_x2(df):
    length = len(stat_names)
    for i in range(length-1):
        for j in range(i + 1, length):
            stat_names.append(stat_names[i] + '_TIMES_' + stat_names[j])
            df[stat_names[i] + '_TIMES_' + stat_names[j]] = df[stat_names[i]] * df[stat_names[j]]  

    return stat_names

def get_prediction(w, b, sample_data):
    """sample_data as an array"""

##    assert len(w) == len(sample_data)
    power = -np.dot(w, sample_data) + b
    return 1/(1 + math.exp(power))

def get_new_params(w, b, train, Class, alpha):
    """Using Gradient Descent"""
    # sum += (prediction - actual)(prediction)(1-prediction)(data_point), for num_train data points
    # Adjustment = -(1/num_train)(sum)

    
    new_params = np.zeros(len(w) + 1)
    error = 0
    num_correct = 0
##    print(w)
    for i in range(len(train.index)):    
        sample_data = np.array([])
        actual = train.loc[i, Class]
        
        for stat in stat_names:
            sample_data = np.append(sample_data, train.loc[i, stat])
       
            
        prediction = get_prediction(w, b, sample_data)
        error += abs(prediction - actual)
        sample_data = np.append(sample_data, 1)
        if round(prediction) == actual:
            num_correct += 1
        
        temp_params = np.full(len(w) + 1, (prediction-actual)*prediction*(1-prediction))
##        temp_params = np.multiply(temp_params, sample_data)

        new_params = np.add(new_params, temp_params)
        
    
##    accuracy = round(num_correct/len(train.index) *100, 2)
    
    new_params = np.multiply(new_params, -alpha/len(w))
##    print('Error:', error)
##    print(new_params)
##    testing_data = np.array([])
##    for stat in stat_names:
##        testing_data = np.append(testing_data, train.loc[10, stat])
##    test_prediction = get_prediction(w, b, testing_data)
##    print('test prediction:', test_prediction)
##    print('test actual:', train.loc[10, Class])
    
    return new_params, error

def create_model(train, Class, alpha, epochs, tries):

    lowest_err = 1E6
    best_w = None
    best_b = None
    print(Class, '\n')
    for x in range(tries):
        w = np.random.randn(1, len(stat_names))
        b = np.random.randn()
##        print(w, b)
        curr_index = 0
        for i in range(epochs):
##            print(w, b)
            new_params, error = get_new_params(w, b, train, Class, alpha)
            w = np.add(w, new_params[:-1])
            b = b + new_params[-1]
##            print('new params', new_params)
##            print('w:', w)
##            if (i+1) % 50 == 0 or i == 0:
##                print('Intermediate Accuracy:', accuracy)

        if error < lowest_err:
##            print('NEW BEST')
            lowest_err = error
            best_w = w
            best_b = b
        print('DONE\n')
        
    
    return [best_w, best_b]

def test_model(test, Class, model):
    num_correct = 0
    w = model[0]
    b = model[1]

    for i in range(len(test.index)):
        print(test.loc[i, stat_names])
        sample_data = np.array([])
        for name in stat_names:
            sample_data = np.append(sample_data, test.loc[i, name])
        prediction = get_prediction(w, b, sample_data)
        guess = round(prediction)
        print('PREDICTION:', prediction)
        print('GUESS:', guess)
        print('ACTUAL:', test.loc[i, Class])
        if guess == test.loc[i, Class]:
            num_correct += 1
            
    print('Accuracy:', round(num_correct/len(test.index) * 100, 2))
            

    
def main(df, test_percent, alpha, epochs, tries, compete = False, check = False):
    
##    random_state = np.random.randint(10000)
    random_state = rand
    train, test = EPL_ML.split_df(df, test_percent, random_state)
    train_copy, test_copy = EPL_ML.split_df(df_copy, test_percent, random_state)

    # Training Phase
    
    models = {}

    for Class, guess in Classes.items():
        
        models.update({Class : None})
        
        models[Class] = create_model(train, Class, alpha, epochs, tries)
        
        print(models)

    # Test Phase    
    num_correct = 0
    score = 0
    for i in range(len(test.index)):
        if check or compete:
            print(test_copy.loc[i, stat_names[:7]])
        sample_data = np.array([])
        most_confidence = 0
        final_guess = None
        
        if compete:
            answer = input('\nWhat do you think? ')
            if int(answer) == test.loc[i, 'Winner']:
                score += 1
            
        for name in stat_names:
            sample_data = np.append(sample_data, test.loc[i, name])
            
        for Class, guess, in Classes.items():
            prediction = get_prediction(models[Class][0], models[Class][1], sample_data)
            if check or compete:
                print('')
                print(Class)
                print('Confidence: ' + str(round(prediction*100, 2)) + '%')
            
            if prediction > most_confidence:
                most_confidence = prediction
                final_guess = guess
        if final_guess == test.loc[i, 'Winner']:
            num_correct += 1
            
        if check or compete:
            print("\nMACHINE's GUESS:", final_guess)
            print('ACTUAL:', test.loc[i, 'Winner'])

        if compete:
            print('YOU:', score, 'out of', i+1)
            
        if check or compete:
            print('MACHINE:', num_correct, 'out of', i+1)
            print('')
            
    print('Accuracy:', round(num_correct/len(test.index) * 100, 2))
    return models

if __name__ == "__main__":
##    rand = np.random.randint(10000)
    MAX_ITER = 150
    rand = 7
    
    """ Data Normalisation """
    for name in stat_names:
        if name == 'Shot_Diff' or name == 'Corner_Diff':
            pass
        else:
            df[name] = df[name]/df[name].max()
    df['Shot_Diff'] = df['Home_Shots'] - df['Away_Shots']
    df['Corner_Diff'] = df['Home_Corners'] - df['Away_Corners']

    
##    stat_names = make_higher_order(df, 2)
##    stat_names = make_higher_order(df, 3)

##    stat_names = get_x1_times_x2(df)
##    main(df, 0.2, 1E-2, 200)
    manual_models = main(df, 0.3, 0.01, 5, 30, True)






##    """ Using sklearn """
##    from sklearn.linear_model import LogisticRegression
##    from sklearn.model_selection import train_test_split
##
##    
####    rand = 2
##    X = df[stat_names]
##    Y = df['Winner']
##    X_copy = df_copy[stat_names[:7]]
##    Y_copy = df_copy['Winner']
##
##    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.2, random_state = rand)
##    train_x_copy, test_x_copy, train_y_copy, test_y_copy = train_test_split(X_copy, Y_copy, test_size = 0.2, random_state = rand)
##
##    model = LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg', max_iter = 400)
##    model.fit(train_x, train_y)
##
##    predictions = model.predict(test_x)
##    confidences = model.predict_proba(test_x)
##    
##    test_x.reset_index(inplace = True)
##    test_y.reset_index(inplace = True, drop = True)
##    test_x_copy.reset_index(inplace = True)
##    test_y_copy.reset_index(inplace = True, drop = True)
##    del test_x['index']
##
##
##
##
##    """ Prints out each test sample and the model's prediction """
##    num_correct = 0
##    score = 0
##    for i in range(len(test_x.index)):
##        print(test_x_copy.loc[i, stat_names[:7]])
##        answer = input('What do you think? ')
##        print('')
##        for index in range(len(list(Classes.keys()))):
##            print(list(Classes.keys())[index])
##            print('Confidence:', round(confidences[i][index], 2)*100)
##            
##        print("\nMachine's Guess:", predictions[i])
##        print('Actual:', test_y.loc[i])
####        print('\n')
##        if int(answer) == test_y[i]:
##            score += 1
##        if predictions[i] == test_y[i]:
##            num_correct += 1
##        print('\nYOU:', score, 'out of', i+1)
##        print('MACHINE:', num_correct, 'out of', i+1)
##        print('')

##    """ Prints out parameters for each model """
##    i = 0
##    all_mapped_coef = {}
##    for Class, guess in Classes.items():
##        coefficients = {}
##        for x in range(len(model.coef_[i])):
##            coefficients.update({stat_names[x] : model.coef_[i][x]})
##        i += 1
##        all_mapped_coef.update({Class : coefficients})
##        print(Class)
##        print(all_mapped_coef[Class])
##        print('\n')
##        
##    print('ACCURACY:', round(model.score(test_x, test_y)*100, 3))


##    with open('Comparison', 'w') as f:
##        f.write(str(manual_models))
##        f.write(str(model.coef_))
