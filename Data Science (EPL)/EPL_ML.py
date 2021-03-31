from EPL_Data_Science import *
from sklearn.model_selection import train_test_split
import math

""" GOAL: Classify into win, lose or draw based on observations on stats
(Possession, Corners, Fouls) """


Classes = [-1, 0, 1]

def split_df(df, percent_test, random_state = np.random.randint(10000)):
    train, test = train_test_split(df, test_size=0.2, random_state = random_state)
    train.reset_index(inplace = True)
    test.reset_index(inplace = True)
    return train, test

def test_set_to_intervals(test_set):
    """Inputs: a dataframe.
    Returns: a dataframe with the stats classified into intervals"""
    new_test = pd.DataFrame()
    for stat, interval in get_stats_dict().items():
        test_stat = pd.cut(test_set[stat], bins = interval)
        new_test[stat] = test_stat
    return new_test

def get_stats_dict():
    """Creates a dictionary mapping stats to their intervals"""
    
    pos_intervals = pd.interval_range(start = 0, end = 100, periods = 10)
    corner_intervals = pd.interval_range(start = 0, end = 16, periods = 8)
    foul_intervals = pd.interval_range(start = 0, end = 15, periods = 5)
    shot_intervals = pd.interval_range(start = 0, end = 24, periods = 8)
    shot_on_intervals = pd.interval_range(start = 0, end = 16, periods = 8)
    off_intervals = pd.interval_range(start = 0, end = 10, periods = 10)

    stats = {
             'Home_Pos' : pos_intervals, 
             'Home_Corners' : corner_intervals, 'Away_Corners' : corner_intervals,
##             'Home_Fouls' : foul_intervals, 'Away_Fouls' : foul_intervals,
             'Home_Shots' : shot_intervals, 'Away_Shots' : shot_intervals,
             'Home_Shots_On' : shot_on_intervals, 'Away_Shots_On' : shot_on_intervals}
##             'Home_Off' : off_intervals, 'Away_Off' : off_intervals}
    return stats

def get_prior1(training_set):
    """Inputs: a dataframe.
    Returns: a dictionary mapping each class to its probability in the datframe"""
    prior_class_occurs = training_set['Winner'].value_counts()/len(training_set)
    prior1 = {1:prior_class_occurs[1], 0:prior_class_occurs[0], -1:prior_class_occurs[-1]}
    return prior1

    
def get_prior2(training_set):
    """Inputs: a dataframe.
    Returns: a dictionary in this form
    {  Class  :   {stat  :  {interval  :  probability  }  }  }"""
    
    
    all_probs = {-1 : None, 0 : None, 1 : None }

    for Class in Classes:

        prior_observations = training_set.loc[training_set['Winner'] == Class]

        class_stats = {}
        for stat, interval in get_stats_dict().items():
            each_stat = {}
            
            prior_pos = pd.cut(prior_observations[stat], bins = interval)
            prior_pos = prior_pos.value_counts()/len(prior_pos)
            class_stats.update({stat:prior_pos.to_dict()})
            
        all_probs[Class] = class_stats
        
    return all_probs



def classify_test_sets(df, num_runs, percent_test, check_guesses = False, compete = False):

    all_accuracies = []
    strong_indicators = ['Home_Shots_On', 'Away_Shots_On']
    for each_run in range(num_runs):
        
        # Step 1: Split into training and test sets
        train, test = split_df(df, percent_test)
        new_test = test_set_to_intervals(test)
        
        # Training set is used to calculate priors (i.e probabilities that
        
        # 1. A class occurs
        prior1 = get_prior1(train)
        
        # 2. That an observation occurs GIVEN that the class occured
        prior2 = get_prior2(train)
        
        correct_guesses = 0
        correct_ans = 0
        
        for i in range(len(new_test)):

            # Goal: Find the class that maximises the products of
            # Priors 1 (for all stats) and Prior 2
            # i.e argmax (for each class, c) of the products (for all stats s) of
            # P(s | c)*P(c)   

            highest = 0
            guess = None
            for Class in Classes:
                p = prior1[Class]
                for stat, interval in get_stats_dict().items():
                    try:
                        if stat in strong_indicators:
                            p *= 5*max(1E-6, prior2[Class][stat][new_test.loc[i, stat]])
                        else:
                            p *= max(1E-6, prior2[Class][stat][new_test.loc[i, stat]])
                    except KeyError:
                        continue
                if p > highest:
                    guess = Class
                    highest = p
            if guess == test.loc[i, 'Winner']:
                correct_guesses += 1
            if compete or check_guesses:
                print('\n')
                print(test.loc[i, get_stats_dict().keys()])
                if compete:
                    ans = input('Enter to proceed: ')
                    if int(ans) == test.loc[i, 'Winner']:
                        correct_ans += 1

            
            if check_guesses: 
                print("\nMachine's Guess:", guess)
                print('Actual:', test.loc[i, 'Winner'], '\n')
                if compete:
                    print('MACHINE:', correct_guesses, 'out of', i+1)
                    print('YOU:', correct_ans, 'out of', i+1)
                else:
                    try:
                        print('Accuracy of', round((correct_guesses/(i+1)) * 100, 2))
                    except ZeroDivisionError:
                        continue
                
        accuracy = round(correct_guesses/len(new_test) * 100,2) 
        all_accuracies.append(accuracy)
        
        print('Run #' + str(each_run+1))
        print('Accuracy of', accuracy)

    print('\nDone!')
    print('Mean Accuracy:', sum(all_accuracies)/len(all_accuracies))
    
    return all_accuracies



if __name__ == "__main__":
    classify_test_sets(df, 10, 0.2, True)
##classify_test_sets(df, 10, 0.2)

