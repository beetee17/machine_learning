from EPL_Data_Science import *
import EPL_ML 
import EPL_Lin_Reg
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    
    def set_to_intervals(Set, stats_dict):
        """Inputs: a dataframe.
        Returns: a dataframe with the stats classified into intervals"""
        new_set = pd.DataFrame()
        for stat, interval in stats_dict.items():
            set_stat = pd.cut(Set[stat], bins = interval)
            new_set[stat] = set_stat
        return new_set

    stats = EPL_ML.get_stats_dict()
    X = df[stats.keys()]
    Y = df['Winner']
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.2, random_state = 0)
##    print(train_x.head())
    train_x = set_to_intervals(train_x, stats)
    test_x = set_to_intervals(test_y, stats)
    print(train_x.head())
