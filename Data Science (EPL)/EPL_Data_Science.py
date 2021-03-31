import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
##import scipy
##from scipy.stats import pearsonr
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

##Z = df[['Home_Pos', 'Home_Shots', 'Home_Shots_On',
##        'Home_Corners', 'Home_Off', 'Home_Fouls', 'Away_Pos',
##        'Away_Shots', 'Away_Shots_On',
##        'Away_Corners', 'Away_Off', 'Away_Fouls']]

data = pd.read_csv('EPL_Stats.csv')
df = pd.DataFrame(data)
df = df.astype(int, copy = True)
df['Winner'] = np.where(df['Home_Goals'] > df['Away_Goals'], 1,
                          (np.where(df['Home_Goals'] < df['Away_Goals'], -1, 0)))

                        
if __name__ == "__main__":
    # Data Normalisation
    ##df['Home_Pos'] = df['Home_Pos']/df['Home_Pos'].max()
    ##df['Home_Shots'] = df['Home_Shots']/df['Home_Shots'].max()
    ##df['Home_Shots_On'] = df['Home_Shots_On']/df['Home_Shots_On'].max()
    ##df['Home_Corners'] = df['Home_Corners']/df['Home_Corners'].max()





    ##plt.figure()
    ##sns.regplot(x = 'Home_Pos', y = 'Home_Shots', data = df)
    ##sns.residplot(x = 'Home_Pos', y = 'Home_Shots', data = df)
    ##plt.show()

    ##pearson, p_val = pearsonr(df['Home_Shots'], df['Home_Goals'])
    ##print(pearson, p_val)


    # Using all match stats to predict total shots

    Z = df[['Home_Pos', 'Home_Goals', 'Home_Corners']]
    Y = df[['Home_Shots']]


    # Splitting DataFrame into test and training sets
    x_train, x_test, y_train, y_test = train_test_split(Z, Y, test_size=0.2)
    ##print("number of test samples :", x_test.shape[0])
    ##print("number of training samples:",x_train.shape[0])

    pr = PolynomialFeatures(degree = 1)
    x_train_pr = pr.fit_transform(x_train)
    x_test_pr=pr.fit_transform(x_test)

    RidgeModel= Ridge(alpha = 0.001)
    RidgeModel.fit(x_train_pr, y_train)
    yhat_train = RidgeModel.predict(x_train_pr)
    ##print("Coefficients are: ", RidgeModel.coef_)

    yhat_test = RidgeModel.predict(x_test_pr)

    ##print('predicted vs test values:', yhat_test[0:20])
    ##print('test set values:', y_test[0:20].values)
    ##print('Training R^2', RidgeModel.score(x_train_pr, y_train))

    ### Apply model to test set

    ##print('Test R^2:', RidgeModel.score(x_test_pr, y_test))


    # Distribution plots for training and test sets (actual vs predictions)
    ##plt.figure()
    ##plt.title("Training")
    ##sns.distplot(y_train, hist = False, color ='r', label = 'Actual')
    ##sns.distplot(yhat_train, hist = False, color ='b', label = 'Prediction')
    ##plt.xlabel('Total # of Shots')
    ##plt.ylabel('Proportion of Games')
    ##plt.xlim(0,)
    ##
    ##plt.figure()
    ##plt.title("Test")
    ##sns.distplot(y_test, hist = False, color ='r', label = 'Actual')
    ##sns.distplot(yhat_test, hist = False, color ='b', label = 'Prediction')
    ##plt.xlabel('Total # of Shots')
    ##plt.ylabel('Proportion of Games')
    ##plt.xlim(0,)
    ##
    ##plt.show()

    plt.figure()
    plt.scatter(df['Home_Pos'], df['Home_Shots'])
    plt.show()

