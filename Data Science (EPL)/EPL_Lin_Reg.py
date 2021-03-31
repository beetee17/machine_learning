from EPL_Data_Science import *
import EPL_ML
import numpy as np  
import matplotlib.pyplot as plt

def plot_func(function, x_range):  
    x = np.array(x_range)  
    y = eval(function)
    plt.plot(x, y)  


def scatter_plot(df, predictor, target):
    plt.scatter(df[predictor], df[target])
                
def get_MSE(predictions, actuals):
    """
    Input:
        list of predicted values by regression model
        list of actual values corresponding to predictions
    Returns:
        The mean squared error (MSE)"""
    
    assert len(predictions) == len(actuals)
    total_squared_error = 0
    
    for i in range(len(predictions)):
        total_squared_error += (predictions[i] - actuals[i])**2
        
    mean_squared_error = total_squared_error/len(predictions)
    return mean_squared_error


def create_model(slope, const):
    """
    Input:
        m: gradient parameter
        b: intercept parameter
    Returns:
        an equation that can be plotted using plot_func()"""

    return str(slope) + '*x + ' + str(const)

def optimise_model(df, predictor, target, epochs, alpha, show, check):
    
    df.reset_index(inplace = True)

    lowest_MSE = 1E6
    best_slope = None
    best_const = None
    
    for new_init_values in range(15):
        
        slope = np.random.randint(-max(df[target].to_list())/min(df[predictor].to_list()),
                                          max(df[target].to_list())/min(df[predictor].to_list()))
        const = np.random.randint(0.7*min(df[target].to_list()), 1.3*max(df[target].to_list()))

        if check:
            
            print('Try #:', new_init_values)
            print('Initial Slope:', slope)
            print('Initial Constant Term:', const, '\n')
        
        for epoch in range(1, epochs + 1):

            predictions = []
            actuals = df[target]

            for i in range(len(df[predictor])):
                x = df.loc[i, predictor]
                y = eval(create_model(slope, const))
                predictions.append(y)

            MSE = get_MSE(predictions, actuals)
            if check:
                if epoch % 10 == 0:
                    print('Epoch #' + str(epoch))
                    print('Slope:', round(slope, 4), 'Constant Term:', round(const,4))
                    print('Mean Squared Error:', round(MSE, 4))
                    print('\n')
            
            slope, const = get_new_params(slope, const, df, predictor, predictions, actuals, alpha)

        if MSE < lowest_MSE:
            lowest_MSE = MSE
            best_slope = slope
            best_const = const
            
    print('Mean Squared Error on Training Data:', round(lowest_MSE, 4))
    
    if show:
        plt.figure()
        plt.title('Linear Regression on Training Data')
        plt.xlabel(predictor)
        plt.ylabel(target)
        plot_func(create_model(best_slope, best_const), df[predictor])
        scatter_plot(df, predictor, target)
        plt.show() 

    return slope, const


def get_new_params(slope, const, df, predictor, predictions, actuals, alpha):
    """Calculates the partial derivative of each parameter w.r.t the MSE function and
    returns the new parameter values for subsequent epoch"""
    
    const_ans = 0
    slope_ans = 0
    
    assert len(predictions) == len(actuals)
    
    for i in range(len(predictions)):
        const_ans += (predictions[i] - actuals[i])
        slope_ans += ((predictions[i] - actuals[i])*df.loc[i, predictor])
        
    const_ans *= (2*alpha/len(predictions))
    slope_ans *= (2*alpha/len(predictions))

    slope -= slope_ans
    const -= const_ans
    
    return slope, const

def predict(slope, const, input_val):
    x = input_val
    prediction = eval(create_model(slope, const))
    return prediction

def main(df, percent_test, predictor, target, epochs, alpha = 0.0001,
               show_training = False, check_training = False, show_test = False):
    
    train, test = EPL_ML.split_df(df, percent_test)
    slope, const = optimise_model(train, predictor, target, epochs, alpha,
                                  show_training, check_training)

    predictions = []
    actuals = test[target].to_list()
    
    for i in range(len(test[predictor])):
        prediction = predict(slope, const, test.loc[i, predictor])
        predictions.append(prediction)
        
    MSE = get_MSE(predictions, actuals)
    print('Mean Squared Error on Test Data:', round(MSE, 4))

    if show_test:
        plt.figure()
        plt.title("Model's Performance on Test Data")
        plt.xlabel(predictor)
        plt.ylabel(target)
        plot_func(create_model(slope, const), df[predictor])
        scatter_plot(df, predictor, target)
        plt.show()
        
    return MSE

        
        
    
if __name__ == "__main__":
    main(df, 0.1, 'Home_Pos', 'Home_Shots', 10, 0.0001, True, False, True)
##    main(df, 0.1, 'Home_Pos', 'Away_Shots', 10, 0.0001, True, False, True)
##    main(df, 0.1, 'Away_Pos', 'Home_Shots', 10, 0.0001, True, False, True)
