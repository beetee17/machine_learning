def simulate(all_teams, num_sims, interactive = False):
    import numpy as np
    import pandas as pd
    
    all_sims = pd.DataFrame() 
    all_sims.insert(0, 'Home', np.nan)
    all_sims.insert(1, 'Away', np.nan)
    
    for each_sim in range(num_sims):
        sim = [] 
        teams = np.array(all_teams)
        draw = np.random.choice(teams, (4,2), replace = False)
        
        
        if interactive:     
            for match in draw:
                next_run = input('Next Match?\n')   
                print(match[0] + ' vs ' + match[1])
                print('')
            
        for i in range(len(draw)):
            all_sims.loc[len(all_sims)] = draw[i]
            
    all_sims['Overall'] = all_sims['Home'].astype(object) + ' vs ' + all_sims['Away'].astype(object) 

    prob = all_sims['Overall'].value_counts()
    prob = prob.to_frame()

    prob.rename(index = str, columns = {'Overall':'Fequency'}, inplace= True)
    prob['Probability'] = prob/num_sims

    return prob



    
all_teams = ['Manchester United', 'Liverpool', 'Porto', 'Ajax', 'Barcelona',
         'Tottenham', 'Juventus', 'Manchester City']

print(simulate(all_teams, 1, True))
