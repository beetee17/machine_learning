##import pandas as pd
##import numpy as np
##import seaborn as sns
##import matplotlib.pyplot as plt

import requests
from bs4 import BeautifulSoup as bs
import csv

headers = ['Home_Goals', 'Away_Goals', 'Home_Pos', 'Away_Pos',
           'Home_Shots', 'Away_Shots', 'Home_Shots_On', 'Away_Shots_On',
           'Home_Fouls', 'Away-Fouls', 'Home_Off',
           'Away_Off', 'Home_Corners', 'Away_Corners']
all_rows = []

##for game_id in range(513584, 513840):
for game_id in range(513584, 513684):
    match_report = 'http://www.espn.com/soccer/matchstats?gameId={}'.format(game_id)
    match_stats = requests.get(match_report)
    soup = bs(match_stats.text, 'lxml')
    row = []
    
    for div in soup.find_all('div', {'class':"score-container"}):
        for score in div.find_all('span', {'data-stat':"score"}):
##            print(score.text.strip())
            row.append(score.text.strip())
           
    for div in soup.find_all('div', {'class':"content"}):
        for possession in div.find_all('span', {'class':'chartValue', 'data-stat':"possessionPct"}):
            row.append(possession.text)

    for div in soup.find_all('div', {'class':"content"}):
        for shots in div.find_all('span', {'data-stat':"shotsSummary"}):
            row.append(shots.text.split()[0])
            
    for div in soup.find_all('div', {'class':"content"}):
        for shots in div.find_all('span', {'data-stat':"shotsSummary"}):
            shots_on = shots.text.split()[1]
            new_shots_on = shots_on.lstrip('(')
            newer_shots_on = new_shots_on.rstrip(')')
            row.append(newer_shots_on)

    for div in soup.find_all('div', {'class':'stat-list'}):
        for fouls in div.tbody.find_all('td', {'data-stat':'foulsCommitted'}):
            row.append(fouls.text)

    for div in soup.find_all('div', {'class':'stat-list'}):
        for offsides in div.tbody.find_all('td', {'data-stat':'offsides'}):
            row.append(offsides.text)

    for div in soup.find_all('div', {'class':'stat-list'}):
        for corners in div.tbody.find_all('td', {'data-stat':'wonCorners'}):
            row.append(corners.text)
    
    if len(row) < 12:
        continue
    else:
        all_rows.append(row)
##        print(all_rows[-1])
       
with open('EPL_Stats.csv', 'w+') as save_file:
    writer = csv.writer(save_file)
    writer.writerow(headers)
    for each_row in all_rows:
        writer.writerow(each_row)
