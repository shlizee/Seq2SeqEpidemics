import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ilinet_full = pd.DataFrame(pd.read_csv('ILINet.csv'))
ilinet_full_seasons =ilinet_full[0:782] #ignoring current season for now
data = ilinet_full_seasons['ILITOTAL'].values
weeks = ilinet_full_seasons['WEEK'].values[0:52]
fig1=plt.figure(figsize=(15, 10), dpi= 80, facecolor='w', edgecolor='k')
plt.plot(data, color = 'black')
plt.title("ILINet Data 2004-2019")
plt.ylabel("Number of Patients for ILI")
plt.show()


end_weeks=ilinet_full_seasons.index[ilinet_full_seasons['WEEK'] == 39].tolist()
start_weeks=ilinet_full_seasons.index[ilinet_full_seasons['WEEK'] == 40].tolist()

end_weeks=ilinet_full_seasons.index[ilinet_full_seasons['WEEK'] == 39].tolist()
start_weeks=ilinet_full_seasons.index[ilinet_full_seasons['WEEK'] == 40].tolist()

seasons=[]
for i in range(len(start_weeks)):
    c=start_weeks[i]
    seasons.append(data[c:c+52])
    
fig2=plt.figure(figsize=(15, 10), dpi= 80, facecolor='w', edgecolor='k')
plt.plot(seasons[-1],color = 'black')
y_pos = np.arange(len(weeks))
plt.xticks(y_pos, weeks)
plt.xlabel("Weeks", fontsize=30)
plt.ylabel("Number of New Cases", fontsize=30)
plt.title("2018-2019 Flu Season", fontsize=30)
plt.show()
seasons=np.vstack(seasons)


