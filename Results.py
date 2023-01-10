import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#file = pd.read_csv('Riloff_Results.csv')

#g = sns.catplot(x='Models', y='Score', hue='Type', data=file, kind='bar', height=8, aspect=2, legend=False)
#g.ax.text(-0.4, 0.81+0.05, '0.81', size='x-small')
#g.ax.text(-0.2, 0.83+0.05, '0.83', size='x-small')
#g.ax.text(0, 0.81+0.05, '0.81', size='x-small')
#g.ax.text(0.2, 0.83+0.05, '0.83', size='x-small')

#g.ax.text(0.6, 0.67+0.05, '0.67', size='x-small')
#g.ax.text(0.8, 0.82+0.05, '0.82', size='x-small')
#g.ax.text(1.0, 0.74+0.05, '0.74', size='x-small')
#g.ax.text(1.2, 0.82+0.05, '0.82', size='x-small')

#g.ax.text(1.6, 0.87+0.05, '0.87', size='x-small')
#g.ax.text(1.8, 0.87+0.05, '0.87', size='x-small')
#g.ax.text(2.0, 0.84+0.05, '0.84', size='x-small')
#g.ax.text(2.2, 0.87+0.05, '0.87', size='x-small')

#g.ax.text(2.6, 0.67+0.05, '0.67', size='x-small')
#g.ax.text(2.8, 0.82+0.05, '0.82', size='x-small')
#g.ax.text(3.0, 0.74+0.05, '0.74', size='x-small')
#g.ax.text(3.2, 0.82+0.05, '0.82', size='x-small')

#g.ax.text(3.6, 0.67+0.05, '0.67', size='x-small')
#g.ax.text(3.8, 0.82+0.05, '0.82', size='x-small')
#g.ax.text(4.0, 0.74+0.05, '0.74', size='x-small')
#g.ax.text(4.2, 0.82+0.05, '0.82', size='x-small')

#g.ax.text(4.6, 0.67+0.05, '0.67', size='x-small')
#g.ax.text(4.8, 0.82+0.05, '0.82', size='x-small')
#g.ax.text(5.0, 0.74+0.05, '0.74', size='x-small')
#g.ax.text(5.2, 0.82+0.05, '0.82', size='x-small')

#g.ax.text(5.6, 0.85+0.05, '0.85', size='x-small')
#g.ax.text(5.8, 0.86+0.05, '0.86', size='x-small')
#g.ax.text(6.0, 0.84+0.05, '0.84', size='x-small')
#g.ax.text(6.2, 0.86+0.05, '0.86', size='x-small')

#g.ax.text(6.6, 0.83+0.05, '0.83', size='x-small')
#g.ax.text(6.8, 0.85+0.05, '0.85', size='x-small')
#g.ax.text(7.0, 0.83+0.05, '0.83', size='x-small')
#g.ax.text(7.2, 0.85+0.05, '0.85', size='x-small')

#g.ax.text(7.6, 0.9+0.05, '0.9', size='x-small')
#g.ax.text(7.8, 0.9+0.05, '0.9', size='x-small')
#g.ax.text(8.0, 0.9+0.05, '0.9', size='x-small')
#g.ax.text(8.2, 0.9+0.05, '0.9', size='x-small')

#g.ax.text(8.6, 0.67+0.05, '0.67', size='x-small')
#g.ax.text(8.8, 0.82+0.05, '0.82', size='x-small')
#g.ax.text(9.0, 0.74+0.05, '0.74', size='x-small')
#g.ax.text(9.2, 0.82+0.05, '0.82', size='small')

#plt.xticks(rotation=70)
#plt.ylim([0.0, 1.0])
#plt.title('Riloff Dataset')
#plt.show()

#file1 = pd.read_csv('Training_time.csv')
#riloff = file1[file1['Datasets'] == 'Riloff']
#semeval = file1[file1['Datasets'] == 'SemEval']
#harvested = file1[file1['Datasets'] == 'Harvested']
#plt.scatter(riloff['Models'], riloff['Time'], label='Riloff Dataset')
#plt.scatter(semeval['Models'], semeval['Time'], label='SemEval Dataset')
#plt.scatter(harvested['Models'], harvested['Time'], label='Harvested Dataset')
#plt.legend()
#plt.xticks(rotation=70)
#plt.ylabel('Time')
#plt.xlabel('Models')
#plt.title('Training Time')
#plt.show()

#file2 = pd.read_csv('SemEval_Results.csv')

#g = sns.catplot(x='Models', y='Score', hue='Type', data=file2, kind='bar', height=8, aspect=2, legend=False)
#g.ax.text(-0.4, 0.62+0.05, '0.62', size='x-small')
#g.ax.text(-0.2, 0.62+0.05, '0.62', size='x-small')
#g.ax.text(0, 0.62+0.05, '0.62', size='x-small')
#g.ax.text(0.2, 0.62+0.05, '0.62', size='x-small')

#g.ax.text(0.6, 0.62+0.05, '0.62', size='x-small')
#g.ax.text(0.8, 0.61+0.05, '0.61', size='x-small')
#g.ax.text(1.0, 0.59+0.05, '0.59', size='x-small')
#g.ax.text(1.2, 0.61+0.05, '0.61', size='x-small')

#g.ax.text(1.6, 0.64+0.05, '0.64', size='x-small')
#g.ax.text(1.8, 0.64+0.05, '0.64', size='x-small')
#g.ax.text(2.0, 0.64+0.05, '0.64', size='x-small')
#g.ax.text(2.2, 0.64+0.05, '0.64', size='x-small')

#g.ax.text(2.6, 0.25+0.05, '0.25', size='x-small')
#g.ax.text(2.8, 0.50+0.05, '0.50', size='x-small')
#g.ax.text(3.0, 0.33+0.05, '0.33', size='x-small')
#g.ax.text(3.2, 0.50+0.05, '0.50', size='x-small')

#g.ax.text(3.6, 0.65+0.05, '0.65', size='x-small')
#g.ax.text(3.8, 0.62+0.05, '0.62', size='x-small')
#g.ax.text(4.0, 0.61+0.05, '0.61', size='x-small')
#g.ax.text(4.2, 0.62+0.05, '0.62', size='x-small')

#g.ax.text(4.6, 0.65+0.05, '0.65', size='x-small')
#g.ax.text(4.8, 0.61+0.05, '0.61', size='x-small')
#g.ax.text(5.0, 0.59+0.05, '0.59', size='x-small')
#g.ax.text(5.2, 0.61+0.05, '0.61', size='x-small')

#g.ax.text(5.6, 0.62+0.05, '0.62', size='x-small')
#g.ax.text(5.8, 0.56+0.05, '0.56', size='x-small')
#g.ax.text(6.0, 0.49+0.05, '0.49', size='x-small')
#g.ax.text(6.2, 0.56+0.05, '0.56', size='x-small')

#g.ax.text(6.6, 0.58+0.05, '0.58', size='x-small')
#g.ax.text(6.8, 0.58+0.05, '0.58', size='x-small')
#g.ax.text(7.0, 0.58+0.05, '0.58', size='x-small')
#g.ax.text(7.2, 0.58+0.05, '0.58', size='x-small')

#g.ax.text(7.6, 0.66+0.05, '0.66', size='x-small')
#g.ax.text(7.8, 0.66+0.05, '0.66', size='x-small')
#g.ax.text(8.0, 0.66+0.05, '0.66', size='x-small')
#g.ax.text(8.2, 0.66+0.05, '0.66', size='x-small')

#g.ax.text(8.6, 0.59+0.05, '0.59', size='x-small')
#g.ax.text(8.8, 0.59+0.05, '0.59', size='x-small')
#g.ax.text(9.0, 0.59+0.05, '0.59', size='x-small')
#g.ax.text(9.2, 0.59+0.05, '0.59', size='small')

#plt.xticks(rotation=70)
#plt.ylim([0.0, 1.0])
#plt.legend(loc='upper right')
#plt.title('SemEval Dataset')
#plt.show()

file3 = pd.read_csv('Harvested_Results.csv')

g = sns.catplot(x='Models', y='Score', hue='Type', data=file3, kind='bar', height=8, aspect=2, legend=False)
g.ax.text(-0.4, 0.73+0.05, '0.73', size='x-small')
g.ax.text(-0.2, 0.73+0.05, '0.73', size='x-small')
g.ax.text(0, 0.73+0.05, '0.73', size='x-small')
g.ax.text(0.2, 0.73+0.05, '0.73', size='x-small')

g.ax.text(0.6, 0.70+0.05, '0.70', size='x-small')
g.ax.text(0.8, 0.70+0.05, '0.70', size='x-small')
g.ax.text(1.0, 0.70+0.05, '0.70', size='x-small')
g.ax.text(1.2, 0.70+0.05, '0.70', size='x-small')

g.ax.text(1.6, 0.73+0.05, '0.73', size='x-small')
g.ax.text(1.8, 0.73+0.05, '0.73', size='x-small')
g.ax.text(2.0, 0.73+0.05, '0.73', size='x-small')
g.ax.text(2.2, 0.73+0.05, '0.73', size='x-small')

g.ax.text(2.6, 0.54+0.05, '0.54', size='x-small')
g.ax.text(2.8, 0.50+0.05, '0.50', size='x-small')
g.ax.text(3.0, 0.35+0.05, '0.35', size='x-small')
g.ax.text(3.2, 0.50+0.05, '0.50', size='x-small')

g.ax.text(3.6, 0.71+0.05, '0.71', size='x-small')
g.ax.text(3.8, 0.71+0.05, '0.71', size='x-small')
g.ax.text(4.0, 0.71+0.05, '0.71', size='x-small')
g.ax.text(4.2, 0.71+0.05, '0.71', size='x-small')

g.ax.text(4.6, 0.73+0.05, '0.73', size='x-small')
g.ax.text(4.8, 0.72+0.05, '0.72', size='x-small')
g.ax.text(5.0, 0.72+0.05, '0.72', size='x-small')
g.ax.text(5.2, 0.73+0.05, '0.73', size='x-small')

g.ax.text(5.6, 0.71+0.05, '0.71', size='x-small')
g.ax.text(5.8, 0.71+0.05, '0.71', size='x-small')
g.ax.text(6.0, 0.71+0.05, '0.71', size='x-small')
g.ax.text(6.2, 0.71+0.05, '0.71', size='x-small')

g.ax.text(6.6, 0.73+0.05, '0.73', size='x-small')
g.ax.text(6.8, 0.73+0.05, '0.73', size='x-small')
g.ax.text(7.0, 0.73+0.05, '0.73', size='x-small')
g.ax.text(7.2, 0.73+0.05, '0.73', size='x-small')

g.ax.text(7.6, 0.76+0.05, '0.76', size='x-small')
g.ax.text(7.8, 0.76+0.05, '0.76', size='x-small')
g.ax.text(8.0, 0.76+0.05, '0.76', size='x-small')
g.ax.text(8.2, 0.76+0.05, '0.76', size='x-small')

g.ax.text(8.6, 0.60+0.05, '0.60', size='x-small')
g.ax.text(8.8, 0.60+0.05, '0.60', size='x-small')
g.ax.text(9.0, 0.60+0.05, '0.60', size='x-small')
g.ax.text(9.2, 0.60+0.05, '0.60', size='small')

plt.xticks(rotation=70)
plt.ylim([0.0, 1.0])
#plt.legend(loc='upper right')
plt.title('Harvested Dataset')
plt.show()


file4 = pd.read_csv('Baseline.csv')
g = sns.catplot(x='Dataset', y='Score', hue='Models', data=file4, kind='bar', aspect=2, legend=False)
plt.legend(loc='upper right')
plt.show()