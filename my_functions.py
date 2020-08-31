import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score

# ignore the annoying setting copy warning:
pd.options.mode.chained_assignment = None

def plot_stat_summary(csv_filename,plots_by_row=False,plots_by_treatment=True):
	dataset = pd.read_csv(csv_filename)
	
	dataset_with_efficacy = get_efficacy(dataset)
	
	# Check that the efficacy across rows is relatively constant
	# Do not want to get tricked by yield which can be impacted by field irregularities
	
	# average efficacy across rows:
	#print("statistics grouped by row for treatment efficacy")
	#print("------------------------------------------------")
	stats = dataset_with_efficacy.groupby(by='row').describe()
	stats_by_row = stats['efficacy_in_percent']
	stats_by_row['10%'] = dataset_with_efficacy[['efficacy_in_percent','row']].groupby(by='row').quantile(q=0.10)
	stats_by_row['90%'] = dataset_with_efficacy[['efficacy_in_percent','row']].groupby(by='row').quantile(q=0.90)
	#print(stats_by_row)

	if plots_by_row:
		fig = plt.figure(figsize=(16,6))

		for r in dataset_with_efficacy['row'].unique():
			plt.plot(dataset_with_efficacy['range'].loc[dataset_with_efficacy['row']==r],\
				dataset_with_efficacy['efficacy_in_percent'].loc[dataset_with_efficacy['row']==r])
	
		plt.title('efficacy per row')
		plt.xlabel('range')
		plt.ylabel('efficacy')
	
	#print("statistics grouped by treatment for efficacy")
	#print("------------------------------------------------")
	stats = dataset_with_efficacy.groupby(by='sym').describe()
	stats_by_treat = stats['efficacy_in_percent']
	stats_by_treat['10%'] = dataset_with_efficacy[['efficacy_in_percent','sym']].groupby(by='sym').quantile(q=0.10)
	stats_by_treat['90%'] = dataset_with_efficacy[['efficacy_in_percent','sym']].groupby(by='sym').quantile(q=0.90)
	#print(stats_by_treat)
	
	if plots_by_treatment:
		fig = plt.figure(figsize=(16,6))
		sorted_index = dataset_with_efficacy[['efficacy_in_percent','sym']].groupby(by='sym').\
				median().sort_values(by='efficacy_in_percent',ascending=False).index	
		ax = sns.boxplot(x='sym',y='efficacy_in_percent',data=dataset_with_efficacy,order=sorted_index)
		ax.set_ylim(stats_by_treat['10%'].min(),stats_by_treat['90%'].max())
		ax.set_title('efficacy per treatment')

	return stats_by_row, stats_by_treat, dataset_with_efficacy

def get_efficacy(dataset_as_dataframe):

	data = dataset_as_dataframe;

	# Getting the control positions:
	ctrlrows = data['row'].loc[data['sym']=='ctrl'].values
	ctrlcols = data['range'].loc[data['sym']=='ctrl'].values
	data['efficacy_in_percent'] = data['value']
        
    # Because there can be heterogeneous fields, I am going to take "nearest control" rather than control per row.
    # Units of distance are in plot position
	ctrlvalue = []
	for i in range(0,data['row'].max()+1):
		for j in range(0,data['range'].max()+1):
			dist = []
			for k in range(0,len(ctrlrows)):
				dist.append(((ctrlrows[k]-i)**2+(ctrlcols[k]-j)**2)**0.5)
			ctrl_row_pos = ctrlrows[dist==np.min(dist)][0]
			ctrl_col_pos = ctrlcols[dist==np.min(dist)][0]
			x=data['value'].loc[(data['row']==ctrl_row_pos)&(data['range']==ctrl_col_pos)]
			ctrlvalue.append(x.values[0])
	data['efficacy_in_percent'] = (data['value']-ctrlvalue)/ctrlvalue * 100.
	data['efficacy_in_percent'].loc[data['efficacy_in_percent']==0]=None

	return data

def linearRegression(dataframe_with_efficacy,efficacy=True,normalize=False,include_position=False,testsize=0.25):

	df = dataframe_with_efficacy.copy()
    
	df = df.drop(columns=['rep','field_id'])
    
	if include_position:
		pass
	else:
		df = df.drop(columns=['row','range'])
        
	df['sym'] = df['sym'].astype('category')
	df = pd.get_dummies(df)
	df = df.loc[df['sym_ctrl']==False]  
	df = df.drop(columns='sym_ctrl')
    
	if efficacy:
		target = 'efficacy_in_percent'
		df = df.drop(columns=['value'])
	else:
		target = 'value'
		df = df.drop(columns=['efficacy_in_percent'])
        
	features = df.drop(columns=target).columns
    
	# Make training set. Leave out 25% data for testing.
    
	# create multiple linear regression object
	mlr = LinearRegression(fit_intercept=True)
    
	# Whether or not to normalize:
	mlr.normalize=normalize
    
	# Separate into 75% train and 25% test:
	# Test size is default 25%
	x_train, x_test, y_train, y_test = train_test_split(df[features], df[target], test_size=testsize, shuffle=True)

	# fit linear regression
	mlr.fit(x_train, y_train)

	# get the slope and intercept of the line best fit.
	# print(mlr.intercept_)

	print('features in order of decreasing value of coeficients:')
	print('feature: coefficient value; target: ',target)
	print('--------------------------')

	sorted_idx = np.argsort(mlr.coef_)
	f=features[sorted_idx[::-1]]
	c=mlr.coef_[sorted_idx[::-1]]
	for cidx, ff in enumerate(f):
		print(ff,": ",c[cidx])
        
	# Run the model on the test set, plot the comparison.
	y_prediction = mlr.predict(x_test)
	rmse_model = (np.mean(y_prediction-y_test)**2)**0.5
    
	cv4 = cross_val_score(mlr, df[features], df[target], cv=4, scoring="neg_mean_squared_error")
	rmse_model_cross_fold_4 = (np.mean(np.sign(cv4)*cv4))**0.5

	print('')
	print('')
	print('--------------------------------------')
	print('rmse_for_cross_val_four_times: ',np.round(rmse_model_cross_fold_4,2))
	print('--------------------------------------')
	print('')
	print('')
	print('')
	print('Comparing predicted vs truth value in dataset')
	print('---------------------------------------------------------')

	fig = plt.figure(figsize=(6,6))
	plt.scatter(y_test,y_prediction)
	plt.title('predicted vs test: RMSE = %s [units]' % (np.round(rmse_model,2)))
	plt.ylabel('predicted value for %s' % (target))
	plt.xlabel('test value for %s' % (target))
    
	return mlr


























