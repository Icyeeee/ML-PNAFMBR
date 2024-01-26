"""
Forth part of the PN-AMFBR project

In this part, we aim to compare the performance of the sythetic datasets with the original dataset using RF

original data set: using average to make up for the 

"""

import os
import pandas as pd
import numpy as np
from collections import OrderedDict
import pprint
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squred_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

class ComparisonAnalysis:

	def __init__(self, file_name_A, file_name_B, random_state = None, n_test = 20):

		self.file_name_A = file_name_A
        self.file_name_B = file_name_B
		self.random_state = random_state '随机数生成种子，确保实验的可重复性'

		self.root_dir = "Data/" '数据文件位于data的文件夹中'
		self.n_test = n_test "确定testing data数量"

		self._load_data()  '调用了_load_data()方法,表明每当firstanalysis类的一个新实例被创建时,都会自动加载数据'

	def _load_data(self, file_name): 'FirstAnalysis里的私有方法,在python里,以_开头的为私有方法,主要供内部使用'

		# Loading the csv file
		df = pd.read_csv(self.root_dir + self.file_name + ".csv", index_col = 0)

		# Slicing the test set from real samples

		test_df = real_df.sample(n = self.n_test, random_state = self.random_state)
		train_df_real = real_df.loc[set(real_df.index) - set(test_df.index)]

		# Shuffling the synth train set
		synth_df = synth_df.sample(frac = 1, random_state = self.random_state) 'frac=1指的是对整个dataframe进行抽样,重新排序'

		# Dropping the RS column
		self.test_df = test_df.drop(columns = "Label", inplace = False)
		self.train_df_real = train_df_real.drop(columns = "Label", inplace = False)
		self.synth_df = synth_df.drop(columns = "Label", inplace = False)

		print ("Data is loaded...")
        return df.drop(columns = "Label", inplace = False)

    def _train_evaluate(self, df):
        X = df.iloc[:, :-1]
        Y = df.iloc[:, -1]

        
        #splitting the dataset into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
        # 这里应该要加一句怎么确定test data是从rdata里随机选的，train data是sytheticdata + rdata剩下的部分

        #Training and evaluationg Linear Regression
        lin_model = LinearRegression()
        lin_model.fit(X_train, Y_train)
        lin_pred = lin_model.predict(X_test)
        lin_mse = mean_squred_error(Y_test, lin_pred)
        lin_mae = mean_absolute_error(Y_test, lin_pred)
        lin_r2 = r2_score(Y_test, lin_pred)

        # Training and evaluating Random Forest
        rf_model = RandomForestRegressor (random_state = self.random_state)
        rf_model.fit(X_train, Y_train)
        rf_pred = rf_model.predict(X_test)
        rf_mse = mean_squred_error(Y_test, rf_pred)
        rf_mae = mean_absolute_error(Y_test, rf_pred)
        rf_r2 = r2_score(Y_test, rf_pred)

        return lin_mse, rf_mse
    
    def compare_datasets(self):
        df_A = self._load_data(self.file_name_A)
        df_B = self._load_data(self.file_name_B)

        lin_mse_A, lin_mae_A, lin_r2_A, rf_mse_A, rf_mae_A, rf_r2_A = self._train_evaluate(df_A)
        lin_mse_B, lin_mae_B, lin_r2_B, rf_mse_B, rf_mae_B, rf_r2_B = self._train_evaluate(df_B)

        print(f"Dataset A - Linear Regression: MSE: {lin_mse_A}, MAE: {lin_mae_A}, R2: {lin_r2_A}")
        print(f"Dataset A - Random Forest: MSE: {rf_mse_A}, MAE: {rf_mae_A}, R2: {rf_r2_A}")
        print(f"Dataset B - Linear Regression: MSE: {lin_mse_B}, MAE: {lin_mae_B}, R2: {lin_r2_B}")
        print(f"Dataset B - Random Forest: MSE: {rf_mse_B}, MAE: {rf_mae_B}, R2: {rf_r2_B}")

# Example of how to use the class

if __name__ == "__main__":
    analysis = ComparisonAnalysis(file_name_A='dataset_A', file_name_B='dataset_B')
    analysis.compare_datasets()


# 	def _get_regression_results(self, y_true, y_pred): '定义一个方法来计算真实值和预测值之间的相关系数'
# 		return np.corrcoef(y_true, y_pred)[0][1] "返回ytrue和ypred之间的相关系数,np.corrcoef 返回一个相关系数矩阵，其中 [0][1] 是真实值和预测值之间的相关系数。"

# 	def run(self, N = 5): '用于执行实验'

# 		lin_errs, rf_errs = [], [] '用于存储线性回归和随机森林模型的错误率'
# 		percentages = np.linspace(0, 1, N)

# 		X_test, Y_test = self.test_df.iloc[:, :-1], self.test_df.iloc[:, -1]

# 		for perc in percentages:
# 			print (perc, 'percentage is about to analyzed in')

# 			# Getting the train set X and Y
# 			idx = int(perc * len(self.synth_df))
# 			temp_df = pd.concat([self.train_df_real, self.synth_df.iloc[:idx, :]])
# 			X_train, Y_train =  temp_df.iloc[:, :-1], temp_df.iloc[:, -1]

# 			# Training linear model
# 			lin_model = LinearRegression(fit_intercept = True, normalize=False)
# 			lin_model.fit(X_train, Y_train)

# 			# Getting the linear regression erros
# 			lin_test_err = self._get_regression_results(Y_test, lin_model.predict(X_test))
# 			lin_errs.append(lin_test_err)

# 			# Training random forest model
# 			rf_model = RandomForestRegressor(
# 							n_estimators = 20, 
# 							max_depth = 5,
# 							min_samples_split = 2,
# 							min_samples_leaf = 1,
# 							max_features = 'auto',
# 							random_state = self.random_state,
# 							n_jobs = -1,
# 							verbose = 1
# 							)
# 			rf_model.fit(X_train, Y_train)

# 			# Getting the random forest regression erros
# 			rf_test_err = self._get_regression_results(Y_test, rf_model.predict(X_test))
# 			rf_errs.append(rf_test_err)

# 		df = pd.DataFrame()
# 		df['Linear'] = lin_errs
# 		df['RF'] = rf_errs
# 		# df['rn_state'] = [self.random_state for _ in percentages]
# 		df.to_csv(f"Task1/Task1-{self.file_name}.csv")


# 		plt.plot(percentages, lin_errs, label = 'Linear')
# 		plt.plot(percentages, rf_errs, label = 'RandomForest')
# 		plt.xlabel("Percentage of train set")
# 		plt.ylabel("Correlation Coefficient")
# 		plt.title(f"{self.file_name}-{self.random_state}")
# 		plt.legend()
# 		plt.grid()
# 		plt.savefig(f"Task1/Task1-{self.file_name}.png")
# 		plt.show()

# if __name__ == "__main__": '脚本的入口点,用于创建实例并执行实验'

# 	rn_state = int(np.random.random()*1000)
# 	rn_state = 289

# 	print (f"\n\nrn_state:{rn_state}\n\n")

# 	myAnaysis = FirstAnalysis(file_name = 'ammonium22',
# 						random_state = rn_state,
# 						n_test = 20)
# 	myAnaysis.run(N = 11)
