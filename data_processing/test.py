from sklearn import linear_model
regr = linear_model.LinearRegression()
regr = linear_model.LinearRegression()
time = [[1,1,2]]
con = [[1,2,3,]]
regr.fit(time, con)
pred = regr.predict(time)
