class modelCompile:
    def __init__(self, pred_var, targ_var, scaled):
        '''
        Class initializes self with predictor features and target variable

        pred_var contains dataframe of independent/predictor features 
        targ_var is the target variable
        scaled: Takes in boolean value. Scales the parameters if True and doesn't scale if False
        '''
        X = pred_var.values
        y = targ_var
        if scaled == True:
            sca = MinMaxScaler()
            X   = sca.fit_transform(X)
            self.X = X
        elif scaled == False:
            self.X = X
        self.y = y


    def modBuilder(self, mod, size, r_s):

        '''
        Function defines the model and splits the data set into train and test.

        mod: Input is the current model being passed in which could be RandomForest, 
        DecisionTree, Log/linear or any kind. Model parameters are chosen in the function.
        size: The prefered percentage of test size out of the whole dataset.
        r_s: random state set. 
        '''
        # X and y arrays are returned from init class function
        X = self.X
        y = self.y
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=r_s, test_size=size)
        y_train = np.ravel(y_train)
        self.f_mod = mod.fit(X_train, y_train)
        self.y_test = y_test
    
        return X_train, X_test


    def pred_(self, test, reg):
        '''
        Function predicts the xtest using the current model selected

        test: specifies the split train test dataset from the mod builder function
        reg: if True, rounds up the predicted value to the nearest whole number, 
        if False, appends the predicted whole numbers directly
        '''
        #Model is returned from the modbuilder function. Reused to predict test set
        f_mod = self.f_mod
        pred = f_mod.predict(test)
        eval_ = []
        for i in pred:      
            if reg == True:
                yhat = int(round(i))
                eval_.append(yhat)
            else:
                eval_.append(i)

        self.eval_ = eval_

        return eval_
    

    def yhatPred(self, test, numbers):
        '''
        This function prints the ytest and ypred side by side in a dataframe format.

        test: The test values that were predicted. 
        numbers: numbers of preview loaded from the table
        '''
        tab = pd.DataFrame({'Actual': test, 'Predicted' : self.eval_})
        tab = tab.head(numbers)

        return tab

    
    def cf_(self, cf, predictors):
        '''
        Function plots a bar graph containing coefficients/importance against the feature names

        cf: string of 'coefficient' or 'importance' since our model will choose between model.coef_ or model.feature_importances
        predictors: a list containing independent features fit in to the model
        '''
        mod = self.f_mod
        if cf == 'coefficient':
            coefficients  = pd.DataFrame(mod.coef_.ravel())
        elif cf == 'importance':
            coefficients  = pd.DataFrame(mod.feature_importances_)
 
        coef_df     = pd.DataFrame(predictors)
        coef_sumry  = pd.merge(coefficients, coef_df, left_index= True,
                              right_index= True, how = 'left')
        coef_sumry.columns = ['coefficients','features']
        coef_sumry  = coef_sumry.sort_values(by = 'coefficients', ascending = False)
        plt.figure(figsize=(20,10))
        chart = sns.barplot(x=coef_sumry['features'], y=coef_sumry['coefficients'], palette='Set1')
        chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
        plt.show()


    def accuracy_(self):
        '''
        This function returns the value of RMSE of the model fitted in the class function
        '''

        mse = mean_squared_error(self.y_test, self.eval_)
        rmse = np.sqrt(mse)
        
        return rmse
