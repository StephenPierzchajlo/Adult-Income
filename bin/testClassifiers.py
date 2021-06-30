def testClassifiers(classifierList, X_train, y_train, X_vl, y_val):
    '''
    Return a dataframe with 1 row for each classifier inputed in the function's
    arguements. Each row contains: Classifier name, accuracy, recall score, and
    precision score.
    data: Input: classifierList, X_train, y_train, X_test, y_test. 
    classifierList: List of classifiers you want to test. 
    Example: [DecisionTreeClassifier, KNeighborsClassifier, GaussianNB].
    X_train: Matrix of training predictors (numeric).
    y_train: Vector of training response variable identity (numeric).
    X_test: Matrix of testing predictors (numeric).
    y_test: Vector of testing response variable identity (numeric).
    '''
    # Make empty dataframe.
    model_df = pd.DataFrame(columns=['Classifier', 'Accuracy', 'Recall', 'Precision', 'f1'])

    # Repeat for each classifier in the list defined in the functions arguement.
    for i in classifierList:
        
        # For each model:
        model = i()
        
        # Get model name from current classifier.
        model_name = type(model).__name__
        
        # Fit the data with the training model.
        model.fit(X_train, y_train)
        
        # make predictions on test data using training model.
        yhat = model.predict(X_val)
        
        # test accuracy for current model.
        acc = accuracy_score(y_val, yhat)
        
        # test recall for current model
        recall = recall_score(y_val, yhat)
        
        # test precision for current model.
        precision = precision_score(y_val, yhat)
        
        f1 = f1_score(y_val, yhat)
    
        # Create list for model i with caculated information.
        row = [model_name, acc, recall, precision, f1]
        
        # Add list as new row in model_df.
        model_df.loc[len(model_df)] = row
    
    # Return dataframe with performance of each model.
    return model_df