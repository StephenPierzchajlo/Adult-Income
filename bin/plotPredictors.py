def plotPredictors(data, predictor, width, height):
    '''
    Return a plot with frequency of categorical variables for an inputed predictor.
    data: Input dataframe in pandas format.
    predictor: Name of predictor column, in quotes ("").
    width: Width of plot.
    height: Height of plot.
    '''
    # Set plot size.
    plt.figure(figsize=(width,height))
    plt.title(predictor)
    ax = sns.countplot(x=predictor, data=data)
    for p in ax.patches:
        height = p.get_height()
        return plt.show()