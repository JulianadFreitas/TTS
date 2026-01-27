import pandas

dataframe = pandas.read_csv('reasons_for_exclusion.csv')
languages = dataframe.groupby(['language', 'reason']).count()
languages.to_csv('reasons_for_exclusion_grouped.csv')
