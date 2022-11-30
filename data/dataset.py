import pandas as pd
import numpy as np

raw_data = pd.read_excel('data/data.xlsx', names=['Title', 'Year', 'Deaths', 'Duration', 'Rating'])

# filter out all the NaN values, caused by bad input file formatting
currentMovieTitle = ''
breakFlag = False
# fill in all the title cell values
for i, row in raw_data.iterrows():
    if not pd.isna(row['Title']) and pd.isna(row['Year']):
        currentMovieTitle += row['Title'] + ' '
        breakFlag = True

    if breakFlag and not pd.isna(row['Year']):
        raw_data.at[i, 'Title'] = currentMovieTitle

        currentMovieTitle = ''
        breakFlag = False
# leave out all the NaN rows
data = raw_data[raw_data['Year'].notnull()].copy()

# apply normalization
MIN_YEAR = data['Year'].min()
data['Year'] = data['Year'] - MIN_YEAR + 1

# export:
# calculate number of data samples (observations)
N = data.shape[0]

# extract factors (x1, x2, x3) and observations (y) values
numerical_data = data.loc[:, 'Year':'Duration'].to_numpy()
M = numerical_data.shape[1] # factors count

y = data.loc[:, 'Rating'].to_numpy()

if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    np.set_printoptions(suppress=True)

    print('Dataset: ', data, sep='\n')
    print('Y (observations) values: ', y, sep='\n')