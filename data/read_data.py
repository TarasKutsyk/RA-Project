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
data['Year'] = data['Year'] - 2000

# export:
# calculate number of data samples (observations)
N = data.shape[0]

# construct design matrix (F) and observations (y) arrays
numerical_data = data.loc[:, 'Year':'Duration'].to_numpy()
y = data.loc[:, 'Rating'].to_numpy()

F = np.concatenate([np.ones((N, 1)), numerical_data], axis=1)

if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    np.set_printoptions(suppress=True)

    print('Dataset: ', data, sep='\n')
    print('Design matrix: ', F, sep='\n')
    print('Y (observations) values: ', y, sep='\n')
