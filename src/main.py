import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import copy

def random_forest_cross_validation(processed_dataset_df):
    clf = RandomForestClassifier(n_estimators=55, max_depth=5, random_state=0)
    target = 'HeartDisease'
    features = processed_dataset_df.columns.to_list() 
    features.remove(target)
    #print(heart_failure_df[target])
    X = processed_dataset_df[features].values
    y = processed_dataset_df[target].values
    kf = KFold(n_splits=6)
    result = cross_val_score(clf, X, y, cv = kf)
    clf.fit(X,y)
    print("K-Fold (R^2) Scores: {0}".format(result))
    print("Mean R^2 for Cross-Validation K-Fold: {0}".format(result.mean()))

def process_as_one_hot_enconding(dataframe):
    #make dummies columns for categorical values (one hot encoding)
    one_hot_encoded_df = pd.get_dummies(dataframe)
    #the one-hot enconding with get_dummies results in true and false values instead of 1 and 0
    #I will convert them to integers
    bool_cols=one_hot_encoded_df.select_dtypes("bool").columns.to_list() #take all columns that have boolean type
    one_hot_encoded_df[bool_cols] = one_hot_encoded_df[bool_cols].astype(int) #convert
    return one_hot_encoded_df

def process_string_categorical_data_into_integers(dataframe):
    label_encoder = LabelEncoder()
    processed_df = copy.deepcopy(dataframe)
    #transform each column that has string values
    for column in dataframe.columns.to_list():
        if isinstance(dataframe[column].values[0], str):
            processed_df[column] = label_encoder.fit_transform(dataframe[column])
    return processed_df

def normalize_minmax_scaling(dataframe):
    min_max_scaler = preprocessing.MinMaxScaler()
    normalized = min_max_scaler.fit_transform(dataframe)
    normalized = pd.DataFrame(normalized,columns=dataframe.columns.to_list())
    return normalized
def main():
    heart_failure_df = pd.read_csv('./dataset/heart.csv')

    #PROCESS CATEGORICAL VALUES (one hot encoding)
    one_hot_encoded_heart_failure_df = process_as_one_hot_enconding(heart_failure_df)
    #WITHOUT ONE HOT ENCODING
    processed_heart_failure_df=process_string_categorical_data_into_integers(heart_failure_df)

    #REMOVE OUTLIERS 

    #NORMALIZATION - No need for random forest, knn needs to be normalized
    normalized_df = normalize_minmax_scaling(processed_heart_failure_df)
    #RANDOM FOREST
    random_forest_cross_validation(processed_heart_failure_df)
    random_forest_cross_validation(one_hot_encoded_heart_failure_df)
    random_forest_cross_validation(normalized_df)
    #KNN



main()
