import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def modelTrain(csv_file='canDonate.csv'):
    df = pd.read_csv(csv_file)

    gender_encoder = LabelEncoder()
    df['Gender'] = gender_encoder.fit_transform(df['Gender'])

    selected_columns = ['Age','Gender','Weight_kg','Hemoglobin_g_dL','Num_Past_Donations','Months_Since_Last_Donation']
    X = df[selected_columns]
    y = df['Eligible']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    return model, selected_columns, gender_encoder

def predictor(input_list, model, selected_columns, gender_encoder):
    input_list[1] = gender_encoder.transform([input_list[1]])[0]
    
    input_df = pd.DataFrame([input_list], columns=selected_columns)
    pred = model.predict(input_df)
    return pred[0]