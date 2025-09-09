import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import pickle

data = pd.read_csv('WineQT.csv')

X = data.drop('quality', axis=1)
y = data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def run_experiment(model_name, model, model_params):
    with mlflow.start_run():
        model.fit(X_train, y_train)
        model_predict = model.predict(X_test)

        accuracy = accuracy_score(y_test, model_predict)
        precision = precision_score(y_test, model_predict, average='weighted')
        recall = recall_score(y_test, model_predict, average='weighted')
        f1 = f1_score(y_test, model_predict, average='weighted')

        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('f1', f1)

        for param_name, param_value in model_params.items():
            mlflow.log_param(param_name, param_value)

        mlflow.log_param('train_size', len(X_train))
        mlflow.log_param('test_size', len(X_test))

        mlflow.sklearn.log_model(model, 'model')

        with open(f'models/{model_name}.pkl', 'wb') as f:
            pickle.dump(model, f)


mlflow.set_experiment('First Experiment - SVC')

model1 = SVC(random_state=42)
model1_params = {'model_type': 'SVC',
                 'random_state': 42}
run_experiment('SVC', model1, model1_params)


mlflow.set_experiment('Second Experiment - Boosting')

model2 = AdaBoostClassifier(random_state=42)
model2_params = {'model_type': 'AdaBoost Classifier',
                 'random_state': 42}
run_experiment('AdaBoostClassifier', model2, model2_params)

model3 = GradientBoostingClassifier(random_state=42)
model3_params = {'model_type': 'Gradient Boosting Classifier',
                 'random_state': 42}
run_experiment('GradientBoostingClassifier', model3, model3_params)


mlflow.set_experiment('Third Experiment - Random Forest Classifier')

model4 = RandomForestClassifier(random_state=42)
model4_params = {'model_type': 'Random Forest Classifier',
                 'random_state': 42}
run_experiment('RandomForestClassifier1', model4, model4_params)

model5 = RandomForestClassifier(random_state=42)
model5_params = {'model_type': 'Random Forest Classifier',
                 'random_state': 42,
                 'n_estimators': 128,
                 'max_depth': 150,
                 'min_samples_split': 2}
run_experiment('RandomForestClassifier2', model5, model5_params)