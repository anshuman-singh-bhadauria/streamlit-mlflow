import streamlit as st
import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
import mlflow.sklearn
import streamlit as st 
import numpy as np 

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score


def main():
    mlflow.set_tracking_uri("http://localhost:5000/")
    st.write(mlflow.get_tracking_uri())
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.title('Streamlit Example')

    st.write("""
    # Explore different classifier and datasets
    Which one is the best?
    """)

    create_experiment = st.checkbox("Create experiment")
    if not create_experiment:
        return

    experiment_name=st.text_input("Experiment name")
    if not len(experiment_name)>0:
        return
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        st.write("Created experiment: ",experiment_name, " with experiment id: ",experiment_id)
    except Exception:
        st.warning("Experiment: "+experiment_name+ " already exists.")
        mlflow.set_experiment(experiment_name)

    

    with mlflow.start_run():
        
        dataset_name = st.sidebar.selectbox(
            'Select Dataset',
            ('Iris', 'Breast Cancer', 'Wine')
        )
        
        st.write(f"## {dataset_name} Dataset")

        classifier_name = st.sidebar.selectbox(
            'Select classifier',
            ('KNN', 'SVM', 'Random Forest')
        )

        def get_dataset(name):
            data = None
            if name == 'Iris':
                data = datasets.load_iris()
            elif name == 'Wine':
                data = datasets.load_wine()
            else:
                data = datasets.load_breast_cancer()
            X = data.data
            y = data.target
            return X, y

        X, y = get_dataset(dataset_name)
        st.write('Shape of dataset:', X.shape)
        st.write('number of classes:', len(np.unique(y)))
        mlflow.log_param("Dataset",dataset_name)
        mlflow.log_param("Shape_of_dataset",X.shape)
        mlflow.log_param("number_of_classes",len(np.unique(y)))

        def add_parameter_ui(clf_name):
            params = dict()
            if clf_name == 'SVM':
                C = st.sidebar.slider('C', 0.01, 10.0)
                params['C'] = C
            elif clf_name == 'KNN':
                K = st.sidebar.slider('K', 1, 15)
                params['K'] = K
            else:
                max_depth = st.sidebar.slider('max_depth', 2, 15)
                params['max_depth'] = max_depth
                n_estimators = st.sidebar.slider('n_estimators', 1, 100)
                params['n_estimators'] = n_estimators
            return params

        params = add_parameter_ui(classifier_name)

        def get_classifier(clf_name, params):
            clf = None
            if clf_name == 'SVM':
                clf = SVC(C=params['C'])
            elif clf_name == 'KNN':
                clf = KNeighborsClassifier(n_neighbors=params['K'])
            else:
                clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
                    max_depth=params['max_depth'], random_state=1234)
            return clf

        clf = get_classifier(classifier_name, params)
        mlflow.log_param("Algorithm_name",classifier_name)
        #### CLASSIFICATION ####

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
        train_checkbox = st.checkbox("Train model")
        if not train_checkbox:
            return

        clf.fit(X_train, y_train)
        
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="sklearn-model",
            registered_model_name="sk-learn-"+dataset_name+"-"+classifier_name
        )
        mlflow.log_params(clf.get_params(deep=True))
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("Accuracy_score", acc)
        
        st.write(f'Classifier = {classifier_name}')
        st.write(f'Accuracy =', acc)
        st.write("Model saved in run: ",mlflow.active_run().info.run_uuid)

    #### PLOT DATASET ####
    # Project the data onto the 2 primary principal components
    pca = PCA(2)
    X_projected = pca.fit_transform(X)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    fig = plt.figure()
    plt.scatter(x1, x2,
            c=y, alpha=0.8,
            cmap='viridis')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()

    #plt.show()
    st.pyplot()

    

    

    

if __name__=="__main__":
    main()

