from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from .data_prosessing import process_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    file_path = "./data/sample_aqi_data.csv"
    X_train, X_test, y_train, y_test = process_data(file_path)
    
    model = train_model(X_train, y_train)
    
    evaluate_model(model, X_test, y_test)