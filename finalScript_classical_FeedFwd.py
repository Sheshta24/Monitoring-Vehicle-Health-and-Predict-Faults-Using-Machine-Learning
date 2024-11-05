import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt

start_time = dt.datetime.now()
print('Start learning at {}'.format(str(start_time)))

# Load the dataset from the provided CSV file path
data_path ='/Users/sheshta/Library/Mobile Documents/com~apple~CloudDocs/Term 2 MSc/AI for Eng Design/Project 2/EngineFaultDB/Engine Only/EngineFaultDB_Final.csv'
df = pd.read_csv(data_path)

# Set up the matplotlib figure for the heatmap
fig_heatmap, ax_heatmap = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap for the heatmap
colormap = sns.diverging_palette(220, 10, as_cmap=True)

# Calculate and draw the heatmap with annotations for each cell
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap=colormap, vmax=1, vmin=-1,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.tight_layout()

# Set up the matplotlib figure for the box plots
fig_boxplot, axes_boxplot = plt.subplots(5, 3, figsize=(15, 20))  # Adjust for the correct layout
axes_boxplot = axes_boxplot.flatten()  # Flatten the axes array for easy iteration

# List of column names to create box plots for, excluding 'Fault'
columns_to_plot = ['MAP', 'TPS', 'Force', 'Power', 'RPM', 'Consumption L/H',
                   'Consumption L/100KM', 'Speed', 'CO', 'HC', 'CO2', 'O2', 'Lambda', 'AFR']

# Create box plots for each column
for i, column in enumerate(columns_to_plot):
    sns.boxplot(x='Fault', y=column, data=df, palette="Set3", ax=axes_boxplot[i])
    axes_boxplot[i].set_title(column)
    axes_boxplot[i].set_xlabel('')
    axes_boxplot[i].set_ylabel('')

# Remove any unused subplots
for j in range(i + 1, len(axes_boxplot)):
    fig_boxplot.delaxes(axes_boxplot[j])

plt.tight_layout()

# Function to preprocess data
def preprocess_data(df, correlation_threshold=0.1):
    corr_matrix = df.corr()
    high_corr_features = corr_matrix.index[corr_matrix['Fault'].abs() > correlation_threshold].tolist()
    high_corr_features.remove('Fault')
    X = df[high_corr_features]
    y = df['Fault']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_scaled, y)
    return X_smote, y_smote

def plot_fault_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Fault', data=df, palette='pastel')
    plt.title('Distribution of Fault Types in EngineFaultDB')
    plt.xlabel('Fault Type')
    plt.ylabel('Count')
    plt.tight_layout()  # Adjust layout to make room for the plot
    plt.show()

# Split data
def split_data(X, y, for_nn=False):
    if not for_nn:
        return train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        y_encoded = to_categorical(y)
        return train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train machine learning models
def train_ml_model(model, params, X_train, y_train):
    random_search = RandomizedSearchCV(model, params, n_iter=10, cv=5, random_state=42)
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_

# Evaluate models
def evaluate_model(model, X_test, y_test, model_name, for_nn=False):
    if not for_nn:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    else:
        _, accuracy = model.evaluate(X_test, y_test, verbose=0)
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        print(classification_report(y_test_classes, y_pred_classes))
    
    print(f"{model_name} Test Accuracy: {accuracy}")
    return accuracy, y_pred

# Plot confusion matrix
def plot_confusion_matrix(y_test, y_pred, model_name):
    if len(y_pred.shape) > 1:  # For neural network predictions
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} Confusion Matrix")
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Create neural network
def create_neural_network(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Plot training history
def plot_training_history(history, title):
    plt.figure(figsize=(12, 6))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'{title} Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'{title} Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    
    plot_fault_distribution(df)
    
    X_smote, y_smote = preprocess_data(df, correlation_threshold=0.1)
    
    # Train and evaluate RandomForest
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = split_data(X_smote, y_smote)
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_rf, y_train_rf)
    accuracy_rf, y_pred_rf = evaluate_model(rf_model, X_test_rf, y_test_rf, "RandomForest")
    plot_confusion_matrix(y_test_rf, y_pred_rf, "RandomForest")

    # Train and evaluate SVM
    X_train_svm, X_test_svm, y_train_svm, y_test_svm = split_data(X_smote, y_smote)
    svm_model = SVC(random_state=42, probability=True)
    svm_model.fit(X_train_svm, y_train_svm)
    accuracy_svm, y_pred_svm = evaluate_model(svm_model, X_test_svm, y_test_svm, "SVM")
    plot_confusion_matrix(y_test_svm, y_pred_svm, "SVM")

    # Train and evaluate KNN
    X_train_knn, X_test_knn, y_train_knn, y_test_knn = split_data(X_smote, y_smote)
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train_knn, y_train_knn)
    accuracy_knn, y_pred_knn = evaluate_model(knn_model, X_test_knn, y_test_knn, "KNN")
    plot_confusion_matrix(y_test_knn, y_pred_knn, "KNN")

    # Train and evaluate Neural Network
    X_train_nn, X_test_nn, y_train_nn, y_test_nn = split_data(X_smote, y_smote, for_nn=True)
    nn_model = create_neural_network(X_train_nn.shape[1], y_train_nn.shape[1])
    history = nn_model.fit(X_train_nn, y_train_nn, validation_split=0.1, epochs=100, batch_size=32, verbose=1)
    nn_accuracy, _ = evaluate_model(nn_model, X_test_nn, y_test_nn, "Feed Forward Neural Network", for_nn=True)
    plot_training_history(history, "Feed Forward Neural Network")
    plot_confusion_matrix(y_test_nn, nn_model.predict(X_test_nn), "Feed Forward Neural Network")

    # Plot comparison
    accuracies = {
        'RandomForest': accuracy_rf,
        'SVM': accuracy_svm,
        'KNN': accuracy_knn,
        'Feed Forward Neural Network': nn_accuracy
    }
    plt.figure(figsize=(10, 7))
    plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'red', 'purple'])
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.show()
    
    end_time = dt.datetime.now() 
    print('Stop learning {}'.format(str(end_time)))
    elapsed_time= end_time - start_time
    print('Elapsed learning {}'.format(str(elapsed_time)))

if __name__ == "__main__":
    main()