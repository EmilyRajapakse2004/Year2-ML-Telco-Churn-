from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def train_decision_tree(X_train, y_train):
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    return dt_model

def build_neural_network(input_dim):
    nn_model = Sequential()
    nn_model.add(Dense(32, input_dim=input_dim, activation='relu'))
    nn_model.add(Dropout(0.2))
    nn_model.add(Dense(16, activation='relu'))
    nn_model.add(Dense(1, activation='sigmoid'))
    nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return nn_model
