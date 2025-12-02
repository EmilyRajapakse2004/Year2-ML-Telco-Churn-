from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def train_decision_tree(X_train, y_train, max_depth=5):
    """
    Train a Decision Tree classifier.
    """
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt.fit(X_train, y_train)
    return dt

def train_neural_network(X_train, y_train, epochs=50, batch_size=32):
    """
    Train a Neural Network classifier.
    Returns the trained model and training history.
    """
    nn = Sequential([
        Dense(32, input_dim=X_train.shape[1], activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = nn.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=0)
    return nn, history
