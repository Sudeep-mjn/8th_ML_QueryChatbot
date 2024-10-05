import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import random
import matplotlib.pyplot as plt

# Initialize NLTK and download required resources
nltk.download('punkt')
nltk.download('wordnet')

# Initialize WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents from JSON file
with open('data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# Iterate through each intent in the JSON file
for intent in data['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the pattern
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add documents in the corpus
        documents.append((w, intent['tag']))
        # Add to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, convert to lowercase, and remove duplicates from words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

# Print statistics about the corpus
print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

# Save words and classes to pickle files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

# Generate bag of words for each sentence in documents
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    # Create bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # Output is '1' for current tag and '0' for others
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle training data and convert to numpy array
random.shuffle(training)
training = np.array(training, dtype=object)  # Use dtype=object to avoid VisibleDeprecationWarning

# Separate features and labels
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Print confirmation
print("Training data created")

# Define the model architecture
model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))  # Increased neurons
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))  # Reduced dropout rate
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model using Adam optimizer
adam = Adam(learning_rate=0.001)  # Use Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# Train model and save it
history = model.fit(train_x, train_y, epochs=300, batch_size=8, verbose=1,
                    validation_split=0.2, callbacks=[early_stopping, lr_scheduler])
model.save('model.h5')

print("Model created")

# Evaluate the model on the training set (you can replace train_x and train_y with test data if available)
loss, accuracy = model.evaluate(train_x, train_y, verbose=1)

print(f"Model Loss: {loss:.4f}")
print(f"Model Accuracy: {accuracy:.4f}")

# Plotting training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plotting training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()