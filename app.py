import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
from flask import Flask, render_template, request
import pyttsx3  # Importing the text-to-speech library
import threading  # Import threading for speech synthesis
import difflib  # For finding close matches

# Initialize Text-to-Speech engine globally
engine = pyttsx3.init()
is_speaking = False  # Track if the engine is currently speaking

# Load your model
model = load_model('model.h5')

# Load your intents JSON file with UTF-8 encoding
with open('data.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Load your words and classes pickled files
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))  # Load classes

lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)  
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    if ints:
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
    else:
        result = "I'm sorry, I didn't understand that. Can you please try again?"
    return result

def speak(text):
    global is_speaking
    is_speaking = True
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 0)   # Volume level (0.0 to 1.0)
    engine.say(text)
    threading.Thread(target=engine.runAndWait).start()
    is_speaking = False

def chatbot_response(msg):
    global is_speaking
    
    # Stop current speech if it's speaking
    if is_speaking:
        engine.stop()

    ints = predict_class(msg, model)
    res = getResponse(ints, intents)

    # Start a new thread for speech synthesis
    threading.Thread(target=speak, args=(res,)).start()

    return res

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")


# Load quiz questions from JSON file
def load_quiz_questions():
    with open('quiz_data.json', 'r') as f:
        return json.load(f)

# Sample quiz questions
quiz_questions = load_quiz_questions()

@app.route("/quiz_index")                                                  #Quizindex
def quizIndex():
    return render_template("quiz_index.html")


@app.route('/quiz', methods=['GET'])
def quiz():
    # Randomly select 10 questions
    selected_questions = random.sample(quiz_questions, 10)
    return render_template('quiz.html', questions=selected_questions)

@app.route("/submit_quiz", methods=["POST"])
def submit_quiz():
    """Handle quiz submission and show results."""
    correct_count = 0  # Initialize correct answers counter
    responses = []

    for i in range(10):  # Assuming 10 questions
        question = request.form.get(f'question_{i}')  # Get the question text
        user_answer = request.form.get(f'answers_{i}')  # Get the selected answer
        
        # Find the corresponding question in the original list
        for question_data in quiz_questions:
            if question_data['question'] == question:
                correct_answer = question_data['answer']
                responses.append({
                    'question': question_data['question'],
                    'user_answer': user_answer,
                    'correct_answer': correct_answer
                })
                # Increment correct count if the user's answer is correct
                if user_answer == correct_answer:
                    correct_count += 1
                break

    return render_template("quiz_results.html", responses=responses, correct_count=correct_count)

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    
    # Debugging output to verify received user input
    print(f"User input received: {userText}")
    
    return chatbot_response(userText)

if __name__ == "__main__":
    app.run(debug=True)  # Enable debug mode for better error messages.
