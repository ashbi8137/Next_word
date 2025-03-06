import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Sample dataset (small text corpus)
text_corpus = [
    "Hello how are you",
    "Hello I am fine",
    "How are you doing",
    "I am learning RNN",
    "RNNs are great for text"
]

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_corpus)
total_words = len(tokenizer.word_index) + 1

# Create input-output sequences
input_sequences = []
for line in text_corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Padding sequences
max_sequence_length = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

# Splitting into input (X) and output (y)
X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Build the RNN model
model = Sequential([
    Embedding(total_words, 10, input_length=max_sequence_length - 1),
    LSTM(100),
    Dense(total_words, activation='softmax')
])

# Compile and train the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=500, verbose=1)

# Function to predict next word
def predict_next_word(seed_text):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted_probs)
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return "?"

# Test the prediction
seed_text = "How are you"
next_word = predict_next_word(seed_text)
print(f"Next word prediction: {seed_text} -> {next_word}")
