import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

# Sample training data (Replace this with your actual training texts)
training_texts = [
    "I feel sad today",
    "I'm very happy!",
    "I'm feeling stressed",
    "I need help with my emotions",
    "I'm okay, just a bit tired",
]

# Initialize and fit the tokenizer
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(training_texts)  # Fit on your real training data

# Save the tokenizer to a file
with open("tokenizer.pkl", "wb") as handle:
    pickle.dump(tokenizer, handle)

print("✅ Tokenizer saved successfully as tokenizer.pkl")
