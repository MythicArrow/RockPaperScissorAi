import random
import numpy as np
import tensorflow as tf
from collections import deque
import threading
from queue import Queue
import logging
from tkinter import *
import os


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class RockPaperScissorsAI:
    def __init__(self):
        self.move_history = deque(maxlen=100)  # Automatically keeps the last 100 entries
        self.labels = {'R': 0, 'P': 1, 'S': 2}
        self.moves = ['R', 'P', 'S']
        self.model = self.build_model()
        self.lock = threading.Lock()  # Lock for thread safety
        self.training_queue = Queue()
        self.exploration_rate = 0.1  # 10% exploration

    def build_model(self):
        """Builds an LSTM model for predicting the next move based on move history."""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(5, 6)),  # 5 previous moves, each with 6 features (user + AI moves)
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')  # Output layer for Rock, Paper, Scissors
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def one_hot_encode_move(self, move):
        """One-hot encodes a move into a 3-dimensional vector."""
        move_encoded = [0] * 3
        move_encoded[self.labels[move]] = 1
        return move_encoded

    def get_ai_move(self):
        """Predicts AI's move using the model and decides the counter-move."""
        with self.lock:  # Ensure thread-safe access to move_history
            if len(self.move_history) < 5:
                return random.choice(self.moves)  # Random move for the first few rounds

            # Prepare input data (last 5 moves)
            input_data = np.array(self.move_history[-5:]).reshape(1, 5, 6)

        prediction = self.model.predict(input_data)
        predicted_move = np.argmax(prediction)

        # Introduce exploration (10% chance of random move)
        if random.random() < self.exploration_rate:
            return random.choice(self.moves)

        # Counter the predicted move
        counter_move = (predicted_move + 1) % 3
        return self.moves[counter_move]

    def update_history(self, user_move, ai_move):
        """Updates the move history with the user's and AI's moves."""
        user_move_encoded = self.one_hot_encode_move(user_move)
        ai_move_encoded = self.one_hot_encode_move(ai_move)
        with self.lock:  # Ensure thread-safe modification of move_history
            self.move_history.append(user_move_encoded + ai_move_encoded)  # Combine both moves

            # Add the history to the training queue for the background thread
            self.training_queue.put(self.move_history.copy())

    def train_model(self):
        """Train the model with the move history in a separate thread."""
        batch_size = 10  # Define a batch size
        while True:
            move_history = self.training_queue.get()
            if len(move_history) < batch_size:
                continue

            X = np.array([move_pair for move_pair in move_history[:-1]])  # Input: all moves except the last
            y = np.array([move_history[i + 1][0] for i in range(len(move_history) - 1)])  # Output: next user move

            # Train the model with mini-batches
            self.model.fit(X, y, batch_size=batch_size, epochs=1, verbose=0)

    def save_model(self, file_path='rock_paper_scissors_model.h5'):
        """Saves the current model to a file."""
        self.model.save(file_path)
        logging.info(f"Model saved to {file_path}")

    def load_model(self, file_path='rock_paper_scissors_model.h5'):
        """Loads the model from a file, if it exists."""
        if os.path.exists(file_path):
            self.model = tf.keras.models.load_model(file_path)
            logging.info(f"Model loaded from {file_path}")
        else:
            logging.info(f"Model file {file_path} not found. Using a new model.")


# Game with GUI
class RockPaperScissorsGame:
    def __init__(self, root):
        self.ai = RockPaperScissorsAI()
        self.ai.load_model()  # Load a pre-trained model if available

        # Start the training thread
        self.training_thread = threading.Thread(target=self.ai.train_model)
        self.training_thread.daemon = True
        self.training_thread.start()

        # GUI Setup
        self.root = root
        self.root.title("Rock Paper Scissors")
        self.create_widgets()

        # Game tracking variables
        self.ai_wins = 0
        self.user_wins = 0
        self.total_games = 0

    def create_widgets(self):
        """Creates the GUI layout."""
        self.label = Label(self.root, text="Choose your move", font=("Helvetica", 16))
        self.label.pack(pady=20)

        self.rock_button = Button(self.root, text="Rock", command=lambda: self.play_game('R'))
        self.rock_button.pack(padx=20, pady=5)

        self.paper_button = Button(self.root, text="Paper", command=lambda: self.play_game('P'))
        self.paper_button.pack(padx=20, pady=5)

        self.scissors_button = Button(self.root, text="Scissors", command=lambda: self.play_game('S'))
        self.scissors_button.pack(padx=20, pady=5)

        self.result_label = Label(self.root, text="", font=("Helvetica", 14))
        self.result_label.pack(pady=20)

        self.score_label = Label(self.root, text="", font=("Helvetica", 14))
        self.score_label.pack(pady=20)

    def play_game(self, user_move):
        """Simulates a single round of the game."""
        ai_move = self.ai.get_ai_move()
        self.result_label.config(text=f"AI chose: {ai_move}")

        # Determine the winner
        if (ai_move == 'R' and user_move == 'S') or (ai_move == 'P' and user_move == 'R') or (
                ai_move == 'S' and user_move == 'P'):
            self.result_label.config(text=f"AI wins! AI chose: {ai_move}")
            self.ai_wins += 1
        elif ai_move == user_move:
            self.result_label.config(text=f"It's a tie! AI chose: {ai_move}")
        else:
            self.result_label.config(text=f"You win! AI chose: {ai_move}")
            self.user_wins += 1

        self.total_games += 1
        self.score_label.config(text=f"AI Wins: {self.ai_wins}, User Wins: {self.user_wins}, Total Games: {self.total_games}")

        # Update history and train
        self.ai.update_history(user_move, ai_move)

        # Save the model periodically
        if self.total_games % 10 == 0:
            self.ai.save_model()


# Start the GUI game
if __name__ == "__main__":
    root = Tk()
    game = RockPaperScissorsGame(root)
    root.mainloop()
