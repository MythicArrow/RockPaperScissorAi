import random
import tensorflow as tf
import numpy as np


class RockPaperScissorsAI:
    def __init__(self):
        self.move_history = []
        self.labels = {'R': 0, 'P': 1, 'S': 2}
        self.moves = ['R', 'P', 'S']
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def get_ai_move(self):
        if len(self.move_history) < 5:
            return random.choice(self.moves)

        input_data = np.array(self.move_history[-5:]).flatten().reshape(1, -1)
        prediction = self.model.predict(input_data)
        predicted_move = np.argmax(prediction)
        counter_move = (predicted_move + 1) % 3
        return self.moves[counter_move]

    def update_history(self, user_move, ai_move):
        user_move_encoded = self.labels[user_move]
        ai_move_encoded = self.labels[ai_move]
        self.move_history.append([user_move_encoded, ai_move_encoded])

        if len(self.move_history) > 100:
            self.move_history.pop(0)

    def train_model(self):
        if len(self.move_history) < 10:
            return

        X = np.array([move_pair for move_pair in self.move_history[:-1]])
        y = np.array([self.move_history[i + 1][0] for i in range(len(self.move_history) - 1)])

        self.model.fit(X, y, epochs=10, verbose=0)


# Simulating a game
def play_game():
    ai = RockPaperScissorsAI()
    ai_wins = 0
    user_wins = 0
    total_games = 0

    while True:
        user_move = input("Enter your move (R for Rock, P for Paper, S for Scissors, Q to quit): ").upper()
        if user_move == 'Q':
            break
        if user_move not in ['R', 'P', 'S']:
            print("Invalid move. Please enter R, P, S, or Q to quit.")
            continue

        ai_move = ai.get_ai_move()
        print(f"AI chose: {ai_move}")

        if (ai_move == 'R' and user_move == 'S') or (ai_move == 'P' and user_move == 'R') or (
                ai_move == 'S' and user_move == 'P'):
            print("AI wins this round!")
            ai_wins += 1
        elif ai_move == user_move:
            print("It's a tie!")
        else:
            print("You win this round!")
            user_wins += 1

        ai.update_history(user_move, ai_move)
        ai.train_model()
        total_games += 1

    print(f"Final Score - AI Wins: {ai_wins}, User Wins: {user_wins}, Total Games: {total_games}")
    print(f"AI Win Percentage: {(ai_wins / total_games) * 100:.2f}%")


# Start the game
if __name__ == "__main__":
    play_game()





