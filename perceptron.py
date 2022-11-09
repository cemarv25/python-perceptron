import pandas as pd
import random


def load_file(path: str):
    return pd.read_csv(path)


def predict(row, weights):
    result = weights[0]
    for i in range(len(row)):
        result += weights[i + 1] * row[i]

    return 1.0 if result >= 0.0 else 0.0


def perceptron(df: pd.DataFrame):
    test_pct = 0.2
    rows = df.shape[0]

    train_rows = round((1 - test_pct) * rows)

    train_df = df[:train_rows]
    test_df = df[train_rows:]
    expected_predictions = train_df['Outcome']

    weights = [0] * (len(df.columns))
    weights[0] = 0.5
    for i in range(1, len(weights)):
        weights[i] = random.randrange(0, 3)

    for _ in range(300):
        for index, row in train_df.iterrows():
            prediction = predict(row[:len(row) - 1], weights)

            if prediction != expected_predictions[index]:
                for i in range(len(row)):
                    weights[i + 1] = weights[i + 1] + row[i] * row[-1]


if __name__ == '__main__':
    df = load_file(input('Filename of dataset to read: '))
    perceptron(df)
