import pandas as pd
import random


def load_file(path: str):
    return pd.read_csv(path)


def predict(row, weights):
    result = weights[0]
    for i in range(len(row)):
        result += weights[i + 1] * row[i]

    return 1.0 if result > 0.0 else -1.0


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
            expected = expected_predictions[index]

            if prediction != expected:
                for i in range(len(row) - 1):
                    weights[i + 1] = weights[i + 1] + row[i] * expected

    test(test_df, weights)


def test(df, weights):
    correct = 0
    incorrect = 0
    expected_predictions = df['Outcome']
    for index, row in df.iterrows():
        prediction = predict(row[:len(row) - 1], weights)
        expected = expected_predictions[index]

        if prediction == expected:
            correct += 1
            print('Correct prediction')
        else:
            incorrect += 1
            print('Incorrect prediction')

    print('\n----- Stats -----')
    print(f'\tFinal weights: {weights}')
    print(f'\tTotal correct predictions: {correct}')
    print(f'\tTotal incorrect predictions: {incorrect}')
    print(f'\tPercentage: {round((correct / (correct + incorrect)) * 100)}%')


if __name__ == '__main__':
    df = load_file(input('Filename of dataset to read: '))
    df.loc[df['Outcome'] == 0, 'Outcome'] = -1
    perceptron(df)
