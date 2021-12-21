import numpy as np
import glob, csv
import matplotlib.pyplot as plt
from sklearn import (svm,  metrics)

class YaguData:
    def __init__(self, name):
        self.team_name = name
    data = []
    target = []
    target_names = ['LOSE', 'WIN', 'DRAW'] # DRAW를 빼면 정확도 0.466 -> 0.708까지
    feature_names = ['ATK_AVG', 'ATK_OPS', 'PIT_ERA', 'PIT_WHIP', 'DEF_A', 'DEF_E', 'Cont_Winning_Rate_20', 'Diff', 'HOME_AWAY']

def read_data(filename, team): # Load a dataset
    files = glob.glob(filename)
    yagu = YaguData(team)
    for file in files:
        with open(file, 'r', encoding='utf-8-sig') as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                if line and not line[0].strip().startswith('#'): # If 'line' is valid and not a header
                    yagu.target.append(0 if line[0] == 'L' else (1 if line[0] == 'W' else 2))
                    value = [float(line[i]) for i in range(1, len(line)- 1)]
                    value.append(0 if line[-1] == 'H' else 1)
                    yagu.data.append(value)
    yagu.data = np.array(yagu.data)
    yagu.target = np.array(yagu.target)
    return yagu

if __name__ == '__main__':
    print("##### Welcome to YaguMagu!! #####")
    # Load a dataset
    yagu = read_data('data/lotte.csv', "LOTTE GIANTS")   
    print(f'Your Team: {yagu.team_name}')                              
    # Train a model
    model = svm.SVC()                                                                          
    model.fit(yagu.data, yagu.target)

    # Test the model
    predict = model.predict(yagu.data)
    n_correct = sum(predict == yagu.target)
    accuracy = metrics.balanced_accuracy_score(yagu.target, predict)
    print(f'Accuracy: {accuracy}')                   

    # Visualize testing results
    xs = [x for x in range(0, yagu.target.size)]
    ys = [yagu.target[x] - predict[x] for x in xs]

    plt.title(f'svm.SVC (Balanced_Accuracy={accuracy:.3f})')
    plt.plot(xs, ys, 'r-')
    plt.xlabel('Data size')
    plt.ylabel('Answer - prediction')
    plt.grid()
    plt.show()