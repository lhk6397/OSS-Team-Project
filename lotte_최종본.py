import numpy as np
import glob, csv
import matplotlib.pyplot as plt
from sklearn import (svm,  metrics)

class LOTTEData:
    data = []
    target = []
    target_names = ['LOSE', 'WIN', 'DRAW'] # DRAW를 빼면 정확도 0.466 -> 0.708까지
    feature_names = ['LOTTE_ATK_AVG', 'LOTTE_ATK_OPS', 'LOTTE_PIT_ERA', 'LOTTE_PIT_WHIP', 'LOTTE_DEF_A', 'LOTTE_DEF_E', 'Cont_Winning_Rate_20', 'Diff', 'HOME_AWAY']

def read_data(filename): # Load a dataset
    files = glob.glob(filename)
    lotte = LOTTEData()
    for file in files:
        with open(file, 'r', encoding='utf-8-sig') as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                if line and not line[0].strip().startswith('#'): # If 'line' is valid and not a header
                    lotte.target.append(0 if line[0] == 'L' else (1 if line[0] == 'W' else 2))
                    value = [float(line[i]) for i in range(1, len(line)- 1)]
                    value.append(0 if line[-1] == 'H' else 1)
                    lotte.data.append(value)
    lotte.data = np.array(lotte.data)
    lotte.target = np.array(lotte.target)
    return lotte

if __name__ == '__main__':
    # Load a dataset
    lotte = read_data('data/lotte.csv')                                 
    # Train a model
    model = svm.SVC()                                                                          
    model.fit(lotte.data, lotte.target)

    # Test the model
    predict = model.predict(lotte.data)
    n_correct = sum(predict == lotte.target)
    accuracy = metrics.balanced_accuracy_score(lotte.target, predict)
    print(f'Accuracy: {accuracy}')                   

    # Visualize testing results
    xs = [x for x in range(0, lotte.target.size)]
    ys = [lotte.target[x] - predict[x] for x in xs]

    plt.title(f'svm.SVC (Balanced_Accuracy={accuracy:.3f})')
    plt.plot(xs, ys, 'r-')
    plt.xlabel('Data size')
    plt.ylabel('Answer - prediction')
    plt.grid()
    plt.show()