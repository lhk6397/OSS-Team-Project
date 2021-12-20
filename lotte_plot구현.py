import numpy as np
import glob, csv
import matplotlib.pyplot as plt
from sklearn import (svm, linear_model, metrics)
from matplotlib.lines import Line2D

class LOTTEData:
    data = []
    target = []
    target_names = ['LOSE', 'WIN', 'DRAW'] # DRAW를 빼면 정확도 0.473 -> 0.704까지
    feature_names = ['LOTTE_ATK_AVG', 'LOTTE_ATK_OPS', 'LOTTE_PIT_ERA', 'LOTTE_PIT_WHIP', 'LOTTE_DEF_A', 'LOTTE_DEF_E', 'Cont_Winning_Rate_20', 'Diff', 'HOME_AWAY']

def load_lotte_data(filename):
    lotte = LOTTEData()
    try:
        with open(filename, 'r') as f:
            for line in f:
                values = [word for word in line.split(', ')]
                lotte.target.append(0 if values[0] == 'L' else (1 if values[0] == 'W' else 2))
                value = [float(i) for i in values[1:9]]
                lotte.data.append(value)
                value.append(0 if values[9] == 'H' else 1)
    except Exception as ex:
        print(f'Cannot run the program. (message: {ex})')
    lotte.data = np.array(lotte.data)
    lotte.target = np.array(lotte.target)
    return lotte

if __name__ == '__main__':
    # Load a dataset
    lotte = load_lotte_data('data/lotte.data')                                 
    # Train a model
    model = svm.SVC()                                                                          
    model.fit(lotte.data, lotte.target)

    # Test the model
    predict = model.predict(lotte.data)
    n_correct = sum(predict == lotte.target)
    accuracy = metrics.balanced_accuracy_score(lotte.target, predict)   
    
    print(predict)
    print(lotte.target)
    print(accuracy)                   

# Visualize testing results
xs = [x for x in range(0, lotte.target.size)]
ys = [lotte.target[x] - predict[x] for x in xs]

plt.title(f'svm.SVC (Balanced_Accuracy={accuracy:.3f})')
plt.plot(xs, ys, 'r-')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()