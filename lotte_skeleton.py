import numpy as np
import matplotlib.pyplot as plt
from sklearn import (svm, linear_model, metrics)
from matplotlib.lines import Line2D

def load_lotte_data(filename):
    class LOTTEData:
        data = []
        target = []
        target_names = ['LOSE', 'WIN', 'DRAW']
        feature_names = ['Win_Lose_Draw', 'HOME_AWAY', 'LOTTE_ATK_AVG', 'LOTTE_ATK_OPS', 'LOTTE_PIT_ERA', 'LOTTE_PIT_WHIP', 'LOTTE_DEF_A', 'LOTTE_DEF_E',' Cont_Winning_Rate_20']
    lotte = LOTTEData()
    try:
        with open(filename, 'r') as f:
            for line in f:
                values = [word for word in line.split(',')]
                lotte.target.append(0 if values[0] == 'L' else (1 if values[0] == 'W' else 2))
                lotte.data.append(0 if values[1] == 'H' else 1)
                value = [float(i) for i in values[2:]]
                lotte.data.append(value)
    except Exception as ex:
        print(f'Cannot run the program. (message: {ex})')
    lotte.data = np.array(lotte.data)
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
