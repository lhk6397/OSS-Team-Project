import numpy as np
import matplotlib.pyplot as plt
from sklearn import (svm, linear_model, metrics) # Mission #2 and #3) You need to import some modules if necessary
from matplotlib.lines import Line2D # For the custom legend

def load_lotte_data(filename):
    class LOTTEData:
        data = []
        target = []
        target_names = ['DEFEAT', 'WIN', 'DRAW']
        feature_names = ['HOME_AWAY', 'LOTTE_ATK_AVG', 'LOTTE_ATK_OPS', 'LOTTE_PIT_ERA', 'LOTTE_PIT_WHIP', 'LOTTE_DEF_A', 'LOTTE_DEF_E',' Cont_Winning_Rate_20']
    lotte = LOTTEData()
    # TODO
    try:
        with open(filename, 'r') as f:
            for line in f:
                values = [word for word in line.split(',')]
                lotte.target.append(0 if values[0] == 'H' else 1)
                value = [float(i) for i in values[1:]]
                lotte.data.append(value)
    except Exception as ex:
        print(f'Cannot run the program. (message: {ex})')
    lotte.data = np.array(lotte.data)
    return lotte

if __name__ == '__main__':
    # Load a dataset
    lotte = load_lotte_data('data/lotte.data')                                                      # Mission #1) Implement 'load_wdbc_data()'
    # Train a model
    model = svm.SVC()                                                                            # Mission #2) Try at least two different classifiers
    model.fit(lotte.data, lotte.target)

    # Test the model
    predict = model.predict(lotte.data)
    n_correct = sum(predict == lotte.target)
    accuracy = metrics.balanced_accuracy_score(lotte.target, predict)                             # Mission #3) Calculate balanced accuracy

