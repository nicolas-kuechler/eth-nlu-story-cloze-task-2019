import matplotlib.pyplot as pyplot
import numpy as np
import argparse
import pandas as pd

def illustrate_distribution(epoch_dir, name):
    load = pd.read_csv(epoch_dir+name+'.csv', sep=',', usecols=['AnswerRightEnding', 'ProbQuiz1', 'ProbQuiz2'])
    
    pred = load[['ProbQuiz1', 'ProbQuiz2']]
    label = load[['AnswerRightEnding']]-1
    
    pred = pred.values
    label = label.values
    #pred = np.reshape(pred,(-1,2))[:,0]
    label = np.reshape(label,(-1))

    pred = norm(pred)

    print(len(pred),len(label))
    print(label==1)
    print(pred[:,0])
    print(label)
    bins = np.linspace(0, 1, 50)
    x = pred[:,0][label==0]
    y = pred[:,0][label==1]
    pyplot.hist(x, bins, alpha=0.5, histtype='barstacked', label='P(right ending)')
    pyplot.hist(y, bins, alpha=0.5, histtype='barstacked', label='1-P(wrong ending)')
    pyplot.legend(loc='upper left')

    pyplot.savefig(epoch_dir+name+'.png', bbox_inches='tight')
    pyplot.close()

    #only center
    x = 0.01
    bins = np.linspace(0.5-x, 0.5+x, 50)
    x = pred[:,0][label==0]
    y = pred[:,0][label==1]
    pyplot.hist(x, bins, alpha=0.5, histtype='barstacked', label='P(right ending)')
    pyplot.hist(y, bins, alpha=0.5, histtype='barstacked', label='1-P(wrong ending)')
    pyplot.legend(loc='upper left')

    pyplot.savefig(epoch_dir+name+'_center.png', bbox_inches='tight')
    pyplot.close()
    return

def norm(pred):
    pred[:,0] = pred[:,0]/(pred[:,0]+pred[:,1])
    pred[:,1] = pred[:,1]/(pred[:,0]+pred[:,1])
    return pred

def main():
    names = ['valid_results_converted (2)', 'test_results_converted'] #+ ['valid_results_converted', 'eth_test_results_converted']
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epoch_dir', help='running directory of epoch')
    args = parser.parse_args()

    epoch_dir = args.epoch_dir

    print(f"Using Epoch Dir: {epoch_dir}")

    for name in names:
        print(f'name: {name}')
        illustrate_distribution(epoch_dir, name)

if __name__ == '__main__':
    main()
