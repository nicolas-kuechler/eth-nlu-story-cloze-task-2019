import matplotlib.pyplot as pyplot
import numpy as np
import argparse
import pandas as pd


def illustrate_distribution(epoch_dir, name):
        load = pd.read_csv(epoch_dir+name+'.csv', sep=',')
        #print(load)
        load['ProbQuiz1Normalized'] = load['ProbQuiz1'] / \
                (load['ProbQuiz1']+load['ProbQuiz2'])
        load['ProbQuiz2Normalized'] = load['ProbQuiz2'] / \
                (load['ProbQuiz1']+load['ProbQuiz2'])

        def function(x):
                x['PredProbNormalized'] = x['ProbQuiz1Normalized'] if x['PredRightEnding'] == 1 else x['ProbQuiz2Normalized'] 
                return x

        #load['PredProbNormalized'] = load['ProbQuiz1Normalized'] if load['PredRightEnding'] == 1 else load['ProbQuiz1Normalize2']
        load = load.apply(function,axis=1)


        #load['PredProbNormalized'][load['PredRightEnding']==1] = load['ProbQuiz1Normalized'][load['PredRightEnding']==1]
        #load['PredProbNormalized'][load['PredRightEnding']==2] = load['ProbQuiz2Normalized'][load['PredRightEnding']==2]

        x = load[load['CorrectPred']]['PredProbNormalized'].values
        y = load[~load['CorrectPred']]['PredProbNormalized'].values

        bins = np.linspace(0.5, 1., 20)
        pyplot.hist(x, bins, alpha=0.5, histtype='barstacked',
                label='predicted right ending')
        pyplot.hist(y, bins, alpha=0.5, histtype='barstacked',
                label='predicted wrong ending')
        pyplot.legend(loc='upper right')
        pyplot.title('Certainty of model')
        pyplot.xlabel('probability')
        pyplot.ylabel('#predictions')

        pyplot.savefig(epoch_dir+name+'.png', bbox_inches='tight')
        pyplot.close()

        x = load[load['CorrectPred']]['PredProbNormalized'].values
        y = np.array(((load[~load['CorrectPred']]['PredProbNormalized']*(-1.)) + 1.).values)
        #print(y)
        bins = np.linspace(0., 1., 20)
        pyplot.hist(x, bins, alpha=0.5, histtype='barstacked',
                label='predicted right ending',align='mid')
        pyplot.hist(y, bins, alpha=0.5, histtype='barstacked',
                label='predicted wrong ending',align='mid')
        pyplot.legend(loc='upper right')
        pyplot.title('Certainty of model')
        pyplot.xlabel('probability')
        pyplot.ylabel('#predictions')

        pyplot.savefig(epoch_dir+name+'_center.png', bbox_inches='tight')
        pyplot.close()
        
        x = load[load['CorrectPred']]['PredProbNormalized'].values
        y = np.array(((load[~load['CorrectPred']]['PredProbNormalized']*(-1.)) + 1.).values)
        x = np.concatenate((x,y),axis=0)
        #print(y)
        bins = np.linspace(0., 1., 20)
        pyplot.hist(x, bins, alpha=0.5, histtype='barstacked',
                label='probability given to right ending',align='mid')
        pyplot.legend(loc='upper right')
        pyplot.title('Distribution for right ending')
        pyplot.xlabel('probability')
        pyplot.ylabel('#predictions')

        pyplot.savefig(epoch_dir+name+'_distro.png', bbox_inches='tight')
        pyplot.close()
        return


def norm(pred):
    save = pred[:, 0].copy()
    pred[:, 0] = pred[:, 0]/(pred[:, 0]+pred[:, 1])
    pred[:, 1] = pred[:, 1]/(save+pred[:, 1])
    return pred


def main():
    names = ['valid_results_converted', 'test_results_converted']
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
