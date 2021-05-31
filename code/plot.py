import seaborn as sns
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-p','--paths',type=str, nargs = '+', dest = 'paths',
                      help='The directories of results to be plotted.')
    args = args.parse_args()

    paths = args.paths
    for path in paths:
        summaries = pd.concat([pd.read_csv(path + '/' + file) for file in os.listdir(path) if file[:7] == 'summary'],
                              ignore_index=True)
        sns.lineplot(x='budget', y='sqrt_pehe',data=summaries)
    plt.show()