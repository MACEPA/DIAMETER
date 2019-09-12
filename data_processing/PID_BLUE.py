import numpy as np
import pandas as pd
from scipy.special import expit
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc
from matplotlib.backends.backend_pdf import PdfPages
# set chained assignment to ignore warnings
pd.set_option('chained_assignment', None)


def main():
    df = pd.read_csv('C:/Users/lzoeckler/Desktop/for_allison/PID_BLUE_FULL_cleaned.csv')
    df['null_IgM'] = df['IgG'].add(df['IgA'])
    end_df = df.copy(deep=True)

    IgG_data = df['IgG'].values
    IgM_data = df['IgM'].values
    IgA_data = df['IgA'].values
    combined_data = df['IgG_IgA_IgM'].values
    null_IgM_data = df['null_IgM'].values
    all_data = df[['IgG', 'IgM', 'IgA']].as_matrix()
    test_vals = df['binary_result']
    binary_data = test_vals.values.reshape(-1, 1)

    clf_IgG = linear_model.LogisticRegression(random_state=0)
    clf_IgG.fit(IgG_data.reshape(-1, 1), binary_data)
    probabilities = clf_IgG.predict_proba(IgG_data.reshape(-1, 1))
    end_df['probability_of_1_IgG_based'] = probabilities[:, 1]

    clf_combined = linear_model.LogisticRegression(random_state=0)
    clf_combined.fit(combined_data.reshape(-1, 1), binary_data)
    probabilities = clf_combined.predict_proba(combined_data.reshape(-1, 1))
    end_df['probability_of_1_combined_based'] = probabilities[:, 1]

    pp = PdfPages('C:/Users/lzoeckler/Desktop/for_allison/logistic_PID_BLUE_FULL.pdf')

    f = plt.figure()
    plt.scatter(IgG_data, binary_data, color='black', zorder=20, alpha=0.7)
    x_test = np.linspace(0, 20, 300)
    loss = expit(x_test * clf_IgG.coef_ + clf_IgG.intercept_).ravel()
    plt.plot(x_test, loss, color='red', linewidth=3, alpha=0.7, label='Logistic regression')
    fifty_prob = ((0 - clf_IgG.intercept_) / clf_IgG.coef_).item()
    plt.axvline(fifty_prob, color='k', alpha=0.7, label='50% probability')
    plt.axhline(0.5, color='k', alpha=0.7)
    plt.ylabel('Probability')
    plt.xlabel('IgG')
    plt.xlim(-.2, 6)
    plt.xticks([0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0], rotation='vertical')
    plt.legend()
    plt.title('Logistic regression of IgG values\n50% probability at {}'.format(round(fifty_prob, 2)))
    plt.tight_layout()
    plt.show()
    pp.savefig(f)
    plt.close()

    f = plt.figure()
    test_vals = df['binary_result']
    test_pred = clf_IgG.predict_proba(IgG_data.reshape(-1, 1))[:, 1]
    fpr, tpr, threshold = roc_curve(test_vals, test_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='red', linewidth=3)
    plt.title('IgG ROC\nAUC = {}'.format(round(roc_auc, 3)))
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.tight_layout()
    plt.show()
    pp.savefig(f)
    plt.close()

    f = plt.figure()
    plt.scatter(combined_data, binary_data, color='black', zorder=20, alpha=0.7)
    x_test = np.linspace(0, 20, 300)
    loss = expit(x_test * clf_combined.coef_ + clf_combined.intercept_).ravel()
    plt.plot(x_test, loss, color='green', linewidth=3, alpha=0.7, label='Logistic regression')
    fifty_prob = ((0 - clf_combined.intercept_) / clf_combined.coef_).item()
    plt.axvline(fifty_prob, color='k', alpha=0.7, label='50% probability')
    plt.axhline(0.5, color='k', alpha=0.7)
    plt.ylabel('Probability')
    plt.xlabel('Combined')
    plt.xlim(-.2, 8)
    plt.xticks([0, 0.5, 1.0, 1.5, 2.0, 4.0, 6.0, 8.0], rotation='vertical')
    plt.legend()
    plt.title('Logistic regression of combined values\n50% probability at {}'.format(round(fifty_prob, 2)))
    plt.tight_layout()
    plt.show()
    pp.savefig(f)
    plt.close()

    f = plt.figure()
    test_vals = df['binary_result']
    test_pred = clf_combined.predict_proba(combined_data.reshape(-1, 1))[:, 1]
    fpr, tpr, threshold = roc_curve(test_vals, test_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='green', linewidth=3)
    plt.title('Combined ROC\nAUC = {}'.format(round(roc_auc, 3)))
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.tight_layout()
    plt.show()
    pp.savefig(f)
    plt.close()

    pp.close()


if __name__ == '__main__':
    main()
