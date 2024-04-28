import matplotlib.pyplot as plt
from numpy.random import default_rng
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np


def bootstrap_plot(X_test, y_test, model, treshold, name):
    accuracy_scores = []

    for _ in range(3000):
        rng = default_rng()
        sample_indices = rng.choice(X_test.index, size=100, replace=False)
        X_test_sample = X_test.loc[sample_indices]
        y_test_sample = y_test.loc[sample_indices]

        y_pred_sample = model.predict(X_test_sample, verbose=False)
        if type(y_pred_sample[0]) == np.int64:
            accuracy = accuracy_score(y_test_sample, y_pred_sample)
        else:
            accuracy = accuracy_score(y_test_sample, [1 if y_pred_sample[i][0] > treshold else 0 for i in range(len(y_pred_sample))])

        accuracy_scores.append(accuracy)

    plt.hist(accuracy_scores, bins=10, density=True, color='skyblue')
    plt.title(f'Распределение Accuracy {name} для случайных подвыборок тестовой выборки')
    plt.xlabel('Accuracy Score')
    plt.ylabel('Частота')
    plt.grid(axis='y', alpha=0.75)

    plt.xlim(0.85, 1)

    # Вычисление 5-го и 95-го квантилей
    quantile_5 = np.percentile(accuracy_scores, 5)
    quantile_95 = np.percentile(accuracy_scores, 95)

    plt.axvline(x=quantile_5, color='red', linestyle='--')
    plt.axvline(x=quantile_95, color='red', linestyle='--')

    plt.text(quantile_5, 0, '5-й квантиль', rotation=90)
    plt.text(quantile_95, 0, '95-й квантиль', rotation=90)

    # plt.show()

    print(f"5-й квантиль {name}: {quantile_5}")
    print(f"95-й квантиль {name}: {quantile_95}")


def plot_reliability(data):
    x_values = []
    y_values = []

    step = 0.1
    x_cur = 0
    while x_cur <= 1:
        df_inds = data[(x_cur <= data.preds) & (data.preds <= (x_cur + step))].index
        if df_inds.empty:
            pass
        else:
            mean_label = data.iloc[df_inds].label.values.mean()
            x_values += data.iloc[df_inds].preds.tolist()
            y_values += [mean_label] * len(df_inds)
        x_cur += step
    # plt.step(x_values, y_values, where='post', color='blue', linestyle='-', linewidth=1.5)
    plt.scatter(x_values, y_values, color='blue', s=5)
    plt.xlabel('Предсказания модели', fontsize=12)
    plt.ylabel('Соотношение класса 1 к классу 0', fontsize=12)
    plt.title('График надежности')
    plt.grid(True, linestyle='--', alpha=0.6)
    # plt.show()


def boxplot_reliability(data):
    x_values = []
    y_values = []

    step = 0.1
    x_cur = 0
    while x_cur <= 1:
        df_inds = data[(x_cur <= data.preds) & (data.preds <= (x_cur + step))].index
        if df_inds.empty:
            pass
        else:
            mean_label = data.iloc[df_inds].label.values.mean()
            x_values += data.iloc[df_inds].preds.tolist()
            y_values += [mean_label] * len(df_inds)
        x_cur += step
        plt.boxplot(data.iloc[df_inds].preds.tolist(), showfliers=False, positions=[round(x_cur, 2)])

    plt.xlabel('Предсказания модели', fontsize=12)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.ylabel('Соотношение класса 1 к классу 0', fontsize=12)
    plt.title('График надежности')
    plt.grid(True, linestyle='--', alpha=0.6)

    # plt.show()


def plot_roc_curve(y_true_1, y_score_1, y_true_2, y_score_2):
    fpr_1, tpr_1, _ = roc_curve(y_true_1, y_score_1)
    roc_auc_1 = roc_auc_score(y_true_1, y_score_1)

    fpr_2, tpr_2, _ = roc_curve(y_true_2, y_score_2)
    roc_auc_2 = roc_auc_score(y_true_2, y_score_2)

    plt.figure()
    lw = 2
    plt.plot(fpr_1, tpr_1, color='darkorange', lw=lw, label='ROC curve Catboost (area = %0.4f)' % roc_auc_1)
    plt.plot(fpr_2, tpr_2, color='green', lw=lw, label='ROC curve Нейросеть (area = %0.4f)' % roc_auc_2)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


