import mlrose.mlrose_hiive as mlrose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data.generated.generated_data import get_noisy_nonlinear_with_non_noisy_labels
from sklearn.metrics import accuracy_score
import pickle
import os
from sklearn.model_selection import KFold
import time

GS_FILE_NAME_PREFIX = "Analysis_Data/NOISY_NONLINEAR_NN_DATA"

PLOT_SAVE_LOCATION = "Figures/"
ALGO = "Neural Network"
DATASET = "Noisy Non-Linear"
GLOBAL_FIG_COUNTER = 0

default_title = "{} Default Learning Curve\n Data Set: {}".format(ALGO, DATASET)
final_title = "{} Final Tuning Learning Curve\n Data Set: {}".format(ALGO, DATASET)

title_fontsize = 24
fontsize = 24
legend_fontsize = 16
default_figure_size = (12, 6)


def gradient_decent_loss_curve(X_train, y_train):
    plt.figure(figsize=default_figure_size)
    temp_file_name = 'temp_NN_GD_LC.pickle'

    if (os.path.isfile(temp_file_name)):
        print("WARNING: Not Running Loading: ", temp_file_name)
        with open(temp_file_name, 'rb') as handle:
            lr_nn_model1 = pickle.load(handle)
    else:
        lr_nn_model1 = mlrose.NeuralNetwork(hidden_nodes=[80], activation='relu', \
                                            algorithm='gradient_descent', max_iters=50000, \
                                            bias=True, is_classifier=True, learning_rate=0.0015, \
                                            early_stopping=True, clip_max=5, max_attempts=200, \
                                            random_state=42, curve=True)
        print("Training")
        lr_nn_model1.fit(X_train, y_train)
        with open(temp_file_name, 'wb') as handle:
            pickle.dump(lr_nn_model1, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    y_train_pred = lr_nn_model1.predict(X_train)
    y_train_accuracy = accuracy_score(y_train, y_train_pred)
    print(y_train_accuracy)

    plt.plot(lr_nn_model1.fitness_curve)
    plt.ylim([-2, 0])
    plt.title("Gradient Descent Loss Curve", fontsize=title_fontsize, fontweight='bold')
    plt.xlabel("Iterations", fontsize=fontsize)
    plt.ylabel("Fitness", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig('gd_loss_curve.png')
    plt.show()

def parameter_tuning_rhc_nn(X_train, y_train):
    plt.figure(figsize=default_figure_size)
    restarts = [0, 5, 10]
    rhc_nn_model_results = []

    for r in restarts:
        print("Running restarts {}".format(r))
        temp_file_name = 'temp_NN_rhc_restarts_{}.pickle'.format(r)

        if (os.path.isfile(temp_file_name)):
            print("WARNING: Not Running Loading: ", temp_file_name)
            with open(temp_file_name, 'rb') as handle:
                lr_nn_model1 = pickle.load(handle)
        else:
            lr_nn_model1 = mlrose.NeuralNetwork(hidden_nodes=[80], activation='relu', \
                                                algorithm='random_hill_climb', max_iters=350000, \
                                                bias=True, is_classifier=True, learning_rate=0.0015, \
                                                early_stopping=True, max_attempts=200, \
                                                random_state=42, curve=True, restarts=r)
            print("Training")

            lr_nn_model1.fit(X_train, y_train)
            with open(temp_file_name, 'wb') as handle:
                pickle.dump(lr_nn_model1, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
        rhc_nn_model_results.append(lr_nn_model1.fitness_curve.copy())

        y_train_pred = lr_nn_model1.predict(X_train)
        y_train_accuracy = accuracy_score(y_train, y_train_pred)
        print(y_train_accuracy)

    for r,i in zip(restarts, rhc_nn_model_results):
        plt.plot(i, label="restarts = {}".format(r), linewidth=1.5)

    plt.title("Random Hill Climbing NN Training Loss Curve", fontsize=title_fontsize, fontweight='bold')
    plt.xlabel("Iterations", fontsize=fontsize)
    plt.ylabel("Fitness", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(loc='best', fontsize=legend_fontsize)
    plt.tight_layout()
    plt.savefig('rhc_nn_parameter_tuning.png')
    plt.show()


def parameter_tuning_sa_nn(X_train, y_train):
    plt.figure(figsize=default_figure_size)
    exp_consts = [0.01, 0.1, 0.5, 1]
    init_temps = [10, 100, 1000, 10000]
    sa_nn_model_results = []

    t = 100
    x = 1
    for x in exp_consts:
        print("Running exp_const {}, init_t {}".format(x, t))
        schedule = mlrose.ExpDecay(init_temp=t, exp_const=x)
        temp_file_name = 'temp_NN_sa_init_temp_{}exp_const_{}.pickle'.format(x,t)

        if (os.path.isfile(temp_file_name)):
            print("WARNING: Not Running Loading: ", temp_file_name)
            with open(temp_file_name, 'rb') as handle:
                lr_nn_model1 = pickle.load(handle)
        else:
            lr_nn_model1 = mlrose.NeuralNetwork(hidden_nodes=[80], activation='relu', \
                                                algorithm='simulated_annealing', max_iters=350000, \
                                                bias=True, is_classifier=True, learning_rate=0.0015, \
                                                early_stopping=True, clip_max=5, max_attempts=200, \
                                                random_state=42, curve=True, schedule=schedule)
            print("Training")

            lr_nn_model1.fit(X_train, y_train)
            with open(temp_file_name, 'wb') as handle:
                pickle.dump(lr_nn_model1, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
        sa_nn_model_results.append(lr_nn_model1.fitness_curve.copy())

        y_train_pred = lr_nn_model1.predict(X_train)
        y_train_accuracy = accuracy_score(y_train, y_train_pred)
        print(y_train_accuracy)

    for x,i in zip(exp_consts, sa_nn_model_results):
        plt.plot(i, label="exp_const = {}".format(x))

    plt.title("Simulated Annealing Initial Temp {}\nNN Training Loss Curve".format(t), fontsize=title_fontsize, fontweight='bold')
    plt.xlabel("Iterations", fontsize=fontsize)
    plt.ylabel("Fitness", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(loc='best', fontsize=legend_fontsize)
    plt.tight_layout()
    plt.savefig('sa_nn_parameter_tuning.png')
    plt.show()

def parameter_tuning_ga_nn(X_train, y_train):
    plt.figure(figsize=default_figure_size)
    ga_nn_model_results = []
    p = 500
    pop_sizes = [10, 100, 500, 1000, 5000]
    x = 0.001
    mutation_probs = [0.0005, 0.001, 0.01, 0.1]
    for x in mutation_probs:
        print("Running pop_size {}, mutation_prob {}".format(p, x))
        temp_file_name = 'temp_NN_ga_pop_{}_mutation_prob_{}.pickle'.format(p, x)

        if (os.path.isfile(temp_file_name)):
            print("WARNING: Not Running Loading: ", temp_file_name)
            with open(temp_file_name, 'rb') as handle:
                lr_nn_model1 = pickle.load(handle)
        else:
            lr_nn_model1 = mlrose.NeuralNetwork(hidden_nodes=[80], activation='relu', \
                                                algorithm='genetic_alg', max_iters=1000, \
                                                bias=True, is_classifier=True, learning_rate=0.0015, \
                                                early_stopping=False, clip_max=5, max_attempts=200, \
                                                random_state=42, curve=True, pop_size=p, mutation_prob=x)
            print("Training")

            lr_nn_model1.fit(X_train, y_train)
            with open(temp_file_name, 'wb') as handle:
                pickle.dump(lr_nn_model1, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
        ga_nn_model_results.append(lr_nn_model1.fitness_curve.copy())

        y_train_pred = lr_nn_model1.predict(X_train)
        y_train_accuracy = accuracy_score(y_train, y_train_pred)
        print(y_train_accuracy)

    for x,i in zip(mutation_probs, ga_nn_model_results):
        plt.plot(i, label="mutation_prob = {}".format(x))

    plt.title("Genetic Algorithm Population Size {}\nNN Training Loss Curve".format(p), fontsize=title_fontsize, fontweight='bold')
    plt.xlabel("Iterations", fontsize=fontsize)
    plt.ylabel("Fitness", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(loc='best', fontsize=legend_fontsize)
    plt.tight_layout()
    plt.savefig('ga_nn_parameter_tuning.png')
    plt.show()


def cross_validation_best(nn_model, X, y):
    time = []
    accuracy = []
    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        start_time = time.time()
        nn_model.fit(X_train, y_train)
        time.append(time.time() - start_time)
        y_test_pred = nn_model.predict(X_test)
        y_train_accuracy = accuracy_score(y_test, y_test_pred)
        accuracy.append(y_train_accuracy)
    return time, accuracy


def get_training_and_accuracy(lr_nn_model1, X_train, y_train, X_test, y_test, y_test_non_noisy):
    print("Training")
    start_time = time.time()
    lr_nn_model1.fit(X_train, y_train)
    print("Time Taken To Train: {} s".format(time.time() - start_time))

    y_train_pred = lr_nn_model1.predict(X_train)
    y_train_accuracy = accuracy_score(y_train, y_train_pred)
    y_test_pred = lr_nn_model1.predict(X_test)
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    y_test_non_noisy_accuracy = accuracy_score(y_test_non_noisy, y_test_pred)

    print("Train Accuracy {} Test Accuracy {} Test Non Noisy Accruacy {}".format(y_train_accuracy, y_test_accuracy,
                                                                                 y_test_non_noisy_accuracy))


if __name__ == "__main__":
    import NeuralNetworks.common_NN as common_NN

    X_train, X_test, y_train, y_test, y_test_non_noisy = get_noisy_nonlinear_with_non_noisy_labels()
    num_features = X_train.shape[1]

    tmp = ['random_hill_climb', 'simulated_annealing', 'genetic_alg', 'gradient_descent']

    parameter_tuning_rhc_nn(X_train, y_train)
    gradient_decent_loss_curve(X_train, y_train)
    parameter_tuning_sa_nn(X_train, y_train)
    parameter_tuning_ga_nn(X_train, y_train)

    gd_max_iters=50000
    ga_pop_size = 500
    ga_mutation_prob = 0.001
    ga_max_iters = 1000
    rhc_restarts = 0
    rhc_max_iters = 350000
    sa_init_temp = 100
    sa_exp_const = 1
    sa_max_iters = 350000

    print("Gradient Descent")
    nn_model_gd = mlrose.NeuralNetwork(hidden_nodes=[80], activation='relu', \
                                        algorithm='gradient_descent', max_iters=gd_max_iters, \
                                        bias=True, is_classifier=True, learning_rate=0.0015, \
                                        early_stopping=True, clip_max=5, max_attempts=200, \
                                        random_state=42, curve=True)
    get_training_and_accuracy(nn_model_gd, X_train, y_train, X_test, y_test, y_test_non_noisy)

    print()
    print("Random Hill Climbing")
    nn_model_rhc = mlrose.NeuralNetwork(hidden_nodes=[80], activation='relu', \
                                        algorithm='random_hill_climb', max_iters=rhc_max_iters, \
                                        bias=True, is_classifier=True, learning_rate=0.0015, \
                                        early_stopping=True, max_attempts=200, \
                                        random_state=42, curve=True, restarts=rhc_restarts)
    get_training_and_accuracy(nn_model_rhc, X_train, y_train, X_test, y_test, y_test_non_noisy)

    print()
    print("Simulated Annealing")
    schedule = mlrose.ExpDecay(init_temp=sa_init_temp, exp_const=sa_exp_const)
    nn_model_sa = mlrose.NeuralNetwork(hidden_nodes=[80], activation='relu', \
                                        algorithm='simulated_annealing', max_iters=sa_max_iters, \
                                        bias=True, is_classifier=True, learning_rate=0.0015, \
                                        early_stopping=True, clip_max=5, max_attempts=200, \
                                        random_state=42, curve=True, schedule=schedule)
    get_training_and_accuracy(nn_model_sa, X_train, y_train, X_test, y_test, y_test_non_noisy)

    print()
    print("Genetic Algorithm")
    nn_model_ga = mlrose.NeuralNetwork(hidden_nodes=[80], activation='relu', \
                                        algorithm='genetic_alg', max_iters=ga_max_iters, \
                                        bias=True, is_classifier=True, learning_rate=0.0015, \
                                        early_stopping=False, clip_max=5, max_attempts=200, \
                                        random_state=42, curve=True, pop_size=ga_pop_size, mutation_prob=ga_mutation_prob)

    get_training_and_accuracy(nn_model_ga, X_train, y_train, X_test, y_test, y_test_non_noisy)

    with open("Best_gd_nn.pickle", 'wb') as handle:
        pickle.dump(nn_model_gd, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    with open("Best_rhc_nn.pickle", 'wb') as handle:
        pickle.dump(nn_model_rhc, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    with open("Best_sa_nn.pickle", 'wb') as handle:
        pickle.dump(nn_model_sa, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    with open("Best_ga_nn.pickle", 'wb') as handle:
        pickle.dump(nn_model_ga, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)