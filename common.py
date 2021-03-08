import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

title_fontsize = 24
fontsize = 24
legend_fontsize = 16
default_figure_size = (15,8)

def convert_bits_to_num(bit_array):
    return sum([i * 2 ** n for n, i in enumerate(bit_array)])

class timer():
    def __init__(self):
        self.start_time = time.time()

    def increment(self):
        return time.time() - self.start_time


def state_fitness_callback(iteration, state, fitness, user_data, done=False, attempt=0, curve=None):
    if done:
        return False
    else:
        if 'timer' in user_data:
            t = user_data['timer']
            user_data['time_taken'].append(t.increment())
        else:
            user_data['timer'] = timer()
            user_data['time_taken'] = [0]
        return True

def get_best_runs_from_runner(sa_run_stats, search_columns):
    # Get's the highest fitness for each parameter combination
    # https://stackoverflow.com/questions/15705630/get-the-rows-which-have-the-max-value-in-groups-using-groupby
    best_sa_run_stats = sa_run_stats.sort_values('Fitness', ascending=False).drop_duplicates(search_columns)
    return best_sa_run_stats


def plot_runner_results_fitness(best_sa_run_stats, ALGO, PROBLEM, param1, param2, legend_loc="best", figsize=default_figure_size):
    # https://stackoverflow.com/questions/40877135/plotting-two-columns-of-dataframe-in-seaborn
    fig, ax1 = plt.subplots(figsize=figsize)
    sns.set_context("paper", rc={"font.size": fontsize, "axes.titlesize": fontsize, "axes.labelsize": fontsize})

    sns.barplot(x=param1, y='Fitness', hue=param2, data=best_sa_run_stats, ax=ax1)

    ax1.legend(loc=legend_loc, title=param2, fontsize=legend_fontsize, fancybox=True, title_fontsize=legend_fontsize)

    title = '{} on {} Problem\n Parameter Tuning vs Fitness Score'.format(ALGO, PROBLEM)
    ax1.set_title(title,fontsize=title_fontsize, fontweight='bold')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    sns.despine(fig)

def plot_runner_results_time(best_sa_run_stats, ALGO, PROBLEM, param1, param2, legend_loc="best", figsize=default_figure_size):
    # https://stackoverflow.com/questions/40877135/plotting-two-columns-of-dataframe-in-seaborn
    fig, ax1 = plt.subplots(figsize=figsize)
    sns.set_context("paper", rc={"font.size": fontsize, "axes.titlesize": fontsize, "axes.labelsize": fontsize})

    sns.barplot(x=param1, y='Time', hue=param2, data=best_sa_run_stats, ax=ax1)

    ax1.legend(loc=legend_loc, title=param2, fontsize=legend_fontsize, fancybox=True, title_fontsize=legend_fontsize)

    title = '{} on {} Problem\n Parameter Tuning vs Time'.format(ALGO, PROBLEM)
    ax1.set_title(title,fontsize=title_fontsize, fontweight='bold')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    sns.despine(fig)

def plot_runner_results_both(best_sa_run_stats, ALGO, PROBLEM, param1, param2, legend_loc="best", figsize=default_figure_size):
    # https://stackoverflow.com/questions/40877135/plotting-two-columns-of-dataframe-in-seaborn
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=figsize)
    sns.set_context("paper", rc={"font.size": fontsize, "axes.titlesize": fontsize, "axes.labelsize": fontsize})

    sns.barplot(x=param1, y='Time', hue=param2, data=best_sa_run_stats, ax=ax1)
    sns.barplot(x=param1, y='Fitness', hue=param2, data=best_sa_run_stats, ax=ax2)

    ax1.legend(loc=legend_loc, title=param2, fontsize=legend_fontsize, fancybox=True, title_fontsize=legend_fontsize)

    x_axis = ax1.axes.get_xaxis()
    x_label = x_axis.get_label()
    x_label.set_visible(False)
    ax2.get_legend().remove()

    title = '{} on {} Problem\n Parameter Tuning for Time and Fitness'.format(ALGO, PROBLEM)
    ax1.set_title(title,fontsize=title_fontsize, fontweight='bold')

    plt.setp(ax1.get_yticklabels(), Fontsize=fontsize)
    plt.setp(ax2.get_yticklabels(), Fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    sns.despine(fig)


def plot_runner_results_fitness_1param(best_sa_run_stats, ALGO, PROBLEM, param1, figsize=default_figure_size):
    # https://stackoverflow.com/questions/40877135/plotting-two-columns-of-dataframe-in-seaborn
    fig, ax1 = plt.subplots(figsize=figsize)
    sns.set_context("paper", rc={"font.size": fontsize, "axes.titlesize": fontsize, "axes.labelsize": fontsize})

    sns.barplot(x=param1, y='Fitness', data=best_sa_run_stats, ax=ax1)

    title = '{} on {} Problem\n Parameter Tuning vs Fitness Score'.format(ALGO, PROBLEM)
    ax1.set_title(title,fontsize=title_fontsize, fontweight='bold')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    sns.despine(fig)

def plot_runner_results_time_1param(best_sa_run_stats, ALGO, PROBLEM, param1, figsize=default_figure_size):
    # https://stackoverflow.com/questions/40877135/plotting-two-columns-of-dataframe-in-seaborn
    fig, ax1 = plt.subplots(figsize=figsize)
    sns.set_context("paper", rc={"font.size": fontsize, "axes.titlesize": fontsize, "axes.labelsize": fontsize})

    sns.barplot(x=param1, y='Time', data=best_sa_run_stats, ax=ax1)

    title = '{} on {} Problem\n Parameter Tuning vs Time'.format(ALGO, PROBLEM)
    ax1.set_title(title,fontsize=title_fontsize, fontweight='bold')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    sns.despine(fig)

def plot_runner_results_both_1param(best_sa_run_stats, ALGO, PROBLEM, param1, figsize=default_figure_size):
    # https://stackoverflow.com/questions/40877135/plotting-two-columns-of-dataframe-in-seaborn
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=figsize)
    sns.set_context("paper", rc={"font.size": fontsize, "axes.titlesize": fontsize, "axes.labelsize": fontsize})

    sns.barplot(x=param1, y='Time', data=best_sa_run_stats, ax=ax1)
    sns.barplot(x=param1, y='Fitness', data=best_sa_run_stats, ax=ax2)

    x_axis = ax1.axes.get_xaxis()
    x_label = x_axis.get_label()
    x_label.set_visible(False)

    title = '{} on {} Problem\n Parameter Tuning for Time and Fitness'.format(ALGO, PROBLEM)
    ax1.set_title(title,fontsize=title_fontsize, fontweight='bold')

    plt.setp(ax1.get_yticklabels(), Fontsize=fontsize)
    plt.setp(ax2.get_yticklabels(), Fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    sns.despine(fig)

def plot_for_problem_size(results, labels, PROBLEM, y, problem_size, x=None, figsize=(20,8), legend_loc="best", linewidth=2.5, log_x=False):
    plt.figure(figsize=figsize)
    for r, l in zip(results, labels):
        index = r["problem_size"].index(problem_size)
        if x is not None:
            plt.plot(r[x][index], r[y][index], label=l, linewidth=linewidth)
            print(l, " max ", x," = ", max(r[x][index]), " max ", y, " = ", max(r[y][index]))
        else:
            plt.plot(r[y][index], label=l, linewidth=linewidth)
            print(l, " Max Iterations = ", " = ", len(r[y][index]), " Max ", y, " = ", max(r[y][index]))

    y = y.replace('_curve', '').capitalize()
    if x is not None:
        x = x.replace('_curve', '').capitalize()
        plt.xlabel(x, fontsize=fontsize)
        title = '{} Problem\n{} vs {}'.format(PROBLEM, y, x)
    else:
        plt.xlabel("Iterations", fontsize=fontsize)
        title = '{} Problem\n{} vs iterations'.format(PROBLEM, y)

    plt.title(title, fontsize=fontsize, fontweight='bold')
    plt.xticks(fontsize=fontsize)
    if log_x:
        plt.xscale('log')
    plt.yticks(fontsize=fontsize)
    plt.ylabel(y, fontsize=fontsize)
    plt.legend(loc=legend_loc, fontsize=legend_fontsize)
    plt.show()

def plot_vs_problem_size(results, labels, PROBLEM, y, figsize=default_figure_size, legend_loc="best", linewidth=2.5, log_x=False):
    plt.figure(figsize=figsize)
    for r, l in zip(results, labels):
        plt.plot(r['problem_size'], r[y], label=l, linewidth=linewidth)

    plt.xticks(fontsize=fontsize)
    if log_x:
        plt.xscale('log')
    plt.yticks(fontsize=fontsize)
    plt.xlabel("Problem Size", fontsize=fontsize)
    y = " ".join([i.capitalize() for i in y.split('_')])
    title = '{} Problem\n{} vs Problem Size'.format(PROBLEM, y)
    plt.title(title, fontsize=fontsize, fontweight='bold')
    plt.ylabel(y, fontsize=fontsize)
    plt.legend(loc=legend_loc, fontsize=legend_fontsize)
    plt.show()

def plot_for_problem_size_all(results, labels, PROBLEM, problem_size, figsize=(20,8), legend_loc="best", linewidth=2.5, log_x=[True, False]):
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=figsize)
    y = "fitness_curve"
    for r, l in zip(results, labels):
        index = r["problem_size"].index(problem_size)
        ax1.plot(r[y][index], label=l, linewidth=linewidth)
        print(l, " Max Iterations = ", " = ", len(r[y][index]), " Max ", y, " = ", max(r[y][index]))
    ax1.set_xlabel("Iterations")

    for r, l in zip(results, labels):
        index = r["problem_size"].index(problem_size)
        x = "call_curve"
        x = "time_curve"
        ax2.plot(r[x][index], r[y][index], label=l, linewidth=linewidth)
        print(l, " max ", x," = ", max(r[x][index]), " max ", y, " = ", max(r[y][index]))

    # ax2.set_xlabel("Fitness Function Calls")
    ax2.set_xlabel("Time Taken (s)")
    title = '{} Problem Comparison of Algorithms\nProblem Size {}'.format(PROBLEM, problem_size)
    ax1.set_ylabel("Fitness")

    fig.suptitle(title, fontsize=fontsize, fontweight='bold')
    y_axis = ax2.axes.get_yaxis()
    y_label = y_axis.get_label()
    y_label.set_visible(False)

    plt.setp(ax1.get_xticklabels(), Fontsize=fontsize)
    plt.setp(ax2.get_xticklabels(), Fontsize=fontsize)

    plt.setp(ax1.get_yticklabels(), Fontsize=fontsize)

    if log_x[0]:
        ax1.set_xscale('log')
    if log_x[1]:
        ax2.set_xscale('log')

    plt.ylabel("Fitness", fontsize=fontsize)
    ax2.legend(loc=legend_loc, fontsize=legend_fontsize, fancybox=True)
    fig.tight_layout()

    plt.show()

def plot_vs_problem_size_both(results, labels, PROBLEM, figsize=(20,6), legend_loc="best"):
    plot_dict = {"problem_size": [], 'label': [], 'max_function_calls': [], 'best_fitness': [], 'max_time': []}
    for r, l in zip(results, labels):
        [plot_dict['problem_size'].append(i) for i in r['problem_size']]
        [plot_dict['label'].append(i) for i in [l] * len(r['problem_size'])]
        [plot_dict['max_function_calls'].append(i) for i in r['max_function_calls']]
        [plot_dict['best_fitness'].append(i) for i in r['best_fitness']]
        [plot_dict['max_time'].append(i) for i in r['max_time']]

    param1 = 'problem_size'

    # https://stackoverflow.com/questions/40877135/plotting-two-columns-of-dataframe-in-seaborn
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=figsize)
    sns.set_context("paper", rc={"font.size": fontsize, "axes.titlesize": fontsize, "axes.labelsize": fontsize})

    sns.barplot(x=param1, y='max_function_calls', hue='label', data=plot_dict, ax=ax1)
    sns.barplot(x=param1, y='best_fitness', hue='label', data=plot_dict, ax=ax2)

    ax2.legend(loc=legend_loc, fontsize=legend_fontsize, fancybox=True)
    ax1.get_legend().remove()

    ax1.set_ylabel("Fitness Function Calls", fontsize=fontsize)
    ax2.set_ylabel("Fitness Value", fontsize=fontsize)

    x_axis = ax1.axes.get_xaxis()
    x_label = x_axis.get_label()
    x_label.set_visible(False)

    title = '{} Algorithm Comparison on Problem Size'.format(PROBLEM)
    ax1.set_title(title,fontsize=title_fontsize, fontweight='bold')

    plt.setp(ax1.get_yticklabels(), Fontsize=fontsize)
    plt.setp(ax2.get_yticklabels(), Fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    sns.despine(fig)

def plot_vs_problem_size_three(results, labels, PROBLEM, figsize=(20,8), legend_loc="best"):
    plot_dict = {"problem_size": [], 'label': [], 'max_function_calls': [], 'best_fitness': [], 'max_time': []}
    for r, l in zip(results, labels):
        [plot_dict['problem_size'].append(i) for i in r['problem_size']]
        [plot_dict['label'].append(i) for i in [l] * len(r['problem_size'])]
        [plot_dict['max_function_calls'].append(i) for i in r['max_function_calls']]
        [plot_dict['best_fitness'].append(i) for i in r['best_fitness']]
        [plot_dict['max_time'].append(i) for i in r['max_time']]

    param1 = 'problem_size'

    # https://stackoverflow.com/questions/40877135/plotting-two-columns-of-dataframe-in-seaborn
    # fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=figsize)

    # Create 2x2 sub plots
    gs = gridspec.GridSpec(2, 2)

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(gs[0, 0])  # row 0, col 0
    ax1.plot([0, 1])

    ax2 = fig.add_subplot(gs[0, 1])  # row 0, col 1
    ax2.plot([0, 1])

    ax3 = fig.add_subplot(gs[1, :])  # row 1, span all columns
    ax3.plot([0, 1])

    sns.set_context("paper", rc={"font.size": fontsize, "axes.titlesize": fontsize, "axes.labelsize": fontsize})

    sns.barplot(x=param1, y='max_function_calls', hue='label', data=plot_dict, ax=ax1)
    sns.barplot(x=param1, y='max_time', hue='label', data=plot_dict, ax=ax2)
    sns.barplot(x=param1, y='best_fitness', hue='label', data=plot_dict, ax=ax3)

    ax1.set_ylabel("Fitness Function Calls", fontsize=fontsize)
    ax2.set_ylabel("Total Time (s)", fontsize=fontsize)
    ax3.set_ylabel("Fitness Value", fontsize=fontsize)

    ax3.set_xlabel("Problem Size", fontsize=fontsize)

    x_axis = ax1.axes.get_xaxis()
    x_label = x_axis.get_label()
    x_label.set_visible(False)
    ax2.get_legend().remove()
    ax1.get_legend().remove()

    ax3.legend(loc=legend_loc, fontsize=legend_fontsize, fancybox=True)

    title = '{} Algorithm Comparison vs Problem Size'.format(PROBLEM)
    fig.suptitle(title, fontsize=fontsize, fontweight='bold')

    plt.setp(ax1.get_yticklabels(), Fontsize=fontsize)
    plt.setp(ax2.get_yticklabels(), Fontsize=fontsize)
    plt.setp(ax3.get_yticklabels(), Fontsize=fontsize)

    plt.setp(ax1.get_xticklabels(), Fontsize=fontsize)
    plt.setp(ax2.get_xticklabels(), Fontsize=fontsize)
    plt.setp(ax3.get_xticklabels(), Fontsize=fontsize)
    sns.despine(fig)
