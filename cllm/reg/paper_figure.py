import seaborn as sns
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
def read_results(target_folder, method):
    seeds = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]
    all_pdf_arr = []
    for seed in seeds:
        all_pdf_arr.append(pd.read_csv(target_folder / '{}_{}.csv'.format(method, seed)))
    all_pdf_arr = pd.concat(all_pdf_arr)
    return all_pdf_arr

def apply_base_settings():
    plt.clf()
    sns.set_style('whitegrid')
    # sns.set_palette("colorblind")
    # sns.set_palette(['#DAE8FC', '#FFE6CC', '#D5E8D4'])
    sns.set_palette(['#6C8EBF', '#D79B00', '#82B366'])
    # update the scale of the figure.
    sns.set_context("paper", font_scale=2)
    # Manually specify the space for the margin
    plt.subplots_adjust(left=0.3, right=0.95, top=0.95, bottom=0.3)

def produce_base_figure_coverage(dataset):
    # set paper white theme.
    apply_base_settings()
    # make the figure with larger text, should be ready for publication
    # plt.rcParams.update({'font.size': 34})
    if dataset == 'tbe':
        target_folder = Path('./data_out/paper_base/exp_result/tde2_base_noreg_all')
        file_name = 'tbe_no_reg_base_coverage.pdf'
    elif dataset == 'std':
        target_folder = Path('./data_out/paper_base/exp_result/std_base_noreg_all')
        file_name = 'std_no_reg_base_coverage.pdf'
    elif dataset == 'ds1k':
        target_folder = Path('./data_out/paper_base/exp_result/ds1k_base_noreg_all')
        file_name = 'ds1k_no_reg_base_coverage.pdf'
    elif dataset == 'ds10kall_deepseek_iodesc':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kall_deepseek_base_noreg_all_iodesc')
        file_name = 'ds10kall_deepseek_iodesc_base_coverage.pdf'
    else:
        raise NotImplementedError()
    method='base'
    all_pdf_arr = read_results(target_folder, method)
    # accuracy.
    pd_data = all_pdf_arr[all_pdf_arr['name'] == 'accuracy']
    pd_data['value'] = pd_data['value'].astype(float)
    pd_data['alpha'] = pd_data['alpha'].astype(float)
    pd_data['ealpha'] = 1 - pd_data['alpha']
    # set y label to accuracy
    fig, ax = plt.subplots()
    ax.set_ylabel('Coverage')
    ax.set_xlabel(r'$\alpha$')
    # we need to convert alpha to string for boxplot to overlay propoerly with lineplot
    pd_data['alpha'] = pd_data['alpha'].astype(str)
    ax = sns.boxplot(pd_data, whis=[0, 100], x='alpha', y='value', ax=ax, width=0.5)
    sns.lineplot(pd_data, x='alpha', y='ealpha', ax=ax, color='orange', marker='o', linewidth=3, markersize=10)
    # show the plot
    # plt.show()
    # save to file
    output_folder = Path('./data_out/paper_figures')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / file_name, bbox_inches='tight')

def produce_base_figures_coverage_all():
    # get all datasets.
    base_target_folder = Path('./data_out/paper_base/exp_result')
    datasets = [
        ('STD', 'std_base_noreg_all'),
        ('TDE', 'tde2_base_noreg_all'),
        # ('ds1k', 'DS1K', 'ds1k_base_noreg_all'),
        ('DS1K', 'ds10kall_deepseek_base_noreg_all_iodesc'),
    ]
    
    output_name = Path('all_base_coverage.pdf')

    apply_base_settings()
    
    def load_data(dataset, target_folder):
        all_pdf_arr = read_results(target_folder, 'base')
        # accuracy.
        pd_data = all_pdf_arr[all_pdf_arr['name'] == 'accuracy']
        pd_data['value'] = pd_data['value'].astype(float)
        pd_data['alpha'] = pd_data['alpha'].astype(float)
        pd_data['ealpha'] = 1 - pd_data['alpha']
        pd_data['dataset'] = dataset
        return pd_data
    
    all_data = []
    for dataset, target_folder in datasets:
        all_data.append(load_data(dataset, base_target_folder / target_folder))
    all_data = pd.concat(all_data)
    print(all_data)

    fig, ax = plt.subplots()
    ax.set_ylabel('Coverage')
    ax.set_xlabel(r'$\alpha$')
    # we need to convert alpha to string for boxplot to overlay propoerly with lineplot
    all_data['alpha'] = all_data['alpha'].astype(str)
    ax = sns.boxplot(all_data, whis=[0, 100], x='alpha', y='value', ax=ax, hue='dataset', width=0.5)
    sns.lineplot(all_data, x='alpha', y='ealpha', ax=ax, color='orange', marker='o', linewidth=3, markersize=10)
    # show the plot
    # plt.show()
    # save to file
    output_folder = Path('./data_out/paper_figures/all')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / output_name, bbox_inches='tight')

def produce_base_figure_size(dataset):
    # set paper white theme.
    apply_base_settings()
    if dataset == 'tbe':
        target_folder = Path('./data_out/paper_base/exp_result/tde2_base_noreg_all')
        file_name = 'tbe_no_reg_base_size.pdf'
        total_num = 225
    elif dataset == 'std':
        target_folder = Path('./data_out/paper_base/exp_result/std_base_noreg_all')
        file_name = 'std_no_reg_base_size.pdf'
        total_num = 60
    elif dataset == 'ds1k':
        target_folder = Path('./data_out/paper_base/exp_result/ds1k_base_noreg_all')
        file_name = 'ds1k_no_reg_base_size.pdf'
        total_num = 1000
    elif dataset == 'ds10kall_deepseek_iodesc':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kall_deepseek_base_noreg_all_iodesc')
        file_name = 'ds10kall_deepseek_iodesc_base_size.pdf'
        total_num = 771
    else:
        raise NotImplementedError()
    method='base'
    all_pdf_arr = read_results(target_folder, method)
    # size
    pd_data = all_pdf_arr[all_pdf_arr['name'] == 'avg_size']
    pd_data['value'] = pd_data['value'].astype(float) / total_num * 100
    pd_data['alpha'] = pd_data['alpha'].astype(float)
    # set y label to accuracy
    fig, ax = plt.subplots()
    ax.set_ylabel('Average Retrieval Size (%)')
    ax.set_xlabel(r'$\alpha$')
    # we need to convert alpha to string for boxplot to overlay propoerly with lineplot
    pd_data['alpha'] = pd_data['alpha'].astype(str)
    ax = sns.boxplot(pd_data, whis=[0, 100], x='alpha', y='value', ax=ax, width=0.3)
    # show the plot
    # plt.show()
    # save to file
    output_folder = Path('./data_out/paper_figures')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / file_name, bbox_inches='tight')

def produce_base_figures_size_all():
    apply_base_settings()
    base_target_folder = Path('./data_out/paper_base/exp_result')
    datasets = [
        ('STD', 'std_base_noreg_all', 60),
        ('TDE', 'tde2_base_noreg_all', 225),
        ('DS1K', 'ds10kall_deepseek_base_noreg_all_iodesc', 771),
    ]
    output_name = Path('all_base_size.pdf')

    def load_data(dataset, target_folder, total_num):
        all_pdf_arr = read_results(target_folder, 'base')
        # size
        pd_data = all_pdf_arr[all_pdf_arr['name'] == 'avg_size']
        pd_data['value'] = pd_data['value'].astype(float) / total_num * 100
        pd_data['alpha'] = pd_data['alpha'].astype(float)
        pd_data['dataset'] = dataset
        return pd_data

    all_data = []
    for dataset, target_folder, total_num in datasets:
        all_data.append(load_data(dataset, base_target_folder / target_folder, total_num))
    all_data = pd.concat(all_data)
    print(all_data)
    # plot the data
    fig, ax = plt.subplots()
    ax.set_ylabel('Avg Retrieval Percentage (%)')
    ax.set_xlabel(r'$\alpha$')
    sns.boxplot(all_data, x='alpha', y='value', ax=ax, hue='dataset', width=0.5)
    # hide legend title
    ax.legend_.set_title('')
    output_folder = Path('./data_out/paper_figures/all')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / output_name, bbox_inches='tight')

def produce_base_reg_noreg_size_compare(dataset):
    apply_base_settings()
    if dataset == 'tbe':
        target_folder = Path('./data_out/paper_base/exp_result/tde2_struct_test_noreg')
        file_name = 'tbe_reg_noreg_compare_base_size.pdf'
        total_num = 225
    elif dataset == 'std':
        target_folder = Path('./data_out/paper_base/exp_result/std_struct_mgt_as_one_test_noreg')
        file_name = 'std_reg_noreg_compare_base_size.pdf'
        total_num = 60
    else:
        raise NotImplementedError()
    method='base'
    all_pdf_arr = read_results(target_folder, method)
    # size
    pd_data = all_pdf_arr[all_pdf_arr['name'] == 'avg_size']
    pd_data['value'] = pd_data['value'].astype(float) / total_num * 100
    pd_data['alpha'] = pd_data['alpha'].astype(float) 
    pd_data['type'] = 'NoReg'

    all_data = [pd_data]

    # read reg results
    target_folder = Path('./data_out/paper_base/exp_result/tde2_struct_test_reg')
    method='base'
    all_pdf_arr = read_results(target_folder, method)
    # size
    pd_data = all_pdf_arr[all_pdf_arr['name'] == 'avg_size']
    pd_data['value'] = pd_data['value'].astype(float) / total_num * 100
    pd_data['alpha'] = pd_data['alpha'].astype(float) 
    pd_data['type'] = 'SVR'
    all_data.append(pd_data)

    # read neural results
    target_folder = Path('./data_out/paper_base/exp_result/tde2_struct_test_reg_neural')
    method='base'
    all_pdf_arr = read_results(target_folder, method)
    # size
    pd_data = all_pdf_arr[all_pdf_arr['name'] == 'avg_size']
    pd_data['value'] = pd_data['value'].astype(float) / total_num * 100
    pd_data['alpha'] = pd_data['alpha'].astype(float) 
    pd_data['type'] = 'MLP'
    all_data.append(pd_data)

    all_data = pd.concat(all_data)

    # we need to convert alpha to string for boxplot to overlay propoerly with lineplot
    all_data['alpha'] = all_data['alpha'].astype(str)
    # ax = sns.boxplot(all_data, whis=[0, 100], x='alpha', y='value', hue='type', width=0.8)
    
    fig, ax = plt.subplots()

    # sort aplha from 0.5 to 0.05
    all_data['alpha'] = pd.Categorical(all_data['alpha'], categories=sorted(all_data['alpha'].unique(), reverse=True))
    markers = ['o', 's', 'D']
    for i, group in enumerate(all_data['type'].unique()):
        group_data = all_data[all_data['type'] == group]
        
        # Calculate mean, min, and max for each x value
        summary = group_data.groupby('alpha')['value'].agg(['mean', 'min', 'max']).reset_index()
        
        # Plot mean line
        sns.lineplot(data=summary, x='alpha', y='mean', label=group, marker=markers[i], ax=ax, linewidth=3, markersize=10)
        
        # Fill between min and max
        ax.fill_between(summary['alpha'], summary['min'], summary['max'], alpha=0.2)

    # hide title for the legend
    ax.legend_.set_title('')
    # plot.
    ax.set_ylabel('Avg Retrieval Percentage (%)')
    ax.set_xlabel(r'$\alpha$')
    # show the plot
    output_folder = Path('./data_out/paper_figures')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / file_name, bbox_inches='tight')


def produce_base_reg_noreg_coverage_compare(dataset):
    apply_base_settings()
    if dataset == 'tbe':
        target_folder = Path('./data_out/paper_base/exp_result/tde2_struct_test_noreg')
        file_name = 'tbe_reg_noreg_compare_base_coverage.pdf'
    elif dataset == 'std':
        target_folder = Path('./data_out/paper_base/exp_result/std_struct_mgt_as_one_test_noreg')
        file_name = 'std_reg_noreg_compare_base_coverage.pdf'
    else:
        raise NotImplementedError()
    method='base'
    all_pdf_arr = read_results(target_folder, method)
    # size
    pd_data = all_pdf_arr[all_pdf_arr['name'] == 'accuracy']
    pd_data['value'] = pd_data['value'].astype(float)
    pd_data['alpha'] = pd_data['alpha'].astype(float) 
    pd_data['type'] = 'No Reg'

    all_data = [pd_data]

    # read reg results
    target_folder = Path('./data_out/paper_base/exp_result/tde2_struct_test_reg')
    method='base'
    all_pdf_arr = read_results(target_folder, method)
    # size
    pd_data = all_pdf_arr[all_pdf_arr['name'] == 'accuracy']
    pd_data['value'] = pd_data['value'].astype(float)
    pd_data['alpha'] = pd_data['alpha'].astype(float) 
    pd_data['type'] = 'SVR'
    all_data.append(pd_data)

    # read neural results
    target_folder = Path('./data_out/paper_base/exp_result/tde2_struct_test_reg_neural')
    method='base'
    all_pdf_arr = read_results(target_folder, method)
    # size
    pd_data = all_pdf_arr[all_pdf_arr['name'] == 'accuracy']
    pd_data['value'] = pd_data['value'].astype(float)
    pd_data['alpha'] = pd_data['alpha'].astype(float) 
    pd_data['type'] = 'MLP'
    all_data.append(pd_data)
    
    all_data = pd.concat(all_data)
    all_data['ealpha'] = 1 - all_data['alpha']

    # we need to convert alpha to string for boxplot to overlay propoerly with lineplot
    all_data['alpha'] = all_data['alpha'].astype(str)
    ax = sns.boxplot(all_data, whis=[0, 100], x='alpha', y='value', hue='type', width=0.8)
    # hide title for the legend
    ax.legend_.set_title('')
    # plot.
    ax.set_ylabel('Coverage')
    ax.set_xlabel(r'$\alpha$')
    sns.lineplot(all_data, x='alpha', y='ealpha', ax=ax, color='orange', marker='o', linewidth=3, markersize=10)
    # show the plot
    output_folder = Path('./data_out/paper_figures')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / file_name, bbox_inches='tight')


def produce_base_weight_compare(dataset):
    apply_base_settings()
    if dataset == 'tbe':
        target_folder = Path('./data_out/paper_base/exp_result/tde2_base_noreg_all')
        file_name = 'tbe_noreg_compare_base_weight.pdf'
    elif dataset == 'std':
        target_folder = Path('./data_out/paper_base/exp_result/std_base_noreg_all')
        file_name = 'std_noreg_compare_base_weight.pdf'
    else:
        raise NotImplementedError()
    method='base'
    all_pdf_arr = read_results(target_folder, method)
    # size
    pd_data = all_pdf_arr[all_pdf_arr['name'] == 'accuracy']
    pd_data['value'] = pd_data['value'].astype(float)
    pd_data['alpha'] = pd_data['alpha'].astype(float) 
    pd_data['type'] = 'Base'

    all_data = [pd_data]

    # read weight result.
    method='base_weight'
    all_pdf_arr = read_results(target_folder, method)
    # size
    pd_data = all_pdf_arr[all_pdf_arr['name'] == 'accuracy']
    pd_data['value'] = pd_data['value'].astype(float)
    pd_data['alpha'] = pd_data['alpha'].astype(float) 
    pd_data['type'] = 'Weight'
    all_data.append(pd_data)

    all_data = pd.concat(all_data)
    all_data['ealpha'] = 1 - all_data['alpha']
    fig, ax = plt.subplots()

    # we need to convert alpha to string for boxplot to overlay propoerly with lineplot
    
    all_data['alpha'] = all_data['alpha'].astype(str)

    sns.boxplot(all_data, whis=[0, 100], ax=ax, x='alpha', y='value', hue='type', width=0.6)
    # plot.
    ax.set_ylabel('Coverage')
    ax.set_xlabel(r'$\alpha$')
    
    sns.lineplot(all_data, x='alpha', y='ealpha', ax=ax, color='orange', marker='o', linewidth=3, markersize=10)
    # show the plot
    output_folder = Path('./data_out/paper_figures')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / file_name, bbox_inches='tight')

def produce_base_group_cp(dataset, group_num):
    # show score
    apply_base_settings()
    if dataset == 'tbe':
        target_folder = Path('./data_out/paper_base/exp_result/tde2_base_noreg_all')
        file_name = f'tbe_noreg_groupcp_{group_num}.pdf'
    elif dataset == 'std':
        target_folder = Path('./data_out/paper_base/exp_result/std_base_noreg_all')
        file_name = f'std_noreg_groupcp_{group_num}.pdf'
    else:
        raise NotImplementedError()
    method = f'group_{group_num}'
    all_pdf_arr = read_results(target_folder, method)
    # size
    pd_data = all_pdf_arr[all_pdf_arr['name'] == 'gp_num']
    pd_data['alpha'] = pd_data['alpha'].astype(float) 
    
    # count the # of groups.
    def count_groups(x):
        arr = eval(x)
        return sum([1 if x > 0 else 0 for x in arr])
    pd_data['gnum'] = pd_data['value'].apply(count_groups)

    # group by alpha, and count the # of groups for group 1 and group 2.
    for i in range(1, group_num + 1):
        pd_data[f'G{i}'] = pd_data['gnum'].apply(lambda x: x == i)
    pd_data = pd_data.groupby('alpha').sum().reset_index()

    # print(pd_data)

    # flatten the data

    data = []
    for _id, row in pd_data.iterrows():
        for i in range(1, group_num + 1):
            data.append({'alpha': row['alpha'], 'count': row[f'G{i}'], 'Group #': f'{i}'})
    pd_data = pd.DataFrame(data)

    alpha_order = [f'{0.5 - 0.1 * i:.1f}' for i in range(5)]
    alpha_order = [*alpha_order, '0.05']

    # plot the number of groups for varying alpha.
    fig, ax = plt.subplots()
    sns.barplot(pd_data, x='alpha', y='count', ax=ax, hue='Group #', order=alpha_order)
    ax.set_ylabel('# of Runs')
    ax.set_xlabel(r'$\alpha$')
    # oreder the x labels from 0.5 to 0.1
    # ax.set_xticks(range(5))
    # ax.set_xticklabels([f'{0.5 - 0.1 * i:.1f}' for i in range(5)])
    # show the plot
    output_folder = Path('./data_out/paper_figures')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / file_name, bbox_inches='tight')

def produce_abstain_rate_general_(target_folder, method, file_name):
    apply_base_settings()
    all_pdf_arr = read_results(target_folder, method)
    pd_data = all_pdf_arr[all_pdf_arr['name'] == 'gp_num']
    pd_data['alpha'] = pd_data['alpha'].astype(float)
    pd_data['value'] = pd_data['value'].apply(eval)

    pd_data['abstain_rate'] = pd_data['value'].apply(lambda x: x[1] / (x[0] + x[1]))

    # plot the abstain rate for varying alpha.
    fig, ax = plt.subplots()
    alpha_order = [f'{0.5 - 0.1 * i:.1f}' for i in range(5)]
    alpha_order = [*alpha_order, '0.05']

    sns.boxplot(pd_data, x='alpha', y='abstain_rate', ax=ax, order=alpha_order, whis=[0, 100])
    ax.set_ylim(0, 0.4)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Abstain Rate')
    # plot the line of 1 - alpha
    sns.lineplot(x=alpha_order, y=0.2, ax=ax, color='orange', marker='o', linewidth=3, markersize=10)
    # save to file
    output_folder = Path('./data_out/paper_figures')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / file_name, bbox_inches='tight')

def produce_abstain_rate_computeq(dataset):
    apply_base_settings()
    if dataset == 'tbe':
        target_folder = Path('./data_out/paper_base/exp_result/tde2_base_noreg_all')
        file_name = 'tbe_noreg_abstain_rate_computeq.pdf'
    elif dataset == 'std':
        target_folder = Path('./data_out/paper_base/exp_result/std_base_noreg_all')
        file_name = 'std_noreg_abstain_rate_computeq.pdf'
    elif dataset == 'ds1k':
        target_folder = Path('./data_out/paper_base/exp_result/ds1k_base_noreg_all')
        file_name = 'ds1k_noreg_abstain_rate_computeq.pdf'
    elif dataset == 'ds10kbase_deepseek_iodesc  ':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kbase_deepseek_base_noreg_all_iodesc')
        file_name = 'ds10kbase_deepseek_noreg_abstain_rate_computeq_iodesc.pdf'
    elif dataset == 'ds10kall_deepseek_iodesc':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kall_deepseek_base_noreg_all_iodesc')
        file_name = 'ds10kall_deepseek_noreg_abstain_rate_computeq_iodesc.pdf'
    else:
        raise NotImplementedError()
    method = 'fixed_group_computeq_2_8_2'
    produce_abstain_rate_general_(target_folder, method, file_name)

def produce_abstain_rate_computeq_all():
    apply_base_settings()
    base_target_folder = Path('./data_out/paper_base/exp_result')
    datasets = [
        ('TDE', 'tde2_base_noreg_all', 225),
        ('STD', 'std_base_noreg_all', 60),
        ('DS1K', 'ds10kall_deepseek_base_noreg_all_iodesc', 771),
    ]

    output_name = 'all_abstain_rate_computeq.pdf'

    # load data
    def load_data(dataset, target_folder, total_num):
        all_pdf_arr = read_results(target_folder, 'fixed_group_computeq_2_8_2')
        pd_data = all_pdf_arr[all_pdf_arr['name'] == 'gp_num']
        pd_data['alpha'] = pd_data['alpha'].astype(float)
        pd_data['value'] = pd_data['value'].apply(eval)

        pd_data['abstain_rate'] = pd_data['value'].apply(lambda x: x[1] / (x[0] + x[1]))
        pd_data['dataset'] = dataset
        return pd_data

    all_data = []
    for dataset, target_folder, total_num in datasets:
        pd_data = load_data(dataset, target_folder, total_num)
        all_data.append(pd_data)

    # plot the abstain rate for varying alpha.
    fig, ax = plt.subplots()
    alpha_order = [f'{0.5 - 0.1 * i:.1f}' for i in range(5)]
    alpha_order = [*alpha_order, '0.05']

    sns.boxplot(all_data, x='alpha', y='abstain_rate', ax=ax, order=alpha_order, hue='dataset', whis=[0, 100])
    ax.set_ylim(0, 0.4)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Abstain Rate')
    # plot the line of 1 - alpha
    sns.lineplot(x=alpha_order, y=0.2, ax=ax, color='orange', marker='o', linewidth=3, markersize=10)
    # hide the legend
    ax.legend_.set_visible(False)
    # save to file
    output_folder = Path('./data_out/paper_figures/all')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / output_name, bbox_inches='tight')

def produce_abstain_rate_learnq(dataset):
    apply_base_settings()
    if dataset == 'tbe':
        target_folder = Path('./data_out/paper_base/exp_result/tde2_base_noreg_all')
        file_name = 'tbe_noreg_abstain_rate_learnq.pdf'
    elif dataset == 'std':
        target_folder = Path('./data_out/paper_base/exp_result/std_base_noreg_all')
        file_name = 'std_noreg_abstain_rate_learnq.pdf'
    elif dataset == 'ds1k':
        target_folder = Path('./data_out/paper_base/exp_result/ds1k_base_noreg_all')
        file_name = 'ds1k_noreg_abstain_rate_learnq.pdf'
    elif dataset == 'ds10kbase_deepseek_iodesc':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kbase_deepseek_base_noreg_all_iodesc')
        file_name = 'ds10kbase_deepseek_noreg_abstain_rate_learnq_iodesc.pdf'
    elif dataset == 'ds10kall_deepseek_iodesc':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kall_deepseek_base_noreg_all_iodesc')
        file_name = 'ds10kall_deepseek_noreg_abstain_rate_learnq_iodesc.pdf'
    else:
        raise NotImplementedError()
    method = 'fixed_group_learnq_2_8_2'
    produce_abstain_rate_general_(target_folder, method, file_name)


# add missing params
def produce_base_group_cp_size_general_(target_folder, method, file_name, sort=False, total_num=None):
    all_pdf_arr = read_results(target_folder, method)
    # size
    pd_data = all_pdf_arr[all_pdf_arr['name'] == 'gp_avg_sizes']
    pd_data['alpha'] = pd_data['alpha'].astype(float) 
    if sort:
        pd_data['value'] = pd_data['value'].apply(lambda x:sorted(eval(x)))
    else:
        pd_data['value'] = pd_data['value'].apply(eval)
    pd_data['type'] = 'CPLF'
    # only keep the data if each element in gp_avg_sizes is not 0
    pd_data = pd_data[pd_data['value'].apply(lambda x: all(i != 0 for i in x))]
    # put G1, G2 columns
    pd_data['G1'] = pd_data['value'].apply(lambda x: x[0] / total_num * 100)
    pd_data['G2'] = pd_data['value'].apply(lambda x: x[1] / total_num * 100)

    # flatten the data
    data = []
    for _id, row in pd_data.iterrows():
        for i in range(1, 3):
            if i == 1:
                g = 'B'
            elif i == 2:
                g = 'A'
            data.append({'alpha': row['alpha'], 'size': row[f'G{i}'], 'Group': f'{g}'})
    pd_data = pd.DataFrame(data)

    # plot the number of groups for varying alpha.
    fig, ax = plt.subplots()

    alpha_order = [f'{0.5 - 0.1 * i:.1f}' for i in range(5)]
    alpha_order = [*alpha_order, '0.05']

    sns.boxplot(pd_data, whis=[0, 100], x='alpha', y='size', ax=ax, hue='Group', order=alpha_order)
    # set xlabel $\alpha$
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Avg Retrieval Percentage (%)')
    # oreder the x labels from 0.5 to 0.1
    # save to file
    output_folder = Path('./data_out/paper_figures')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / file_name, bbox_inches='tight')

def produce_base_group_cp_group_size(dataset):
    apply_base_settings()
    if dataset == 'tbe':
        target_folder = Path('./data_out/paper_base/exp_result/tde2_base_noreg_all')
        file_name = f'tbe_noreg_abstain_size_base.pdf'
        total_num = 225
    elif dataset == 'std':
        target_folder = Path('./data_out/paper_base/exp_result/std_base_noreg_all')
        file_name = f'std_noreg_abstain_size_base.pdf'
        total_num = 60
    else:
        raise NotImplementedError()

    method = f'group_2'
    produce_base_group_cp_size_general_(target_folder, method, file_name, sort=True, total_num=total_num)


def produce_base_fixgroup_group_size(dataset):
    apply_base_settings()
    if dataset == 'tbe':
        target_folder = Path('./data_out/paper_base/exp_result/tde2_base_noreg_all')
        file_name = f'tbe_noreg_abstain_size_learnq.pdf'    
        total_num = 225
    elif dataset == 'std':
        target_folder = Path('./data_out/paper_base/exp_result/std_base_noreg_all')
        file_name = f'std_noreg_abstain_size_learnq.pdf'
        total_num = 60
    elif dataset == 'ds1k':
        target_folder = Path('./data_out/paper_base/exp_result/ds1k_base_noreg_all')
        file_name = f'ds1k_noreg_abstain_size_learnq.pdf'
        total_num = 1000
    elif dataset == 'ds10kbase_v2':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kbase_v2_fix_alpha_noreg_all')
        file_name = f'ds10kbase_v2_noreg_abstain_size_learnq.pdf'
        total_num = 390
    elif dataset == 'ds10kall_v2':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kall_v2_fix_alpha_noreg_all')
        file_name = f'ds10kall_v2_noreg_abstain_size_learnq.pdf'
        total_num = 782
    elif dataset == 'ds10kbase_deepseek':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kbase_deepseek_fix_alpha_noreg_all')
        file_name = f'ds10kbase_deepseek_noreg_abstain_size_learnq.pdf'
        total_num = 377
    elif dataset == 'ds10kall_deepseek':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kall_deepseek_fix_alpha_noreg_all')
        file_name = f'ds10kall_deepseek_noreg_abstain_size_learnq.pdf'
        total_num = 771
    elif dataset == 'ds10kbase_deepseek_iodesc':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kbase_deepseek_base_noreg_all_iodesc')
        file_name = f'ds10kbase_deepseek_noreg_abstain_size_learnq_iodesc.pdf'
        total_num = 377
    elif dataset == 'ds10kall_deepseek_iodesc':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kall_deepseek_base_noreg_all_iodesc')
        file_name = f'ds10kall_deepseek_noreg_abstain_size_learnq_iodesc.pdf'
        total_num = 771
    else:
        raise NotImplementedError()
    method = 'fixed_group_learnq_2_8_2'
    produce_base_group_cp_size_general_(target_folder, method, file_name, total_num=total_num)


def produce_base_fixgroupcp_group_size(dataset):
    apply_base_settings()
    if dataset == 'tbe':
        target_folder = Path('./data_out/paper_base/exp_result/tde2_base_noreg_all')
        file_name = f'tbe_noreg_abstain_size_computeq.pdf'
        total_num = 225
    elif dataset == 'std':
        target_folder = Path('./data_out/paper_base/exp_result/std_base_noreg_all')
        file_name = f'std_noreg_abstain_size_computeq.pdf'
        total_num = 60
    elif dataset == 'ds1k':
        target_folder = Path('./data_out/paper_base/exp_result/ds1k_base_noreg_all')
        file_name = f'ds1k_noreg_abstain_size_computeq.pdf'
        total_num = 1000
    elif dataset == 'ds10kbase_v2':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kbase_v2_base_noreg_all')
        file_name = f'ds10kbase_v2_noreg_abstain_size_computeq.pdf'
        total_num = 390
    elif dataset == 'ds10kall_v2':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kall_v2_base_noreg_all')
        file_name = f'ds10kall_v2_noreg_abstain_size_computeq.pdf'
        total_num = 782
    elif dataset == 'ds10kbase_deepseek':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kbase_deepseek_base_noreg_all')
        file_name = f'ds10kbase_deepseek_noreg_abstain_size_computeq.pdf'
        total_num = 377
    elif dataset == 'ds10kall_deepseek':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kall_deepseek_base_noreg_all')
        file_name = f'ds10kall_deepseek_noreg_abstain_size_computeq.pdf'
        total_num = 771
    elif dataset == 'ds10kbase_deepseek_iodesc':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kbase_deepseek_base_noreg_all_iodesc')
        # target_folder = Path('./data_out/paper_base/exp_result/ds10kbase_deepseek_fix_alpha_noreg_all_iodesc')
        file_name = f'ds10kbase_deepseek_noreg_abstain_size_computeq_iodesc.pdf'
        total_num = 377
    elif dataset == 'ds10kall_deepseek_iodesc':
        # target_folder = Path('./data_out/paper_base/exp_result/ds10kall_deepseek_fix_alpha_noreg_all_iodesc')
        target_folder = Path('./data_out/paper_base/exp_result/ds10kall_deepseek_base_noreg_all_iodesc')
        file_name = f'ds10kall_deepseek_noreg_abstain_size_computeq_iodesc.pdf'
        total_num = 771
    elif dataset == 'ds10kall_deepseek_iodesc_st':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kall_deepseek_base_noreg_all_iodesc_st')
        file_name = f'ds10kall_deepseek_noreg_abstain_size_computeq_iodesc_st.pdf'
        total_num = 771
    else:
        raise NotImplementedError()
    method = 'fixed_group_computeq_2_8_2'
    produce_base_group_cp_size_general_(target_folder, method, file_name, total_num=total_num)

def produce_base_group_cp_coverage_general_(target_folder, method, file_name):
    all_pdf_arr = read_results(target_folder, method)
    # size
    pd_data = all_pdf_arr[all_pdf_arr['name'] == 'gp_coverages']
    pd_data['alpha'] = pd_data['alpha'].astype(float) 
    pd_data['value'] = pd_data['value'].apply(eval)
    pd_data['type'] = 'CPLF'
    # only keep the data if each element in gp_avg_sizes is not 0
    pd_data = pd_data[pd_data['value'].apply(lambda x: all(i != 0 for i in x))]
    pd_data['G1'] = pd_data['value'].apply(lambda x: x[0])
    pd_data['G2'] = pd_data['value'].apply(lambda x: x[1])
    # only G1 matters.

    pd_data['ealpha'] = 1 - pd_data['alpha']
    pd_data['alpha'] = pd_data['alpha'].astype(str)

    # plot the number of groups for varying alpha.
    fig, ax = plt.subplots()

    alpha_order = [f'{0.5 - 0.1 * i:.1f}' for i in range(5)]
    alpha_order = [*alpha_order, '0.05']

    sns.boxplot(pd_data, whis=[0, 100], x='alpha', y='G1', ax=ax, order=alpha_order)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Coverage')
    # oreder the x labels from 0.5 to 0.1
    # ax.set_xticks(range(5))
    # ax.set_xticklabels([f'{0.5 - 0.1 * i:.1f}' for i in range(5)])
    # plot the line of 1 - alpha
    sns.lineplot(pd_data, x='alpha', y='ealpha', ax=ax, color='orange', marker='o', linewidth=3, markersize=10)

    # save to file
    output_folder = Path('./data_out/paper_figures')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / file_name, bbox_inches='tight')

def produce_base_group_cp_coverage_base(dataset):
    apply_base_settings()
    if dataset == 'tbe':
        target_folder = Path('./data_out/paper_base/exp_result/tde2_base_noreg_all')
        file_name = f'tbe_noreg_abstain_coverage_base.pdf'
    elif dataset == 'std':
        target_folder = Path('./data_out/paper_base/exp_result/std_base_noreg_all')
        file_name = f'std_noreg_abstain_coverage_base.pdf'
    elif dataset == 'ds1k':
        target_folder = Path('./data_out/paper_base/exp_result/ds1k_base_noreg_all')
        file_name = f'ds1k_noreg_abstain_coverage_base.pdf'
    else:
        raise NotImplementedError() 
    method = 'group_2'
    produce_base_group_cp_coverage_general_(target_folder, method, file_name)

def produce_base_group_cp_coverage_learnq(dataset):
    apply_base_settings()
    if dataset == 'tbe':
        target_folder = Path('./data_out/paper_base/exp_result/tde2_base_noreg_all')
        file_name = f'tbe_noreg_abstain_coverage_learnq.pdf'
    elif dataset == 'std':
        target_folder = Path('./data_out/paper_base/exp_result/std_base_noreg_all')
        file_name = f'std_noreg_abstain_coverage_learnq.pdf'
    elif dataset == 'ds1k':
        target_folder = Path('./data_out/paper_base/exp_result/ds1k_base_noreg_all')
        file_name = f'ds1k_noreg_abstain_coverage_learnq.pdf'
    elif dataset == 'ds10kbase_deepseek_iodesc':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kbase_deepseek_base_noreg_all_iodesc')
        file_name = f'ds10kbase_deepseek_noreg_abstain_coverage_learnq_iodesc.pdf'
    elif dataset == 'ds10kall_deepseek_iodesc':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kall_deepseek_base_noreg_all_iodesc')
        file_name = f'ds10kall_deepseek_noreg_abstain_coverage_learnq_iodesc.pdf'
    elif dataset == 'ds10kall_deepseek_iodesc_st':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kall_deepseek_base_noreg_all_iodesc_st')
        file_name = f'ds10kall_deepseek_noreg_abstain_coverage_learnq_iodesc_st.pdf'
    else:
        raise NotImplementedError() 
    method = 'fixed_group_learnq_2_8_2'
    produce_base_group_cp_coverage_general_(target_folder, method, file_name)

def produce_base_group_cp_coverage_all_general_(method, output_name):
    apply_base_settings()
    base_target_folder = Path('./data_out/paper_base/exp_result')
    datasets = [
        ('STD', 'std_base_noreg_all'),
        ('TDE', 'tde2_base_noreg_all'),
        ('DS1K', 'ds10kall_deepseek_base_noreg_all_iodesc'),
    ]

    def load_data(dataset, target_folder):
        all_pdf_arr = read_results(target_folder, method)
        pd_data = all_pdf_arr[all_pdf_arr['name'] == 'gp_coverages']
        pd_data['alpha'] = pd_data['alpha'].astype(float) 
        pd_data['value'] = pd_data['value'].apply(eval)
        pd_data['type'] = 'CPLF'
        # only keep the data if each element in gp_avg_sizes is not 0
        pd_data = pd_data[pd_data['value'].apply(lambda x: all(i != 0 for i in x))]
        pd_data['G1'] = pd_data['value'].apply(lambda x: x[0])
        pd_data['G2'] = pd_data['value'].apply(lambda x: x[1])
        # only G1 matters.

        pd_data['ealpha'] = 1 - pd_data['alpha']
        pd_data['alpha'] = pd_data['alpha'].astype(str)
        pd_data['dataset'] = dataset
        return pd_data
    all_data = []
    for dataset, target_folder in datasets:
        target_folder = base_target_folder / target_folder
        all_data.append(load_data(dataset, target_folder))
    all_data = pd.concat(all_data)
    
    fig, ax = plt.subplots()

    alpha_order = [f'{0.5 - 0.1 * i:.1f}' for i in range(5)]
    alpha_order = [*alpha_order, '0.05']

    sns.boxplot(all_data, whis=[0, 100], x='alpha', y='G1', ax=ax, hue='dataset', order=alpha_order)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Coverage')
    # oreder the x labels from 0.5 to 0.1
    # ax.set_xticks(range(5))
    # ax.set_xticklabels([f'{0.5 - 0.1 * i:.1f}' for i in range(5)])
    # plot the line of 1 - alpha
    sns.lineplot(all_data, x='alpha', y='ealpha', ax=ax, color='orange', marker='o', linewidth=3, markersize=10)

    # save to file
    output_folder = Path('./data_out/paper_figures/all')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / output_name, bbox_inches='tight')

def produce_base_group_cp_coverage_computeq(dataset):
    apply_base_settings()
    if dataset == 'tbe':
        target_folder = Path('./data_out/paper_base/exp_result/tde2_base_noreg_all')
        file_name = f'tbe_noreg_abstain_coverage_computeq.pdf'
    elif dataset == 'std':
        target_folder = Path('./data_out/paper_base/exp_result/std_base_noreg_all')
        file_name = f'std_noreg_abstain_coverage_computeq.pdf'
    elif dataset == 'ds1k':
        target_folder = Path('./data_out/paper_base/exp_result/ds1k_base_noreg_all')
        file_name = f'ds1k_noreg_abstain_coverage_computeq.pdf'
    elif dataset == 'ds10kbase_deepseek_iodesc':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kbase_deepseek_base_noreg_all_iodesc')
        file_name = f'ds10kbase_deepseek_noreg_abstain_coverage_computeq_iodesc.pdf'
    elif dataset == 'ds10kall_deepseek_iodesc':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kall_deepseek_base_noreg_all_iodesc')
        file_name = f'ds10kall_deepseek_noreg_abstain_coverage_computeq_iodesc.pdf'
    elif dataset == 'ds10kall_deepseek_iodesc_st':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kall_deepseek_base_noreg_all_iodesc_st')
        file_name = f'ds10kall_deepseek_noreg_abstain_coverage_computeq_iodesc_st.pdf'
    else:
        raise NotImplementedError() 
    method = 'fixed_group_computeq_2_8_2'
    produce_base_group_cp_coverage_general_(target_folder, method, file_name)

def produce_base_group_cp_coverage_computeq_all():
    produce_base_group_cp_coverage_all_general_('fixed_group_computeq_2_8_2', 'all_abstain_coverage_computeq.pdf')

def produce_base_group_cp_coverage_learnq_all():
    produce_base_group_cp_coverage_all_general_('fixed_group_learnq_2_8_2', 'all_abstain_coverage_learnq.pdf')

def produce_base_group_cp_coverage_computeq_weight(dataset):
    apply_base_settings()
    if dataset == 'tbe':
        target_folder = Path('./data_out/paper_base/exp_result/tde2_base_noreg_all')
        file_name = f'tbe_noreg_abstain_coverage_computeq_weight.pdf'
    elif dataset == 'std':
        target_folder = Path('./data_out/paper_base/exp_result/std_base_noreg_all')
        file_name = f'std_noreg_abstain_coverage_computeq_weight.pdf'
    else:
        raise NotImplementedError() 
    method = 'fixed_group_computeq_weight_2_8_2'
    produce_base_group_cp_coverage_general_(target_folder, method, file_name)


def produce_groupcp_computeq_group_samplenum(dataset):
    apply_base_settings()
    if dataset == 'tbe':
        target_folder = Path('./data_out/paper_base/exp_result/tde2_fix_alpha_noreg_all')
        file_name = f'tbe_noreg_abstain_percent_computeq_compare_varyratio.pdf'
    elif dataset == 'std':
        target_folder = Path('./data_out/paper_base/exp_result/std_fix_alpha_noreg_all')
        file_name = f'std_noreg_abstain_percent_computeq_compare_varyratio.pdf'
    elif dataset == 'ds1k':
        target_folder = Path('./data_out/paper_base/exp_result/ds1k_fix_alpha_noreg_all')
        file_name = f'ds1k_noreg_abstain_percent_computeq_compare_varyratio.pdf'
    elif dataset == 'ds10kbase_deepseek_iodesc':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kbase_deepseek_fix_alpha_noreg_all_iodesc')
        file_name = f'ds10kbase_deepseek_noreg_abstain_percent_computeq_compare_varyratio_iodesc.pdf'
    elif dataset == 'ds10kall_deepseek_iodesc':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kall_deepseek_fix_alpha_noreg_all_iodesc')
        file_name = f'ds10kall_deepseek_noreg_abstain_percent_computeq_compare_varyratio_iodesc.pdf'
    else:
        raise NotImplementedError() 
    methods = [
        'fixed_group_computeq_2_9_1',
        'fixed_group_computeq_2_8_2',
        'fixed_group_computeq_2_7_3',
        'fixed_group_computeq_2_6_4',
        'fixed_group_computeq_2_5_5',
    ]
    all_data = []
    for method in methods:
        all_pdf_arr = read_results(target_folder, method)
        pd_data = all_pdf_arr[all_pdf_arr['name'] == 'gp_num']
        pd_data['value'] = pd_data['value'].apply(eval)
        pd_data['alpha'] = pd_data['alpha'].astype(float) 
        # percent
        pd_data['ratio'] = '{:.1f}'.format(int(method[-1]) * 0.1)
        pd_data['G1'] = pd_data['value'].apply(lambda x: x[0] / sum(x))
        pd_data['G2'] = pd_data['value'].apply(lambda x: x[1] / sum(x))
        pd_data['G1'] = pd_data['G1'].apply(lambda x: x * 100)
        pd_data['G2'] = pd_data['G2'].apply(lambda x: x * 100)
        
        all_data.append(pd_data)
    
    all_data = pd.concat(all_data)

    # flatten the data
    data = []
    for _id, row in all_data.iterrows():
        for i in range(1, 3):
            if i == 1:
                g = 'B'
            elif i == 2:
                g = 'A'
            data.append({'ratio': row['ratio'], 'count': row[f'G{i}'], 'Group': f'{g}'})
    all_data = pd.DataFrame(data)

    fig, ax = plt.subplots()
    sns.boxplot(all_data, whis=[0, 100], x='ratio', y='count', hue='Group')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel('Percentage of Samples (%)')   
    # store to file
    output_folder = Path('./data_out/paper_figures')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / file_name, bbox_inches='tight')


def produce_groupcp_computeq_coverage(dataset):
    apply_base_settings()
    if dataset == 'tbe':
        target_folder = Path('./data_out/paper_base/exp_result/tde2_fix_alpha_noreg_all')
        file_name = f'tbe_noreg_abstain_coverage_computeq_compare_varyratio.pdf'
    elif dataset == 'std':
        target_folder = Path('./data_out/paper_base/exp_result/std_fix_alpha_noreg_all')
        file_name = f'std_noreg_abstain_coverage_computeq_compare_varyratio.pdf'
    elif dataset == 'ds1k':
        target_folder = Path('./data_out/paper_base/exp_result/ds1k_fix_alpha_noreg_all')
        file_name = f'ds1k_noreg_abstain_coverage_computeq_compare_varyratio.pdf'
    elif dataset == 'ds10kbase':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kbase_fix_alpha_noreg_all')
        file_name = f'ds10kbase_noreg_abstain_coverage_computeq_compare_varyratio.pdf'
    elif dataset == 'ds10kall':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kall_fix_alpha_noreg_all')
        file_name = f'ds10kall_noreg_abstain_coverage_computeq_compare_varyratio.pdf'
    elif dataset == 'ds10kbase_v2':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kbase_v2_fix_alpha_noreg_all')
        file_name = f'ds10kbase_v2_noreg_abstain_coverage_computeq_compare_varyratio.pdf'
    elif dataset == 'ds10kall_v2':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kall_v2_fix_alpha_noreg_all')
        file_name = f'ds10kall_v2_noreg_abstain_coverage_computeq_compare_varyratio.pdf'
    elif dataset == 'ds10kbase_deepseek':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kbase_deepseek_fix_alpha_noreg_all')
        file_name = f'ds10kbase_deepseek_noreg_abstain_coverage_computeq_compare_varyratio.pdf'
    elif dataset == 'ds10kall_deepseek':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kall_deepseek_fix_alpha_noreg_all')
        file_name = f'ds10kall_deepseek_noreg_abstain_coverage_computeq_compare_varyratio.pdf'
    elif dataset == 'ds10kbase_deepseek_iodesc':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kbase_deepseek_fix_alpha_noreg_all_iodesc')
        file_name = f'ds10kbase_deepseek_iodesc_noreg_abstain_coverage_computeq_compare_varyratio.pdf'
    elif dataset == 'ds10kall_deepseek_iodesc':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kall_deepseek_fix_alpha_noreg_all_iodesc')
        file_name = f'ds10kall_deepseek_iodesc_noreg_abstain_coverage_computeq_compare_varyratio.pdf'
    else:
        raise NotImplementedError()
    methods = [
        'fixed_group_computeq_2_9_1',
        'fixed_group_computeq_2_8_2',
        'fixed_group_computeq_2_7_3',
        'fixed_group_computeq_2_6_4',
        'fixed_group_computeq_2_5_5',
    ]
    all_data = []
    for method in methods:
        all_pdf_arr = read_results(target_folder, method)
        pd_data = all_pdf_arr[all_pdf_arr['name'] == 'gp_coverages']
        pd_data['value'] = pd_data['value'].apply(eval)
        pd_data['alpha'] = pd_data['alpha'].astype(float) 
        pd_data['coverage'] = pd_data['value'].apply(lambda x: x[0])
        pd_data['ratio'] = '{:.1f}'.format(int(method[-1]) * 0.1)
        all_data.append(pd_data)
    
    all_data = pd.concat(all_data)

    fig, ax = plt.subplots()
    sns.boxplot(all_data, whis=[0, 100], x='ratio', y='coverage')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel('Coverage')   

    # for all x, set y = 0.8
    sns.lineplot(all_data, x='ratio', y=0.8, ax=ax, color='orange', marker='o', linewidth=3, markersize=10)
    # store to file
    output_folder = Path('./data_out/paper_figures')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / file_name, bbox_inches='tight')

def produce_groupcp_computeq_coverage_all():
    apply_base_settings()
    base_target_folder = Path('./data_out/paper_base/exp_result')
    datasets = [
        ('STD', 'std_fix_alpha_noreg_all'),
        ('TDE', 'tde2_fix_alpha_noreg_all'),
        ('DS1K', 'ds10kall_deepseek_fix_alpha_noreg_all_iodesc'),
    ]
    output_name = 'all_abstain_coverage_computeq_compare_varyratio.pdf'

    def load_data(dataset, target_folder):
        methods = [
            'fixed_group_computeq_2_9_1',
            'fixed_group_computeq_2_8_2',
            'fixed_group_computeq_2_7_3',
            'fixed_group_computeq_2_6_4',
            'fixed_group_computeq_2_5_5',
        ]
        all_data = []
        for method in methods:
            all_pdf_arr = read_results(target_folder, method)
            pd_data = all_pdf_arr[all_pdf_arr['name'] == 'gp_coverages']
            pd_data['value'] = pd_data['value'].apply(eval)
            pd_data['alpha'] = pd_data['alpha'].astype(float) 
            pd_data['coverage'] = pd_data['value'].apply(lambda x: x[0])
            pd_data['ratio'] = '{:.1f}'.format(int(method[-1]) * 0.1)
            all_data.append(pd_data)
        
        all_data = pd.concat(all_data)
        all_data['dataset'] = dataset
        return all_data 
    
    all_data = []
    for dataset, target_folder in datasets:
        all_data.append(load_data(dataset, base_target_folder / target_folder))
    all_data = pd.concat(all_data)
    
    fig, ax = plt.subplots()
    sns.boxplot(all_data, whis=[0, 100], x='ratio', y='coverage', hue='dataset')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel('Coverage')   

    # for all x, set y = 0.8
    sns.lineplot(all_data, x='ratio', y=0.8, ax=ax, color='orange', marker='o', linewidth=3, markersize=10)
    # store to file
    output_folder = Path('./data_out/paper_figures/all')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / output_name, bbox_inches='tight')


def produce_groupcp_computeq_group_size(dataset):
    apply_base_settings()
    if dataset == 'tbe':
        target_folder = Path('./data_out/paper_base/exp_result/tde2_fix_alpha_noreg_all')
        file_name = f'tbe_noreg_abstain_size_computeq_compare_varyratio.pdf'
        total_num = 225
    elif dataset == 'std':
        target_folder = Path('./data_out/paper_base/exp_result/std_fix_alpha_noreg_all')
        file_name = f'std_noreg_abstain_size_computeq_compare_varyratio.pdf'
        total_num = 60
    elif dataset == 'ds1k':
        target_folder = Path('./data_out/paper_base/exp_result/ds1k_fix_alpha_noreg_all')
        file_name = f'ds1k_noreg_abstain_size_computeq_compare_varyratio.pdf'
        total_num = 1000
    elif dataset == 'ds10kbase_deepseek_iodesc':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kbase_deepseek_fix_alpha_noreg_all_iodesc')
        file_name = f'ds10kbase_deepseek_iodesc_noreg_abstain_size_computeq_compare_varyratio.pdf'
        total_num = 377
    elif dataset == 'ds10kall_deepseek_iodesc':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kall_deepseek_fix_alpha_noreg_all_iodesc')
        file_name = f'ds10kall_deepseek_iodesc_noreg_abstain_size_computeq_compare_varyratio.pdf'
        total_num = 771
    else:
        raise NotImplementedError() 
    methods = [
        'fixed_group_computeq_2_9_1',
        'fixed_group_computeq_2_8_2',
        'fixed_group_computeq_2_7_3',
        'fixed_group_computeq_2_6_4',
        'fixed_group_computeq_2_5_5',
    ]
    all_data = []
    for method in methods:
        all_pdf_arr = read_results(target_folder, method)
        pd_data = all_pdf_arr[all_pdf_arr['name'] == 'gp_avg_sizes']
        pd_data['value'] = pd_data['value'].apply(eval)
        pd_data['alpha'] = pd_data['alpha'].astype(float) 
        # percent
        pd_data['ratio'] = '{:.1f}'.format(int(method[-1]) * 0.1)
        pd_data['G1'] = pd_data['value'].apply(lambda x: x[0] / total_num * 100)
        pd_data['G2'] = pd_data['value'].apply(lambda x: x[1] / total_num * 100)
        
        all_data.append(pd_data)
    
    all_data = pd.concat(all_data)

    # flatten the data
    data = []
    for _id, row in all_data.iterrows():
        for i in range(1, 3):
            if i == 1:
                g = 'B'
            elif i == 2:
                g = 'A'
            data.append({'ratio': row['ratio'], 'size': row[f'G{i}'], 'Group': f'{g}'})
    all_data = pd.DataFrame(data)

    fig, ax = plt.subplots()
    sns.boxplot(all_data, whis=[0, 100], x='ratio', y='size', hue='Group')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel('Avg Retrieval Percentage (%)')   
    # store to file
    output_folder = Path('./data_out/paper_figures')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / file_name, bbox_inches='tight')


def produce_groupcp_computeq_group_coverage(dataset):
    apply_base_settings()
    if dataset == 'tbe':
        target_folder = Path('./data_out/paper_base/exp_result/tde2_fix_alpha_noreg_all')
        file_name = f'tbe_noreg_abstain_coverage_computeq_compare_varyratio.pdf'
    elif dataset == 'std':
        target_folder = Path('./data_out/paper_base/exp_result/std_fix_alpha_noreg_all')
        file_name = f'std_noreg_abstain_coverage_computeq_compare_varyratio.pdf'
    else:
        raise NotImplementedError() 
    methods = [
        'fixed_group_computeq_2_9_1',
        'fixed_group_computeq_2_8_2',
        'fixed_group_computeq_2_7_3',
        'fixed_group_computeq_2_6_4',
        'fixed_group_computeq_2_5_5',
    ]
    all_data = []
    for method in methods:
        all_pdf_arr = read_results(target_folder, method)
        pd_data = all_pdf_arr[all_pdf_arr['name'] == 'gp_coverages']
        pd_data['value'] = pd_data['value'].apply(eval)
        pd_data['alpha'] = pd_data['alpha'].astype(float) 
        pd_data['coverage'] = pd_data['value'].apply(lambda x: x[0])
        pd_data['ratio'] = '{:.1f}'.format(int(method[-1]) * 0.1)
        all_data.append(pd_data)
    
    all_data = pd.concat(all_data)

    # flatten the data
    data = []
    for _id, row in all_data.iterrows():
        data.append({'ratio': row['ratio'], 'coverage': row['coverage']})
    all_data = pd.DataFrame(data)

    fig, ax = plt.subplots()
    sns.boxplot(all_data, whis=[0, 100], x='ratio', y='coverage')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel('Coverage')   
    # store to file
    output_folder = Path('./data_out/paper_figures')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / file_name, bbox_inches='tight')


def produce_groupcp_group_coverage_varying_gnum(dataset, method_name):
    apply_base_settings()
    if dataset == 'tbe':
        target_folder = Path('./data_out/paper_base/exp_result/tde2_fix_alpha_noreg_all')
        file_name = f'tbe_noreg_abstain_coverage_{method_name}_compare_varygnum.pdf'
    elif dataset == 'std':
        target_folder = Path('./data_out/paper_base/exp_result/std_fix_alpha_noreg_all')
        file_name = f'std_noreg_abstain_coverage_{method_name}_compare_varygnum.pdf'
    elif dataset == 'ds1k':
        target_folder = Path('./data_out/paper_base/exp_result/ds1k_fix_alpha_noreg_all')
        file_name = f'ds1k_noreg_abstain_coverage_{method_name}_compare_varygnum.pdf'
    elif dataset == 'ds10kbase_deepseek_iodesc':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kbase_deepseek_fix_alpha_noreg_all_iodesc')
        file_name = f'ds10kbase_deepseek_iodesc_noreg_abstain_coverage_{method_name}_compare_varygnum.pdf'
    elif dataset == 'ds10kall_deepseek_iodesc':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kall_deepseek_fix_alpha_noreg_all_iodesc')
        file_name = f'ds10kall_deepseek_iodesc_noreg_abstain_coverage_{method_name}_compare_varygnum.pdf'
    else:
        raise NotImplementedError()
    if method_name == 'computeq':
        methods = [
            ('fixed_group_computeq_2_a2', 2),
            ('fixed_group_computeq_3_a2', 3),
            ('fixed_group_computeq_4_a2', 4),
            # ('fixed_group_computeq_5_a2', 5),
        ]
    elif method_name == 'learnq':
        methods = [
            ('fixed_group_learnq_2_a2', 2),
            ('fixed_group_learnq_3_a2', 3),
            ('fixed_group_learnq_4_a2', 4),
            # ('fixed_group_learnq_5_a2', 5),
        ]
    else:
        raise NotImplementedError()
    all_data = []
    for method,code in methods:
        all_pdf_arr = read_results(target_folder, method)

        # all data item are written as (name, value, method, seed)
        # we need to transform it to (method, seed, 'gp_num', 'gp_coverages'), 
        # where 'gp_num' and 'gp_coverages' are the value of name, the value should from the corresponding value cell.
        # the value should be a list of two numbers.
        coverage_data = all_pdf_arr[all_pdf_arr['name'] == 'gp_coverages']
        gpnum_data = all_pdf_arr[all_pdf_arr['name'] == 'gp_num']

        cache_dict = defaultdict(dict)
        for _id, row in coverage_data.iterrows():
            cache_dict[(row['method'], row['seed'])]['gp_coverages'] = eval(row['value'])
        for _id, row in gpnum_data.iterrows():
            cache_dict[(row['method'], row['seed'])]['gp_num'] = eval(row['value'])

        for k, v in cache_dict.items():
            method, seed = k
            coverage = v['gp_coverages']
            gp_num = v['gp_num']
            # calculate the mean coverage
            total_coverage = 0
            for _s, _c in zip(gp_num[:-1], coverage[:-1]):
                total_coverage += _s * _c
            avg_coverage = total_coverage / sum(gp_num[:-1])

            all_data.append({'method': code, 'seed': seed, 
                             'coverage': avg_coverage}) 
            
    all_data = pd.DataFrame(all_data)
    
    fig, ax = plt.subplots()
    sns.boxplot(all_data, whis=[0, 100], x='method', y='coverage')
    ax.set_xlabel('# of Groups')
    ax.set_ylabel('Coverage')   
    # store to file
    output_folder = Path('./data_out/paper_figures')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / file_name, bbox_inches='tight')

def produce_groupcp_group_size_varying_gnum_all():
    apply_base_settings()
    base_target_folder = Path('./data_out/paper_base/exp_result')
    datasets = [
        ('STD', 'std_fix_alpha_noreg_all'),
        ('TDE', 'tde2_fix_alpha_noreg_all'),
        ('DS1K', 'ds10kall_deepseek_fix_alpha_noreg_all_iodesc'),
    ]
    output_name = 'all_abstain_size_computeq_compare_varygnum.pdf'
    
    def load_data(dataset, target_folder):
        methods = [
            ('fixed_group_computeq_2_a2', 2),
            ('fixed_group_computeq_3_a2', 3),
            ('fixed_group_computeq_4_a2', 4),
        ]
        all_data = []
        for method,code in methods:
            all_pdf_arr = read_results(target_folder, method)

            # all data item are written as (name, value, method, seed)
            # we need to transform it to (method, seed, 'gp_num', 'gp_coverages'), 
            # where 'gp_num' and 'gp_coverages' are the value of name, the value should from the corresponding value cell.
            # the value should be a list of two numbers.
            coverage_data = all_pdf_arr[all_pdf_arr['name'] == 'gp_coverages']
            gpnum_data = all_pdf_arr[all_pdf_arr['name'] == 'gp_num']

            cache_dict = defaultdict(dict)
            for _id, row in coverage_data.iterrows():
                cache_dict[(row['method'], row['seed'])]['gp_coverages'] = eval(row['value'])
            for _id, row in gpnum_data.iterrows():
                cache_dict[(row['method'], row['seed'])]['gp_num'] = eval(row['value'])

            for k, v in cache_dict.items():
                method, seed = k
                coverage = v['gp_coverages']
                gp_num = v['gp_num']
                # calculate the mean coverage
                total_coverage = 0
                for _s, _c in zip(gp_num[:-1], coverage[:-1]):
                    total_coverage += _s * _c
                avg_coverage = total_coverage / sum(gp_num[:-1])

                all_data.append({'method': code, 'seed': seed, 
                                'coverage': avg_coverage}) 
                
        all_data = pd.DataFrame(all_data)
        all_data['dataset'] = dataset
        return all_data

    all_data = []
    for dataset, target_folder in datasets:
        all_data.append(load_data(dataset, base_target_folder / target_folder))
    all_data = pd.concat(all_data)
    fig, ax = plt.subplots()
    sns.boxplot(all_data, whis=[0, 100], x='method', y='coverage', hue='dataset')
    # hide legend title
    ax.legend(title='')
    ax.set_xlabel('# of Groups')
    ax.set_ylabel('Coverage')   
    # store to file
    output_folder = Path('./data_out/paper_figures/all')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / output_name, bbox_inches='tight')

def produce_groupcp_group_size_varying_gnum(dataset, method_name):
    apply_base_settings()
    if dataset == 'tbe':
        target_folder = Path('./data_out/paper_base/exp_result/tde2_fix_alpha_noreg_all')
        file_name = f'tbe_noreg_abstain_size_{method_name}_compare_varygnum.pdf'
        total_num = 225
    elif dataset == 'std':
        target_folder = Path('./data_out/paper_base/exp_result/std_fix_alpha_noreg_all')
        file_name = f'std_noreg_abstain_size_{method_name}_compare_varygnum.pdf'
        total_num = 60
    else:
        raise NotImplementedError()
    if method_name == 'computeq':
        methods = [
            ('fixed_group_computeq_2_a2', 2),
            ('fixed_group_computeq_3_a2', 3),
            ('fixed_group_computeq_4_a2', 4),
            # ('fixed_group_computeq_5_a2', 5),
        ]
    elif method_name == 'learnq':
        methods = [
            ('fixed_group_learnq_2_a2', 2),
            ('fixed_group_learnq_3_a2', 3),
            ('fixed_group_learnq_4_a2', 4),
            # ('fixed_group_learnq_5_a2', 5),
        ]
    else:
        raise NotImplementedError()
    all_data = []
    for method,code in methods:
        all_pdf_arr = read_results(target_folder, method)

        # all data item are written as (name, value, method, seed)
        # we need to transform it to (method, seed, 'gp_num', 'gp_avg_sizes'), 
        # where 'gp_num' and 'gp_avg_sizes' are the value of name, the value should from the corresponding value cell.
        # the value should be a list of two numbers.
        avgsize_data = all_pdf_arr[all_pdf_arr['name'] == 'gp_avg_sizes']
        gpnum_data = all_pdf_arr[all_pdf_arr['name'] == 'gp_num']

        cache_dict = defaultdict(dict)
        for _id, row in avgsize_data.iterrows():
            cache_dict[(row['method'], row['seed'])]['gp_avg_sizes'] = eval(row['value'])
        for _id, row in gpnum_data.iterrows():
            cache_dict[(row['method'], row['seed'])]['gp_num'] = eval(row['value'])

        for k, v in cache_dict.items():
            method, seed = k
            avg_size = v['gp_avg_sizes']
            gp_num = v['gp_num']
            # calculate the mean coverage
            total_coverage = 0
            for _s, _c in zip(gp_num[:-1], avg_size[:-1]):
                total_coverage += _s * _c
            avg_size = total_coverage / sum(gp_num[:-1])

            all_data.append({'method': code, 'seed': seed, 
                             'size': avg_size / total_num * 100}) 
            
    all_data = pd.DataFrame(all_data)
    
    fig, ax = plt.subplots()
    sns.boxplot(all_data, whis=[0, 100], x='method', y='size')
    ax.set_xlabel('# of Groups')
    ax.set_ylabel('Average Retrieval Size (%)')   
    # store to file
    output_folder = Path('./data_out/paper_figures')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / file_name, bbox_inches='tight')

def produce_ne_compare_coverage_general(target_folder, methods, file_name):
    apply_base_settings()
    all_data = []
    for (method, name) in methods:
        all_pdf_arr = read_results(target_folder, method)
        if 'base' in method:
            pd_data = all_pdf_arr[all_pdf_arr['name'] == 'accuracy']
            pd_data['value'] = pd_data['value'].astype(float)
            pd_data['coverage'] = pd_data['value']
        else:
            pd_data = all_pdf_arr[all_pdf_arr['name'] == 'gp_coverages']
            pd_data['value'] = pd_data['value'].apply(eval)
            pd_data['coverage'] = pd_data['value'].apply(lambda x: x[0])
        pd_data['alpha'] = pd_data['alpha'].astype(float) 
        pd_data['ealpha'] = 1 - pd_data['alpha']
        pd_data['method'] = name
        all_data.append(pd_data)
    all_data = pd.concat(all_data).reset_index(drop=True)
    
    all_data['alpha'] = all_data['alpha'].astype(str)
    # order alpha    
    alpha_order = [f'{0.5 - 0.1 * i:.1f}' for i in range(5)]
    alpha_order = [*alpha_order, '0.05']

    fig, ax = plt.subplots()
    # print(all_data)
    # exit()
    sns.boxplot(all_data, whis=[0, 100], x='alpha', y='coverage', hue='method', order=alpha_order)
    sns.lineplot(all_data, x='alpha', y='ealpha', marker='*', markersize=10, linewidth=3, color='orange')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Coverage')   
    # store to file
    output_folder = Path('./data_out/paper_figures')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / file_name, bbox_inches='tight')


def produce_ne_compare_coverage_base(dataset):
    if dataset == 'tbe':
        target_folder = Path('./data_out/paper_base/exp_result/tde2_base_noreg_all')
        file_name = f'tbe_noreg_ne_coverage_compare_base.pdf'
    elif dataset == 'std':
        target_folder = Path('./data_out/paper_base/exp_result/std_base_noreg_all')
        file_name = f'std_noreg_ne_coverage_compare_base.pdf'
    elif dataset == 'ds1k':
        target_folder = Path('./data_out/paper_base/exp_result/ds1k_base_noreg_all')
        file_name = f'ds1k_noreg_ne_coverage_compare_base.pdf'
    else:
        raise NotImplementedError()
    methods = [
        ('base', 'FRCP'),
        ('base_weight', 'FRCP+W')
    ]
    produce_ne_compare_coverage_general(target_folder, methods, file_name)

def produce_ne_compare_coverage_abstain(dataset):
    if dataset == 'tbe':
        target_folder = Path('./data_out/paper_base/exp_result/tde2_base_noreg_all')
        file_name = f'tbe_noreg_ne_coverage_compare_abstain.pdf'
    elif dataset == 'std':
        target_folder = Path('./data_out/paper_base/exp_result/std_base_noreg_all')
        file_name = f'std_noreg_ne_coverage_compare_abstain.pdf'
    elif dataset == 'ds1k':
        target_folder = Path('./data_out/paper_base/exp_result/ds1k_base_noreg_all')
        file_name = f'ds1k_noreg_ne_coverage_compare_abstain.pdf'
    else:
        raise NotImplementedError()
    methods = [
        ('fixed_group_computeq_2_8_2', 'FRA+CPA'),
        ('fixed_group_computeq_weight_2_8_2', 'FRA+CPA+W')
    ]
    produce_ne_compare_coverage_general(target_folder, methods, file_name)


def produce_ne_compare_coverage_sc(dataset):
    if dataset == 'tbe':
        target_folder = Path('./data_out/paper_base/exp_result/tde2_base_noreg_all')
        file_name = f'tbe_noreg_ne_coverage_compare_sc.pdf'
    elif dataset == 'std':
        target_folder = Path('./data_out/paper_base/exp_result/std_base_noreg_all')
        file_name = f'std_noreg_ne_coverage_compare_sc.pdf'
    elif dataset == 'ds1k':
        target_folder = Path('./data_out/paper_base/exp_result/ds1k_base_noreg_all')
        file_name = f'ds1k_noreg_ne_coverage_compare_sc.pdf'
    else:
        raise NotImplementedError()
    methods = [
        ('size_group_kp_4', 'CFRA+CPA'),
        ('size_group_weight_kp_4', 'CFRA+CPA+W')
    ]
    produce_ne_compare_coverage_general(target_folder, methods, file_name)

def produce_size_constraint_varying_alpha_size(dataset, figure_type, method_type):
    apply_base_settings()
    if dataset == 'tbe':
        target_folder = Path('./data_out/paper_base/exp_result/tde2_base_noreg_all')
        file_name = f'tbe_noreg_abstain_size_constraint_varyalpha_{figure_type}_{method_type}.pdf'
        total_num = 225
    elif dataset == 'std':
        target_folder = Path('./data_out/paper_base/exp_result/std_base_noreg_all')
        file_name = f'std_noreg_abstain_size_constraint_varyalpha_{figure_type}_{method_type}.pdf'
        total_num = 60
    elif dataset == 'ds1k':
        target_folder = Path('./data_out/paper_base/exp_result/ds1k_base_noreg_all')
        file_name = f'ds1k_noreg_abstain_size_constraint_varyalpha_{figure_type}_{method_type}.pdf'
        total_num = 1000
    elif dataset == 'ds10kall_deepseek_iodesc':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kall_deepseek_base_noreg_all_iodesc')
        file_name = f'ds10kall_deepseek_iodesc_noreg_abstain_size_constraint_varyalpha_{figure_type}_{method_type}.pdf'
        total_num = 771
    else:
        raise NotImplementedError()
    if method_type == 'weight':
        methods = [
            'size_group_weight_kp_4'
        ]
    elif method_type == 'base':
        methods = [
            'size_group_kp_4'
        ]

        if dataset == 'ds10kall_deepseek_iodesc':
            methods = [
                'size_group_kp_1'
            ]
    elif method_type == 'weight_t2':
        methods = [
            'size_group_weight_kp_4_T2'
        ]
    else:
        raise NotImplementedError()
    all_data = []
    for method in methods:
        all_pdf_arr = read_results(target_folder, method)
        avg_size = all_pdf_arr[all_pdf_arr['name'] == 'gp_avg_sizes']
        gp_num = all_pdf_arr[all_pdf_arr['name'] == 'gp_num']
        coverage_data = all_pdf_arr[all_pdf_arr['name'] == 'gp_coverages']

        cache_dict = defaultdict(dict)
        for _id, row in avg_size.iterrows():
            cache_dict[(row['method'], row['seed'], row['alpha'])]['gp_avg_sizes'] = eval(row['value'])
        for _id, row in gp_num.iterrows():
            cache_dict[(row['method'], row['seed'], row['alpha'])]['gp_num'] = eval(row['value'])
        for _id, row in coverage_data.iterrows():
            cache_dict[(row['method'], row['seed'], row['alpha'])]['coverage'] = eval(row['value'])[0]

        for k, v in cache_dict.items():
            method, seed, alpha = k
            gp_sizes = v['gp_avg_sizes']
            gp_num = v['gp_num']
            coverage = v['coverage']
            # compute abstain ratio
            beta = gp_num[1] / sum(gp_num)
            all_data.append({'method': method, 'seed': seed, 
                             'size': gp_sizes[0] / total_num * 100, 'beta': beta, 'alpha': alpha, 'ealpha': 1-alpha, 'coverage': coverage})
    
    all_data = pd.DataFrame(all_data)
    all_data['alpha'] = all_data['alpha'].astype(str)

    fig, ax = plt.subplots()
    if figure_type == 'size':
        sns.boxplot(all_data, whis=[0, 100], x='alpha', y='size', ax=ax)
        # alphas = all_data['alpha'].unique()
        # sns.lineplot(x=alphas, y=[2] * len(alphas), ax=ax, color='orange', marker='o', markersize=10)
        # add legend for two line plots

        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel('Avg Retrieval Percentage (%)')
        ax2 =ax.twinx()
        sns.lineplot(all_data, x='alpha', y='beta', ax=ax2, marker='*', markersize=10, color='purple', linewidth=3)
        ax2.set_ylabel(r'$\beta$')
    elif figure_type == 'coverage':
        sns.boxplot(all_data, whis=[0, 100], x='alpha', y='coverage', ax=ax)
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel('Coverage')
        sns.lineplot(all_data, x='alpha', y='ealpha', ax=ax, marker='*', markersize=10, color='orange', linewidth=3)


    # #  add legend for two line plots
    # lines = ax.get_lines() + ax2.get_lines()
    # labels = ['Coverage', r'$\beta$']
    # ax.legend(lines, labels, loc='upper center')

    # store to file
    output_folder = Path('./data_out/paper_figures')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / file_name, bbox_inches='tight')

def prodce_size_constraint_varying_alpha_coverage_all():
    apply_base_settings()
    # set the ratio of the plot
    plt.rcParams['figure.figsize'] = [10, 5]
    target_base_folder = Path('./data_out/paper_base/exp_result/')
    datasets = [
        ('STD', 'std_base_noreg_all', 60, 'size_group_kp_4'),
        ('TDE', 'tde2_base_noreg_all', 225, 'size_group_kp_4'),    
        ('DS1K', 'ds10kall_deepseek_base_noreg_all_iodesc', 771, 'size_group_kp_1'),
    ]
    output_name = 'all_noreg_abstain_size_constraint_varyalpha_coverage.pdf'
    def load_data(dataset, folder_name, total_num, method):
        all_data = []
        all_pdf_arr = read_results(target_base_folder / folder_name, method)
        avg_size = all_pdf_arr[all_pdf_arr['name'] == 'gp_avg_sizes']
        gp_num = all_pdf_arr[all_pdf_arr['name'] == 'gp_num']
        coverage_data = all_pdf_arr[all_pdf_arr['name'] == 'gp_coverages']

        cache_dict = defaultdict(dict)
        for _id, row in avg_size.iterrows():
            cache_dict[(row['method'], row['seed'], row['alpha'])]['gp_avg_sizes'] = eval(row['value'])
        for _id, row in gp_num.iterrows():
            cache_dict[(row['method'], row['seed'], row['alpha'])]['gp_num'] = eval(row['value'])
        for _id, row in coverage_data.iterrows():
            cache_dict[(row['method'], row['seed'], row['alpha'])]['coverage'] = eval(row['value'])[0]

        for k, v in cache_dict.items():
            method, seed, alpha = k
            gp_sizes = v['gp_avg_sizes']
            gp_num = v['gp_num']
            coverage = v['coverage']
            # compute abstain ratio
            beta = gp_num[1] / sum(gp_num)
            all_data.append({'method': method, 'seed': seed, 
                            'size': gp_sizes[0] / total_num * 100, 'beta': beta, 'alpha': alpha, 'ealpha': 1-alpha, 'coverage': coverage})
        
        all_data = pd.DataFrame(all_data)
        all_data['alpha'] = all_data['alpha'].astype(str)
        all_data['dataset'] = dataset
        return all_data
    
    all_data_arr = []
    for dataset, folder_name, total_num, method in datasets:
        all_data = load_data(dataset, folder_name, total_num, method)
        all_data_arr.append(all_data)
    
    all_data_arr = pd.concat(all_data_arr)
    all_data_arr['dataset'] = all_data_arr['dataset'].astype(str)

    fig, ax = plt.subplots()
    sns.boxplot(all_data_arr, whis=[0, 100], x='alpha', y='coverage', hue='dataset', ax=ax)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Coverage')
    # store to file
    sns.lineplot(all_data_arr, x='alpha', y='ealpha', ax=ax, marker='*', markersize=10, color='orange', linewidth=3)

    output_folder = Path('./data_out/paper_figures/all')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / output_name, bbox_inches='tight')


def produce_size_constraint_varying_k_size(dataset):
    apply_base_settings()
    if dataset == 'tbe':
        target_folder = Path('./data_out/paper_base/exp_result/tde2_base_noreg_all')
        file_name = f'tbe_noreg_abstain_size_constraint_varyk.pdf'
        total_num = 225
    elif dataset == 'std':
        target_folder = Path('./data_out/paper_base/exp_result/std_base_noreg_all')
        file_name = f'std_noreg_abstain_size_constraint_varyk.pdf'
        total_num = 60
    else:
        raise NotImplementedError()
    # methods = [
    #     ('size_group_k_1', '1'),
    #     ('size_group_k_2', '2'),
    #     ('size_group_k_3', '3'),
    #     ('size_group_k_4', '4'),
    # ]
    methods = [
        ('size_group_kp_2', '2'),
        ('size_group_kp_4', '4'),
        ('size_group_kp_6', '6'),
        ('size_group_kp_8', '8'),
    ]
    all_data = []
    for method, mvalue in methods:
        all_pdf_arr = read_results(target_folder, method)
        avg_size = all_pdf_arr[all_pdf_arr['name'] == 'gp_avg_sizes']
        gp_num = all_pdf_arr[all_pdf_arr['name'] == 'gp_num']

        cache_dict = defaultdict(dict)
        for _id, row in avg_size.iterrows():
            cache_dict[(row['method'], row['seed'])]['gp_avg_sizes'] = eval(row['value'])
        for _id, row in gp_num.iterrows():
            cache_dict[(row['method'], row['seed'])]['gp_num'] = eval(row['value'])

        for k, v in cache_dict.items():
            method, seed = k
            gp_sizes = v['gp_avg_sizes']
            gp_num = v['gp_num']
            # compute abstain ratio
            beta = gp_num[1] / sum(gp_num)
            all_data.append({'method': mvalue, 'seed': seed, 
                             'size': gp_sizes[0] / total_num * 100, 'beta': beta})
    
    all_data = pd.DataFrame(all_data)

    fig, ax = plt.subplots()
    sns.boxplot(all_data, whis=[0, 100], x='method', y='size', ax=ax)
    ax2 = ax.twinx()
    sns.lineplot(all_data, x='method', y='beta', ax=ax2, marker='*', markersize=10, color='purple', linewidth=3)

    # ax.set_xlabel(r'$\kappa$')
    ax.set_xlabel(r'$\kappa$ (%)')
    ax.set_ylabel('Avg Retrieval Percentage (%)')
    ax2.set_ylabel(r'$\beta$')
    # store to file
    output_folder = Path('./data_out/paper_figures')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / file_name, bbox_inches='tight')

# compute how many samples need to be generated, and the cost of exanming them.

# def produce_sample_cost_figure(dataset, figure_type):
#     assert figure_type in ['execution', 'generation']
#     apply_base_settings()
#     if dataset == 'tbe':
#         target_folder = Path('./data_out/paper_base/exp_result/tde2_base_noreg_all_alpha_36')
#         file_name = f'tbe_compare_{figure_type}_cost.pdf'
#         total_num = 1125
#     elif dataset == 'std':
#         target_folder = Path('./data_out/paper_base/exp_result/std_base_noreg_all_alpha_36')
#         file_name = f'std_compare_{figure_type}_cost.pdf'
#         total_num = 600
#     else:
#         raise NotImplementedError()

#     # based_method 
#     all_data_arr = []

#     based_method = 'base'
#     pd_arr = read_results(target_folder, based_method)
#     pd_arr = pd_arr[pd_arr['alpha'] == 0.36]

#     pd_size_arr = pd_arr[pd_arr['name'] == 'avg_size']
#     pd_size_arr['value'] = pd_size_arr['value'].astype(float)
#     pd_coverage_arr = pd_arr[pd_arr['name'] == 'accuracy']
#     pd_coverage_arr['value'] = pd_coverage_arr['value'].astype(float)

#     cache_dict = defaultdict(dict)
#     for _id, _row in pd_size_arr.iterrows():
#         cache_dict[(_row['method'], _row['seed'])]['check_num'] = _row['value'] * total_num
#     for _id, _row in pd_coverage_arr.iterrows():
#         cache_dict[(_row['method'], _row['seed'])]['gen_num'] = (1-_row['value']) * total_num
    
#     for k, v in cache_dict.items():
#         method, seed = k
#         all_data_arr.append({'method': method, 'seed': seed, **v})

#     if dataset == 'tbe':
#         target_folder = Path('./data_out/paper_base/exp_result/tde2_base_noreg_all')
#     elif dataset == 'std':
#         target_folder = Path('./data_out/paper_base/exp_result/std_base_noreg_all')
#     else:
#         raise NotImplementedError()

#     def _compute_group_methods(_gname, _all_arr):
#         pd_arr = read_results(target_folder, _gname)
#         pd_arr = pd_arr[pd_arr['alpha'] == 0.2]
        
#         pd_size_arr = pd_arr[pd_arr['name'] == 'gp_avg_sizes']
#         pd_size_arr['value'] = pd_size_arr['value'].apply(eval)
#         pd_coverage_arr = pd_arr[pd_arr['name'] == 'gp_coverages']
#         pd_coverage_arr['value'] = pd_coverage_arr['value'].apply(eval)
#         pd_gnum_arr = pd_arr[pd_arr['name'] == 'gp_num']
#         pd_gnum_arr['value'] = pd_gnum_arr['value'].apply(eval)

#         cache_dict = defaultdict(dict)
#         for _id, _row in pd_size_arr.iterrows():
#             cache_dict[(_row['method'], _row['seed'])]['avg_size'] = _row['value'][0]
#         for _id, _row in pd_coverage_arr.iterrows():
#             cache_dict[((_row['method'], _row['seed']))]['miscoverage'] = 1-_row['value'][0]
#         for _id, _row in pd_gnum_arr.iterrows():
#             cache_dict[((_row['method'], _row['seed']))]['answer_num'] = _row['value'][0]
        
#         for k, v in cache_dict.items():
#             method, seed = k
#             avg_size, miscoverage, ans_num = v['avg_size'], v['miscoverage'], v['answer_num']
#             _all_arr.append({
#                 'method': method, 'seed': seed, 
#                 'check_num': avg_size * ans_num, 'gen_num': (total_num - ans_num) + miscoverage * ans_num
#             })
#     #   fixed_group_compute method.

#     group_learn_method = 'fixed_group_learnq_2_8_2'
#     _compute_group_methods(group_learn_method, all_data_arr)

#     group_compute_method = 'fixed_group_computeq_2_8_2'
#     _compute_group_methods(group_compute_method, all_data_arr)

#     fixed_size_method = 'size_group_k_3'
#     _compute_group_methods(fixed_size_method, all_data_arr)

#     pd_arr = pd.DataFrame(all_data_arr)
#     name_mapping = {
#         'base': 'CP',
#         'fixed_group_learnq_2_8_2': 'LA',
#         'fixed_group_computeq_2_8_2': 'CPA',
#         'size_group_k_3': 'SC',
#     }
#     pd_arr['name'] = pd_arr['method'].apply(lambda x: name_mapping[x])

#     if figure_type == 'execution':
#         # ax = sns.boxplot(pd_arr, x='name', y='check_num', whis=[0, 100])
#         ax = sns.barplot(pd_arr, x='name', y='check_num')
        
#         ax.set_ybound(pd_arr['check_num'].min()-100, pd_arr['check_num'].max()+100)
#     elif figure_type == 'generation':
#         ax = sns.barplot(pd_arr, x='name', y='gen_num')
#         ax.set_ybound(pd_arr['gen_num'].min()-10, pd_arr['gen_num'].max()+10)
#     # save.
    
#     ax.set_xlabel('Method')
#     if figure_type == 'execution':
#         ax.set_ylabel('# of Function Execution')
#     elif figure_type == 'generation':
#         ax.set_ylabel('# of Code Generation')
#     # store to file
#     output_folder = Path('./data_out/paper_figures')
#     output_folder.mkdir(parents=True, exist_ok=True)
#     plt.savefig(output_folder / file_name, bbox_inches='tight')

def produce_cost_vary_alpha_base(dataset):
    apply_base_settings()
    if dataset == 'tbe':
        target_folder = Path('./data_out/paper_base/exp_result/tde2_base_noreg_all')
        file_name = f'tbe_compare_cost_varyalpha_base.pdf'
        total_num = 1125
    elif dataset == 'std':
        target_folder = Path('./data_out/paper_base/exp_result/std_base_noreg_all')
        file_name = f'std_compare_cost_varyalpha_base.pdf'
        total_num = 600
    elif dataset == 'ds1k':
        target_folder = Path('./data_out/paper_base/exp_result/ds1k_base_noreg_all')
        file_name = f'ds1k_compare_cost_varyalpha_base.pdf'
        total_num = 1000
    else:
        raise NotImplementedError()

    based_method = 'base'
    pd_arr = read_results(target_folder, based_method)
    
    pd_size_arr = pd_arr[pd_arr['name'] == 'avg_size']
    pd_size_arr['value'] = pd_size_arr['value'].astype(float)
    pd_coverage_arr = pd_arr[pd_arr['name'] == 'accuracy']
    pd_coverage_arr['value'] = pd_coverage_arr['value'].astype(float)

    cache_dict = defaultdict(dict)
    for _id, _row in pd_size_arr.iterrows():
        cache_dict[(_row['method'], _row['seed'], _row['alpha'])]['check_num'] = _row['value'] * total_num
    for _id, _row in pd_coverage_arr.iterrows():
        cache_dict[(_row['method'], _row['seed'], _row['alpha'])]['gen_num'] = (1-_row['value']) * total_num
    
    all_data_arr = []
    for k, v in cache_dict.items():
        method, seed, alpha = k
        all_data_arr.append({'method': method, 'seed': seed, 'alpha': alpha, **v})
    
    pd_arr = pd.DataFrame(all_data_arr)
    pd_arr['alpha'] = pd_arr['alpha'].astype(str)

    # use colors from the palette.
    palette = sns.color_palette()

    fig, ax = plt.subplots()
    line1 = sns.lineplot(pd_arr, x='alpha', y='check_num', ax=ax, marker='o', markersize=10, color=palette[0])
    ax2 = ax.twinx()
    line2 = sns.lineplot(pd_arr, x='alpha', y='gen_num', ax=ax2, marker='*', markersize=10, color=palette[1])
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('# of Function Execution')
    ax2.set_ylabel('# of Code Generation')
    
    y1_min, y1_max = ax.get_ylim()
    y2_min, y2_max = ax2.get_ylim()
    ax.set_ylim(y1_min, y1_max * 1.2)  # Increase upper limit by 20%
    ax2.set_ylim(y2_min, y2_max * 1.2)  # Increase upper limit by 20%

    # Create a single legend for both lines
    lines = line1.get_lines() + line2.get_lines()
    labels = ['# of Function Execution', '# of Code Generation']
    ax.legend(lines, labels, loc='upper center')

    # store to file
    output_folder = Path('./data_out/paper_figures')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / file_name, bbox_inches='tight')


def produce_cost_vary_alpha_group_general_(dataset, method_name, file_name):
    apply_base_settings()
    if dataset == 'tbe':
        target_folder = Path('./data_out/paper_base/exp_result/tde2_base_noreg_all')
        total_num = 1125
    elif dataset == 'std':
        target_folder = Path('./data_out/paper_base/exp_result/std_base_noreg_all')
        total_num = 600
    else:
        raise NotImplementedError()

    # based_method 
    all_data_arr = []

    def _compute_group_methods(_gname, _all_arr):
        pd_arr = read_results(target_folder, _gname)
        
        pd_size_arr = pd_arr[pd_arr['name'] == 'gp_avg_sizes']
        pd_size_arr['value'] = pd_size_arr['value'].apply(eval)
        pd_coverage_arr = pd_arr[pd_arr['name'] == 'gp_coverages']
        pd_coverage_arr['value'] = pd_coverage_arr['value'].apply(eval)
        pd_gnum_arr = pd_arr[pd_arr['name'] == 'gp_num']
        pd_gnum_arr['value'] = pd_gnum_arr['value'].apply(eval)

        cache_dict = defaultdict(dict)
        for _id, _row in pd_size_arr.iterrows():
            cache_dict[(_row['method'], _row['seed'], _row['alpha'])]['avg_size'] = _row['value'][0]
        for _id, _row in pd_coverage_arr.iterrows():
            cache_dict[((_row['method'], _row['seed'], _row['alpha']))]['miscoverage'] = 1-_row['value'][0]
        for _id, _row in pd_gnum_arr.iterrows():
            cache_dict[((_row['method'], _row['seed'], _row['alpha']))]['answer_num'] = _row['value'][0]
        
        for k, v in cache_dict.items():
            method, seed, alpha = k
            avg_size, miscoverage, ans_num = v['avg_size'], v['miscoverage'], v['answer_num']
            _all_arr.append({
                'method': method, 'seed': seed, 'alpha': alpha,
                'check_num': avg_size * ans_num, 'gen_num': (total_num - ans_num) + miscoverage * ans_num
            })
    _compute_group_methods(method_name, all_data_arr)

    pd_arr = pd.DataFrame(all_data_arr)
    pd_arr['alpha'] = pd_arr['alpha'].astype(str)

    # use colors from the palette.
    palette = sns.color_palette()

    fig, ax = plt.subplots()
    line1 = sns.lineplot(pd_arr, x='alpha', y='check_num', ax=ax, marker='o', markersize=10, color=palette[0])
    ax2 = ax.twinx()
    line2 = sns.lineplot(pd_arr, x='alpha', y='gen_num', ax=ax2, marker='*', markersize=10, color=palette[1])
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('# of Function Execution')
    ax2.set_ylabel('# of Code Generation')

    y1_min, y1_max = ax.get_ylim()
    y2_min, y2_max = ax2.get_ylim()
    ax.set_ylim(y1_min, y1_max * 1.2)  # Increase upper limit by 20%
    ax2.set_ylim(y2_min, y2_max * 1.2)  # Increase upper limit by 20%

    # Create a single legend for both lines
    lines = line1.get_lines() + line2.get_lines()
    labels = ['# of Function Execution', '# of Code Generation']
    ax.legend(lines, labels, loc='upper center')

    # Remove the legend from the second axis
    # ax2.get_legend().remove()

    # store to file
    output_folder = Path('./data_out/paper_figures')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / file_name, bbox_inches='tight')

def produce_cost_vary_alpha_learnq(dataset):
    apply_base_settings()
    method_name = 'fixed_group_learnq_2_8_2'
    file_name = f'{dataset}_compare_cost_varyalpha_{method_name}.pdf'
    produce_cost_vary_alpha_group_general_(dataset, method_name, file_name)

def produce_cost_vary_alpha_computeq(dataset):
    apply_base_settings()
    method_name = 'fixed_group_computeq_2_8_2'
    file_name = f'{dataset}_compare_cost_varyalpha_{method_name}.pdf'
    produce_cost_vary_alpha_group_general_(dataset, method_name, file_name)

def produce_cost_vary_alpha_group_size(dataset):
    apply_base_settings()
    method_name = 'size_group_kp_4'
    file_name = f'{dataset}_compare_cost_varyalpha_{method_name}.pdf'
    produce_cost_vary_alpha_group_general_(dataset, method_name, file_name)


def produce_cost_time_vary_alpha_group_general_(dataset, method_name, file_name):
    apply_base_settings()
    if dataset == 'tbe':
        target_folder = Path('./data_out/paper_base/exp_result/tde2_base_noreg_all')
        total_num = 1125
    elif dataset == 'std':
        target_folder = Path('./data_out/paper_base/exp_result/std_base_noreg_all')
        total_num = 600
    elif dataset == 'ds10kall_deepseek_iodesc':
        target_folder = Path('./data_out/paper_base/exp_result/ds10kall_deepseek_base_noreg_all_iodesc')
        total_num = 13129
    else:
        raise NotImplementedError()

    # based_method 
    all_data_arr = []

    code_gen_time = 7.53
    func_exec_time = 0.19

    def _compute_group_methods(_gname, _all_arr):
        pd_arr = read_results(target_folder, _gname)
        
        pd_size_arr = pd_arr[pd_arr['name'] == 'gp_avg_sizes']
        pd_size_arr['value'] = pd_size_arr['value'].apply(eval)
        pd_coverage_arr = pd_arr[pd_arr['name'] == 'gp_coverages']
        pd_coverage_arr['value'] = pd_coverage_arr['value'].apply(eval)
        pd_gnum_arr = pd_arr[pd_arr['name'] == 'gp_num']
        pd_gnum_arr['value'] = pd_gnum_arr['value'].apply(eval)

        cache_dict = defaultdict(dict)
        for _id, _row in pd_size_arr.iterrows():
            cache_dict[(_row['method'], _row['seed'], _row['alpha'])]['avg_size'] = _row['value'][0]
        for _id, _row in pd_coverage_arr.iterrows():
            cache_dict[((_row['method'], _row['seed'], _row['alpha']))]['miscoverage'] = 1-_row['value'][0]
        for _id, _row in pd_gnum_arr.iterrows():
            cache_dict[((_row['method'], _row['seed'], _row['alpha']))]['answer_num'] = _row['value'][0]
        
        for k, v in cache_dict.items():
            method, seed, alpha = k
            avg_size, miscoverage, ans_num = v['avg_size'], v['miscoverage'], v['answer_num']
            
            _all_arr.append({
                'method': method, 'seed': seed, 'alpha': alpha,
                'check_num': avg_size * ans_num * func_exec_time /60 *10, # 10 test cases.
                'gen_num': ((total_num - ans_num) + miscoverage * ans_num) * code_gen_time /60
            })
    _compute_group_methods(method_name, all_data_arr)

    pd_arr = pd.DataFrame(all_data_arr)
    pd_arr['alpha'] = pd_arr['alpha'].astype(str)

    # use colors from the palette.
    palette = sns.color_palette()

    fig, ax = plt.subplots()
    line1 = sns.lineplot(pd_arr, x='alpha', y='check_num', ax=ax, marker='o', markersize=10, color=palette[0])
    # ax2 = ax.twinx()
    line2 = sns.lineplot(pd_arr, x='alpha', y='gen_num', ax=ax, marker='*', markersize=10, color=palette[1])
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Time Cost (m)')
    # ax.set_ylabel('Function Execution Time (m)')
    # ax2.set_ylabel('Code Generation Time (m)')
    
    print(pd_arr)

    y1_min, y1_max = ax.get_ylim()
    # y2_min, y2_max = ax2.get_ylim()
    ax.set_ylim(y1_min, y1_max * 1.2)  # Increase upper limit by 20%
    # ax2.set_ylim(y2_min, y2_max * 1.2)  # Increase upper limit by 20%

    # Create a single legend for both lines
    lines = line1.get_lines() + line2.get_lines()
    labels = ['Function Execution', 'Code Generation']
    ax.legend(lines, labels, loc='upper center')

    # Remove the legend from the second axis
    # ax2.get_legend().remove()

    # store to file
    output_folder = Path('./data_out/paper_figures')
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / file_name, bbox_inches='tight')

def produce_cost_time_vary_alpha_computeq(dataset):
    apply_base_settings()
    # plt.subplots_adjust(left=0.3, right=0.7, top=0.95, bottom=0.3)
    method_name = 'fixed_group_computeq_2_8_2'
    file_name = f'{dataset}_compare_cost_time_varyalpha_{method_name}.pdf'
    produce_cost_time_vary_alpha_group_general_(dataset, method_name, file_name)


if __name__ == '__main__':
    # produce_base_figure_coverage('std')
    # produce_base_figure_coverage('tbe')
    # produce_base_figure_coverage('ds1k')
    # produce_base_figure_coverage('ds10kall_deepseek_iodesc')

    # produce_base_figure_size('std')
    # produce_base_figure_size('tbe')
    # produce_base_figure_size('ds1k')
    # produce_base_figure_size('ds10kall_deepseek_iodesc')

    # produce_base_reg_noreg_size_compare('std')
    # produce_base_reg_noreg_size_compare('tbe')
    # produce_base_reg_noreg_coverage_compare('std')
    # produce_base_reg_noreg_coverage_compare('tbe')
    # produce_base_weight_compare('tbe')
    # produce_base_weight_compare('std')
    # produce_base_group_cp('std', 2)
    # produce_base_group_cp('std', 3)
    # produce_base_group_cp('tbe', 2)
    # produce_base_group_cp('tbe', 3)

    # produce_abstain_rate_computeq('ds1k')
    # produce_abstain_rate_computeq('tbe')
    # produce_abstain_rate_computeq('std')
    # produce_abstain_rate_computeq('ds10kall_deepseek_iodesc')

    # produce_abstain_rate_learnq('tbe')
    # produce_abstain_rate_learnq('ds1k')
    # produce_abstain_rate_learnq('std')
    # produce_abstain_rate_learnq('ds10kall_deepseek_iodesc')


    # produce_base_group_cp_group_size('tbe')
    # produce_base_group_cp_group_size('std')

    # produce_base_fixgroup_group_size('tbe')
    # produce_base_fixgroup_group_size('ds1k')
    # produce_base_fixgroup_group_size('std')
    # produce_base_fixgroup_group_size('ds10kbase_v2')
    # produce_base_fixgroup_group_size('ds10kall_v2')
    # produce_base_fixgroup_group_size('ds10kall_deepseek_iodesc')
    
    # produce_base_fixgroupcp_group_size('tbe')
    # produce_base_fixgroupcp_group_size('std')
    # produce_base_fixgroupcp_group_size('ds1k')
    # produce_base_fixgroupcp_group_size('ds10kall_v2')
    # produce_base_fixgroupcp_group_size('ds10kbase_v2')
    # produce_base_fixgroupcp_group_size('ds10kbase_deepseek')
    # produce_base_fixgroupcp_group_size('ds10kall_deepseek')
    # produce_base_fixgroupcp_group_size('ds10kbase_deepseek_iodesc')
    # produce_base_fixgroupcp_group_size('ds10kall_deepseek_iodesc')
    produce_base_fixgroupcp_group_size('ds10kall_deepseek_iodesc_st')


    # produce_base_group_cp_coverage_base('tbe')
    # produce_base_group_cp_coverage_base('std')
    
    # produce_base_group_cp_coverage_learnq('tbe')
    # produce_base_group_cp_coverage_learnq('std')
    # produce_base_group_cp_coverage_learnq('ds1k')
    # produce_base_group_cp_coverage_learnq('ds10kall_deepseek_iodesc')
    # produce_base_group_cp_coverage_learnq('ds10kall_deepseek_iodesc_st')

    # produce_base_group_cp_coverage_computeq('tbe')
    # produce_base_group_cp_coverage_computeq('std')
    # produce_base_group_cp_coverage_computeq('ds1k')
    # produce_base_group_cp_coverage_computeq('ds10kall_deepseek_iodesc')
    # produce_base_group_cp_coverage_computeq('ds10kall_deepseek_iodesc_st')


    # produce_base_group_cp_coverage_computeq_weight('tbe')
    # produce_base_group_cp_coverage_computeq_weight('std')


    # produce_groupcp_computeq_group_size('std')
    produce_groupcp_computeq_group_size('tbe')
    # produce_groupcp_computeq_group_size('ds1k')

    # produce_groupcp_computeq_group_size('ds10kall_deepseek_iodesc')

    # produce_groupcp_computeq_group_coverage('std')
    # produce_groupcp_computeq_group_coverage('tbe')

    # produce_groupcp_computeq_group_samplenum('std')
    # produce_groupcp_computeq_group_samplenum('tbe')
    # produce_groupcp_computeq_group_samplenum('ds1k')
    # produce_groupcp_computeq_group_samplenum('ds10kall_deepseek_iodesc')

    # produce_groupcp_computeq_coverage('std')
    # produce_groupcp_computeq_coverage('tbe')
    # produce_groupcp_computeq_coverage('ds1k')
    # produce_groupcp_computeq_coverage('ds10kbase')
    # produce_groupcp_computeq_coverage('ds10kall')
    # produce_groupcp_computeq_coverage('ds10kbase_v2')
    # produce_groupcp_computeq_coverage('ds10kall_v2')

    # produce_groupcp_computeq_coverage('ds10kbase_deepseek')
    # produce_groupcp_computeq_coverage('ds10kall_deepseek')
    # produce_groupcp_computeq_coverage('ds10kbase_deepseek_iodesc')
    # produce_groupcp_computeq_coverage('ds10kall_deepseek_iodesc')

    # produce_groupcp_group_coverage_varying_gnum('std', 'computeq')
    # produce_groupcp_group_coverage_varying_gnum('tbe', 'computeq')
    # produce_groupcp_group_coverage_varying_gnum('ds1k', 'computeq')
    # produce_groupcp_group_coverage_varying_gnum('ds10kall_deepseek_iodesc', 'computeq')

    # produce_groupcp_group_coverage_varying_gnum('std', 'learnq')
    # produce_groupcp_group_coverage_varying_gnum('tbe', 'learnq')
    # produce_groupcp_group_coverage_varying_gnum('ds1k', 'learnq')
    # produce_groupcp_group_coverage_varying_gnum('ds10kall_deepseek_iodesc', 'learnq')

    # produce_groupcp_group_size_varying_gnum('std', 'computeq')
    # produce_groupcp_group_size_varying_gnum('tbe', 'computeq')

    # produce_groupcp_group_size_varying_gnum('std', 'learnq')
    # produce_groupcp_group_size_varying_gnum('tbe', 'learnq')


    # produce_size_constraint_varying_alpha_size('std', 'size', 'base')
    # produce_size_constraint_varying_alpha_size('tbe', 'size', 'base')
    # produce_size_constraint_varying_alpha_size('ds1k', 'size', 'base')

    # produce_size_constraint_varying_alpha_size('ds10kall_deepseek_iodesc', 'size', 'base')

    # produce_size_constraint_varying_alpha_size('std', 'size', 'weight')
    # produce_size_constraint_varying_alpha_size('tbe', 'size', 'weight')
    # produce_size_constraint_varying_alpha_size('ds1k', 'size', 'weight')

    # produce_size_constraint_varying_alpha_size('std', 'coverage', 'base')
    # produce_size_constraint_varying_alpha_size('tbe', 'coverage', 'base')
    # produce_size_constraint_varying_alpha_size('ds1k', 'coverage', 'base')
    # produce_size_constraint_varying_alpha_size('ds10kall_deepseek_iodesc', 'coverage', 'base')

    # produce_size_constraint_varying_alpha_size('std', 'coverage', 'weight')
    # produce_size_constraint_varying_alpha_size('tbe', 'coverage', 'weight')
    # produce_size_constraint_varying_alpha_size('ds1k', 'coverage', 'weight')

    # produce_size_constraint_varying_k_size('std')
    produce_size_constraint_varying_k_size('tbe')

    # # produce_sample_cost_figure('std', 'execution')
    # # produce_sample_cost_figure('tbe', 'execution')
    # # produce_sample_cost_figure('std', 'generation')
    # # produce_sample_cost_figure('tbe', 'generation')

    # produce_cost_vary_alpha_learnq('std')
    # produce_cost_vary_alpha_learnq('tbe')
    # produce_cost_vary_alpha_computeq('std')
    # produce_cost_vary_alpha_computeq('tbe')
    # produce_cost_vary_alpha_group_size('std')
    # produce_cost_vary_alpha_group_size('tbe')

    # produce_cost_vary_alpha_base('std')
    # produce_cost_vary_alpha_base('tbe')
    # produce_cost_vary_alpha_base('ds1k')

    # produce_ne_compare_coverage_base('std')
    # produce_ne_compare_coverage_base('tbe')
    # produce_ne_compare_coverage_base('ds1k')

    # produce_ne_compare_coverage_abstain('std')
    # produce_ne_compare_coverage_abstain('tbe')

    # produce_ne_compare_coverage_abstain('ds1k')

    # produce_ne_compare_coverage_sc('std')
    # produce_ne_compare_coverage_sc('tbe')
    # produce_ne_compare_coverage_sc('ds1k')
    
    # produce all datasets.
    # produce_base_figures_coverage_all()
    produce_base_figures_size_all()
    # produce_base_group_cp_coverage_learnq_all()
    # produce_base_group_cp_coverage_computeq_all()
    # produce_groupcp_computeq_coverage_all()
    # produce_groupcp_group_size_varying_gnum_all()
    # prodce_size_constraint_varying_alpha_coverage_all()

    # produce_cost_time_vary_alpha_learnq('ds10kall_deepseek_iodesc')
    # produce_cost_time_vary_alpha_computeq('tbe')
    # produce_cost_vary_alpha_computeq('tbe')
