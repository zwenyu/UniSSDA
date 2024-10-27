'''
Consolidate full results for runs.
'''

import os
import argparse
import itertools
import numpy as np
import pandas as pd


def consolidate_one_expt(args, expt_description, scenarios):
    results_one_expt = {s: [] for s in scenarios}  # list contains result [mean, std] from each method
    results_one_expt['Average'] = []
    for da_method in args.da_method_list:
        run_description = f"expt-{da_method}-{args.sampling}-{args.num_shots}"
        expt_folder = os.path.join(args.expt_logs_dir, expt_description, run_description)

        try:
            report_file_name_avg = f"{args.ckpt}_checkpoint_average_results.xlsx" if not args.transductive else \
                f"{args.ckpt}_checkpoint_train_average_results.xlsx"
            report_save_path_avg = os.path.join(expt_folder, report_file_name_avg)
            report_avg = pd.read_excel(report_save_path_avg)

            report_file_name_std = f"{args.ckpt}_checkpoint_std_results.xlsx" if not args.transductive else \
                f"{args.ckpt}_checkpoint_train_std_results.xlsx"
            report_save_path_std = os.path.join(expt_folder, report_file_name_std)
            report_std = pd.read_excel(report_save_path_std)

            avg_da_method = []
            std_da_method = []
            incomplete_scenarios = [s for s in scenarios]
            for s in scenarios:
                avg_s = round(float(report_avg[report_avg['scenario'] == s][args.metric]), 1)
                std_s = round(float(report_std[report_std['scenario'] == s][args.metric]), 1)
                avg_da_method.append(avg_s)
                std_da_method.append(std_s)
                incomplete_scenarios.remove(s)
                results_one_expt[s].append([avg_s, std_s])
            results_one_expt['Average'].append([round(np.mean(avg_da_method), 1), 0])

        except Exception:
            for s in incomplete_scenarios:
                results_one_expt[s].append([0, 0])  # placeholder for incomplete expt
            results_one_expt['Average'].append([0, 0])

    # format string, bold highest avg in each scenario
    results_str_one_expt = {s: [] for s in results_one_expt.keys()}
    for s in results_one_expt.keys():
        # find max avg in scenario s
        results_s = results_one_expt[s]
        max_s = max([v[0] for v in results_s])
        idx_max_s = [k for k, v in enumerate(results_s) if v[0] == max_s]
        if s == 'Average':
            results_str_s = [str(v[0]) for v in results_s]
        else:
            results_str_s = [' \\rpm '.join(map(str, v)) for v in results_s]
        if args.bold:
            for i in idx_max_s:
                results_str_s[i] = f'\\textbf{{{results_str_s[i]}}}'
        results_str_one_expt[s] = results_str_s

    return results_one_expt, results_str_one_expt


def consolidate_expt(args, scenarios):

    results_str = None
    for expt_description in args.expt_desc_list:
        results_one_expt, results_str_one_expt = consolidate_one_expt(args, expt_description, scenarios)
        if results_str is None:
            results_str = results_str_one_expt
        else:
            results_str = {k: v + results_str_one_expt[k] for k, v in results_str.items()}

    for k, v in results_str.items():
        scenario_name = k.split('_to_')
        scenario_name = ' $\\rightarrow$ '.join([d[0].upper() for d in scenario_name]) if (len(scenario_name) == 2) else scenario_name[0]
        print(scenario_name + ' & ' + ' & '.join(v) + '\\\\')


parser = argparse.ArgumentParser()

parser.add_argument('--expt_logs_dir', default='experiments_logs', type=str)
parser.add_argument('--expt_desc_list', default=['expt_run-txt-Resnet34-office_home-openpartial'], type=str, nargs='+')

parser.add_argument('--dataset', default='office_home', type=str)
parser.add_argument('--domain_pairs', default='all', choices=['all', 'subset'], type=str)

parser.add_argument('--da_method_list', default=['Proposed'], type=str, nargs='+')
parser.add_argument('--sampling', default='kshot', type=str, help='Sampling strategy used to select labeled target data')
parser.add_argument('--num_shots', default='3', type=int, help='Number of labels per class')

parser.add_argument('--ckpt', default="best_acc_trg_val", type=str, choices=['best_acc_src_val', 'best_acc_trg_val', 'last'])
parser.add_argument('--transductive', default=False, action='store_true', help='if False, return metric on samples unseen at training')
parser.add_argument('--metric', default="acc", type=str, help='evaluation metric to return')

parser.add_argument('--bold', default=False, action='store_true', help='if True, bold highest metric value in each column')

args = parser.parse_args()

if __name__ == "__main__":

    if args.dataset == 'office_home':
        if args.domain_pairs == 'all':
            domains = ['Art', 'Clipart', 'Product', 'Real']
    elif args.dataset == 'domain_net':
        if args.domain_pairs == 'all':
            domains = ['clipart', 'painting', 'real', 'sketch']
    scenarios = list(itertools.product(domains, domains))
    same_domain_scenarios = [(d, d) for d in domains]
    scenarios = ['_to_'.join(s) for s in scenarios if s not in same_domain_scenarios]

    consolidate_expt(args, scenarios)