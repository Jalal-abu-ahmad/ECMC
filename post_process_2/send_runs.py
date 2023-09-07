#!/Local/ph_daniel/anaconda3/bin/python -u

import os
import re
import csv

prefix = "/Users/jalal/Desktop/ECMC/ECMC_simulation_results3.0/"
code_prefix = "./"


def params_from_name(name):
    ss = re.split("[_=]", name)
    for i, s in enumerate(ss):
        if s == 'N':
            N = int(ss[i + 1])
        if s == 'h':
            h = float(ss[i + 1])
        if s == 'rhoH':
            rhoH = float(ss[i + 1])
        if s == 'triangle' or s == 'square':
            ic = s
            if ss[i - 1] == 'AF' and s == 'triangle':
                ic = 'honeycomb'
    return N, h, rhoH, ic


def create_op_dir(sim):
    op_dir = os.path.join(prefix, sim, "OP")
    if not os.path.exists(op_dir):
        os.mkdir(op_dir)
    return


def main():
    sims = [d for d in os.listdir(prefix) if d.startswith('N=') and os.path.isdir(os.path.join(prefix, d))]

    for sim in sims:
        create_op_dir(sim)
    f = open(os.path.join(code_prefix, 'post_process_list.txt'), 'wt')
    try:
        writer = csv.writer(f, lineterminator='\n')
        for sim_name in sims:
            N, h, rhoH, ic = params_from_name(sim_name)
            # comment this out to run post_process on all files in simulation results
            if h == 0.8 and 0.7 <= rhoH <= 0.9 and N == 90000 and ic == 'square':
                writer.writerow((sim_name,))

    finally:
        f.close()
        # os.system("condor_submit post_process.sub")


if __name__ == "__main__":
    main()