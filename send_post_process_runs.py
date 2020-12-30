#!/Local/cmp/anaconda3/bin/python -u
import os
import time
# from send_parametric_runs import params_from_name
import re

prefix = "/storage/ph_daniel/danielab/ECMC_simulation_results3.0/"
code_prefix = "/srv01/technion/danielab/OOP_hard_sphere_event_chain/"


def send_specific_run(sim_name, post_types):
    for post_type in post_types:
        out_pwd = prefix + 'out/post_process_' + sim_name + '_' + post_type + '.out'
        err_pwd = prefix + 'out/post_process_' + sim_name + '_' + post_type + '.err'
        time.sleep(2.0)
        os.system("qsub -V -v sim_path=" + sim_name.replace('=', '\=') + ",run_type=" + post_type +
                  " -N " + post_type + "_" + sim_name + " -o " + out_pwd + " -e " + err_pwd +
                  " -l nodes=1:ppn=1,mem=1gb,vmem=2gb -q N " + code_prefix + "post_process_env.sh")


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


def main():
    sims = [d for d in os.listdir(prefix) if d.startswith('N=') and os.path.isdir(os.path.join(prefix, d))]
    for sim_name in sims:
        _, h, _, _ = params_from_name(sim_name)
        default_op = ["gM"]  # , "psi", "Bragg_S", "Bragg_Sm", "pos"]
        if h > 0.85:
            mn = "23"
        if 0.55 <= h <= 0.85:
            mn = "14"
            send_specific_run(sim_name, ["burger_square"])
        if h < 0.55:
            mn = "16"
        send_specific_run(sim_name, [op + mn for op in default_op])


if __name__ == "__main__":
    main()
