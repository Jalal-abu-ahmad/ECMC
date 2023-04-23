# OOP hard sphere event chain

This repository uses event chain monte carlo to simulate anti-ferromagnetic (AF)
latices, mainly in the tetratic phase.
The code is designed to run on a condor cluster, for more information please read
[This](https://htcondor.org/)

## File structure 

### deploy_simulations_on_HTCondor

1. `send_parametric_runs.py` This files sends multiple simulation runs to condor,
    it does this using a temporary txt file called `ecmc_list.txt`, 
    that file houses the input string that is sent to the simulation runner.

2. `ecmc_list.txt` Houses the information that will be sent to THCondor, for example:
    ```commandline
    N=10000_h=0.8_rhoH=0.801_AF_square_ECMC
    N=10000_h=0.8_rhoH=0.801_AF_triangle_ECMC
    N=10000_h=0.8_rhoH=0.802_AF_square_ECMC
    N=10000_h=0.8_rhoH=0.802_AF_triangle_ECMC
    N=10000_h=0.8_rhoH=0.803_AF_square_ECMC
    N=10000_h=0.8_rhoH=0.803_AF_triangle_ECMC
    N=10000_h=0.8_rhoH=0.804_AF_square_ECMC
    N=10000_h=0.8_rhoH=0.804_AF_triangle_ECMC
    N=10000_h=0.8_rhoH=0.805_AF_square_ECMC
    N=10000_h=0.8_rhoH=0.805_AF_triangle_ECMC
    N=10000_h=0.8_rhoH=0.806_AF_square_ECMC
    ```
    This will deploy 11 run each on a different cpu core, each run will call `run_functions.py`.

3. `ECMC.sub` The script that condor runs that tells it to take `ecmc_lists.txt` and create the runs. 
This is called by `send_parametric_runs.py`

4. `py_env.sh` The script that tells Condor to call `run_functions.py` on each line.

5. `py_quench_env.sh` I think this is a deprecated file that did something useful once.

### simulation_results

The folder where the simulation results are saved, the `run_functions.py` 
script writes the results here. The `post_process.py` script reads the
results from here.

### deploy_post_processing_on_simulation_results_on_HTCondor

Same logic as `deploy_simulations_on_HTCondor`

