import sys
from datetime import date

from SnapShot import *
from bragg_structure import BraggStructure
from burger_field import BurgerField
from ising import Ising
from largest_component import LargestComponent
from local_orientation import LocalOrientation
from magnetic_bragg_structure import MagneticBraggStructure
from magnetic_topological_corr import MagneticTopologicalCorr
from order_parameter import OrderParameter
from positional_correlation_function import PositionalCorrelationFunction
from psi_mn import PsiMN

epsilon = 1e-8
day = 86400  # sec


def main(sim_name, calc_type):
    # TODO: add odd loops removal and histogram
    correlation_kwargs = {'randomize': False, 'time_limit': 2 * day}
    prefix = "./simulation_results/"
    sim_path = os.path.join(prefix, sim_name)
    op_dir = os.path.join(sim_path, "OP")
    log = os.path.join(op_dir, "log")
    sys.stdout = open(log, "a")
    if calc_type.endswith("23"):
        m, n = 2, 3
    if calc_type.endswith("14"):
        m, n = 1, 4
    if calc_type.endswith("16"):
        m, n = 1, 6
    calc_correlations, calc_mean, calc_vec, calc_all_reals = True, True, True, True
    if calc_type.startswith("psi"):
        op = PsiMN(sim_path, m, n)
        if calc_type.startswith("psi_mean"):
            calc_correlations = False
    if calc_type.startswith("pos"):
        op = PositionalCorrelationFunction(sim_path, m, n)
        calc_mean, calc_vec = False, False
    if calc_type.startswith("BurgersSquare"):
        try:
            radius = int(calc_type.split('_')[1].split('=')[1])
            op = BurgerField(sim_path, orientation_rad=radius)
        except Exception:
            op = BurgerField(sim_path)
        calc_mean, calc_correlations = False, False
    if calc_type.startswith("Bragg_S"):
        if calc_type.startswith("Bragg_Sm"):
            op = MagneticBraggStructure(sim_path, m, n)
        else:
            op = BraggStructure(sim_path, m, n)
    if calc_type.startswith("gM"):
        op = MagneticTopologicalCorr(sim_path, k_nearest_neighbors=n)
        calc_mean = False
        correlation_kwargs = {}
    if calc_type.startswith('Ising'):
        op = Ising(sim_path, k_nearest_neighbors=n)
        calc_correlations = False
        calc_mean = False
        correlation_kwargs = {}
        calc_all_reals = False
        if calc_type.find('E_T') >= 0:
            if not (calc_type.find('real') >= 0):
                op.correlation(**correlation_kwargs)
                op.write(write_correlations=True, write_vec=False)
            else:
                real = re.split('(_|real=)', calc_type)[-3]
                centers = np.loadtxt(os.path.join(op.sim_path, str(real)))
                op.update_centers(centers, real)
                op.correlation(**correlation_kwargs)
                op.write(write_correlations=True, write_vec=False)
            calc_vec = False
            # no matter if op.corr_path exists or not, run correlation in this case
    if calc_type.startswith('LocalPsi'):
        radius = int(calc_type.split('_')[1].split('=')[1])
        op = LocalOrientation(sim_path, m, n, radius=radius)
        # radius=10 for H=1.8, rhoH=0.8 gives N=(pi*r^2)*H*rhoH/sig^3~56 particles
        correlation_kwargs = {}
        calc_all_reals = False
    if calc_type.startswith('LargestComponent'):
        op = LargestComponent(sim_path, k_nearest_neighbors=n)
        calc_mean = False
        correlation_kwargs = {}
        calc_all_reals = False
    print(
        "\n\n\n-----------\nDate: " + str(date.today()) + "\nType: " + calc_type + "\nCorrelation arguments:" + str(
            correlation_kwargs) + "\nCalc correlations: " + str(calc_correlations) + "\nCalc mean: " + str(
            calc_mean), file=sys.stdout)
    if calc_all_reals:
        op.calc_for_all_realizations(calc_correlations=calc_correlations, calc_mean=calc_mean, calc_vec=calc_vec,
                                     **correlation_kwargs)
    else:
        if calc_vec:
            op.read_or_calc_write()
        if calc_correlations and (not OrderParameter.exists(op.corr_path)):
            op.correlation(**correlation_kwargs)
            op.write(write_correlations=True, write_vec=False)


if __name__ == "__main__":
    local_run = True
    if local_run:
        sim_name = "N=4_h=0.8_rhoH=0.801_AF_square_ECMC"
        calc_type = "Ising-annealing14"
    else:
        sim_name = sys.argv[1]
        calc_type = sys.argv[2]
    main(sim_name=sim_name, calc_type=calc_type)
