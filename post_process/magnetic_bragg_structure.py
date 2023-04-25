from EventChainActions import *
from bragg_structure import BraggStructure

epsilon = 1e-8
day = 86400  # sec


class MagneticBraggStructure(BraggStructure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.op_name = "Bragg_Sm"

    def calc_eikr(self, k):
        # sum_n(z_n*e^(ikr_n))
        # z in [rad,lz-rad]-->z in [-1,1]: (z-lz/2)/(lz/2-rad)
        # For z=rad we have (rad-lz/2)/(lz/2-rad)=-1
        # For z=lz-rad we have (lz-rad-lz/2)/(lz/2-rad)=1.
        rad, lz = 1.0, self.l_z
        return np.array(
            [(r[2] - lz / 2) / (lz / 2 - rad) * np.exp(1j * (k[0] * r[0] + k[1] * r[1])) for r in self.spheres])

    def k_perf(self):
        l = np.sqrt(self.l_x * self.l_y / len(self.spheres))
        if self.m == 1 and self.n == 4:
            return np.pi / l * np.array([1, 1])
        if self.m == 1 and self.n == 6:
            a = np.sqrt(2.0 / np.sqrt(3)) * l
            return np.pi / a * np.array([1.0, 1.0 / np.sqrt(3)])
        if self.m == 2 and self.n == 3:
            a = np.sqrt(4.0 / np.sqrt(3)) * l
            return np.pi / a * np.array([2.0, 2.0])
        raise NotImplementedError
