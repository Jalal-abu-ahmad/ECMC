import time

from EventChainActions import *
from order_parameter import OrderParameter

epsilon = 1e-8
day = 86400  # sec


class PositionalCorrelationFunction(OrderParameter):

    def __init__(self, sim_path, m, n, rect_width=0.1, centers=None, spheres_ind=None, calc_upper_lower=False):
        super().__init__(sim_path, centers, spheres_ind, calc_upper_lower, rect_width=rect_width)
        self.rect_width = rect_width
        self.m, self.n = m, n
        self.op_name = "pos"

    def correlation(self, bin_width=0.1, calc_upper_lower=False, low_memory=True, randomize=False,
                    realizations=int(1e7), time_limit=2 * day):
        psi = PsiMN(self.sim_path, self.m, self.n, centers=self.spheres, spheres_ind=self.spheres_ind)
        psi.calc_order_parameter()
        self.theta = psi.rotate_spheres(calc_spheres=False)
        self.correlation_name = "correlation_theta=" + str(self.theta)
        theta, rect_width = self.theta, self.rect_width
        v_hat = np.array([np.cos(theta), np.sin(theta)])
        lx, ly = self.l_x, self.l_y
        l = np.sqrt(lx ** 2 + ly ** 2) / 2
        bins_edges = np.linspace(0, np.ceil(l / bin_width) * bin_width, int(np.ceil(l / bin_width)) + 1)
        kmax = len(bins_edges) - 1
        self.corr_centers = bins_edges[:-1] + bin_width / 2
        self.counts = np.zeros(len(self.corr_centers))
        init_time = time.time()
        N = len(self.spheres)
        if low_memory:
            if randomize:
                for realization in range(realizations):
                    i, j = random.randint(0, N - 1), random.randint(0, N - 1)
                    if i == j: continue
                    k = self.__pair_dist__(self.spheres[i], self.spheres[j], v_hat, rect_width, bin_width)
                    if k is not None:
                        k = int(min(k, kmax))
                        self.counts[k] += 1
                    if realization % 1000 == 0 and time.time() - init_time > time_limit:
                        break
            else:
                for i in range(N):
                    for j in range(N):
                        if j == i: continue
                        k = self.__pair_dist__(self.spheres[i], self.spheres[j], v_hat, rect_width, bin_width)
                        if k is not None:
                            k = int(min(k, kmax))
                            self.counts[k] += 1
                realization = N * (N - 1)
        else:
            x = np.array([r[0] for r in self.spheres])
            y = np.array([r[1] for r in self.spheres])
            N = len(x)
            dx = (x.reshape((N, 1)) - x.reshape((1, N))).reshape(N ** 2, )
            dy = (y.reshape((N, 1)) - y.reshape((1, N))).reshape(N ** 2, )
            A = np.transpose([dx, dx + lx, dx - lx])
            I = np.argmin(np.abs(A), axis=1)
            J = [i for i in range(len(I))]
            dx = A[J, I]
            A = np.transpose([dy, dy + ly, dy - ly])
            I = np.argmin(np.abs(A), axis=1)
            dy = A[J, I]

            pairs_dr = np.transpose([dx, dy])

            m = lambda A, B: np.matmul(A, B)
            dist_vec = m(v_hat.reshape(2, 1), m(pairs_dr, v_hat).reshape(1, N)).T - pairs_dr
            dist_to_line = np.linalg.norm(dist_vec, axis=1)
            I = np.where(dist_to_line <= rect_width / 2)[0]
            pairs_dr = pairs_dr[I]
            J = np.where(m(pairs_dr, v_hat) > 0)[0]
            pairs_dr = pairs_dr[J]
            rs = m(pairs_dr, v_hat)
            self.counts, _ = np.histogram(rs, bins_edges)
            realization = N ** 2
        print("\nTime Passed: " + str((time.time() - init_time) / day) + " days.\nSummed " + str(
            realization) + " pairs")

        # normalize counts
        l_x, l_y, l_z, rad, rho_H, edge, n_row, n_col = self.write_or_load.load_Input()
        rho2D = self.N / (l_x * l_y)
        # counts --> counts*N*(N-1)/realization-->counts*(N*(N-1)/realization)*(1/(rho*a*N))
        self.op_corr = self.counts / (rho2D * bin_width * rect_width * realization / (self.N - 1))

        if calc_upper_lower:
            assert self.upper is not None, \
                "Failed calculating upper positional correlation because it was not initialized"
            self.upper.correlation(bin_width=bin_width, low_memory=low_memory, randomize=randomize,
                                   realizations=realizations)
            assert self.upper is not None, \
                "Failed calculating lower positional correlation because it was not initialized"
            self.lower.correlation(bin_width=bin_width, low_memory=low_memory, randomize=randomize,
                                   realizations=realizations)

    def __pair_dist__(self, r, r_, v_hat, rect_width, bin_width):
        lx, ly = self.l_x, self.l_y
        dr = np.array(r) - r_
        dxs = [dr[0], dr[0] + lx, dr[0] - lx]
        dx = dxs[np.argmin(np.abs(dxs))]
        dys = [dr[1], dr[1] + ly, dr[1] - ly]
        dy = dys[np.argmin(np.abs(dys))]
        dr = np.array([dx, dy])
        dist_on_line = float(np.dot(dr, v_hat))
        dist_vec = v_hat * dist_on_line - dr
        dist_to_line = np.linalg.norm(dist_vec)
        if dist_to_line <= rect_width / 2 and dist_on_line > 0:
            k = int(np.floor(dist_on_line / bin_width))
            return k
        else:
            return None
