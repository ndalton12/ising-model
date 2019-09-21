import numpy as np
from matplotlib import pyplot


class BoundaryCondition:
    RBC, PBC = range(2)


class Direction:
    RIGHT, TOP, LEFT, BOTTOM = range(4)


class Ising(object):

    def __init__(self, L, J, T):
        self.L = L
        self.N = L * L
        self.TWOJ = 2. * J
        self.T = T
        self.beta = 1. / T

        # Initialize site positions
        # Initialize neighbors table for boundary conditions
        self.nn = np.zeros(shape=(self.N, 4), dtype=np.int16)
        self.position = np.zeros(shape=(L, L), dtype=np.int16)
        self.x = np.zeros(self.N, dtype=np.int16)
        self.y = np.zeros(self.N, dtype=np.int16)

        # Periodic boundary conditions
        n = 0
        for iy in range(L):
            for ix in range(L):
                self.position[iy, ix] = n
                self.x[n] = ix
                self.y[n] = iy
                self.nn[n, Direction.LEFT] = n - 1
                self.nn[n, Direction.RIGHT] = n + 1
                self.nn[n, Direction.TOP] = n + L
                self.nn[n, Direction.BOTTOM] = n - L
                if ix == 0:
                    self.nn[n, Direction.LEFT] = n + L - 1
                if ix == L - 1:
                    self.nn[n, Direction.RIGHT] = n - (L - 1)
                if iy == 0:
                    self.nn[n, Direction.BOTTOM] = n + (L - 1) * L
                if iy == L - 1:
                    self.nn[n, Direction.TOP] = n - (L - 1) * L
                n += 1

        # Initialize spins
        r = np.random.random(self.N) * 2 - 1
        self.spin = np.ones(self.N, dtype=np.int16)
        for i in range(self.N):
            if r[i] < 0:
                self.spin[i] *= -1

        self.Mtot = np.sum(self.spin)
        self.E = 0.
        for i in range(self.N):
            self.E += -J * self.spin[i] * (
                    self.spin[self.nn[i, Direction.RIGHT]] + self.spin[self.nn[i, Direction.TOP]])

        # Transition probabilities
        self.de = np.zeros(shape=(3, 9))
        self.w = np.zeros(shape=(3, 9))
        self.set_temp(self.T)

    def set_temp(self, T):
        self.T = T
        self.beta = 1. / T
        # Lookup tables for transition probabilities
        for i in range(-4, 5):
            self.de[0, i + 4] = -self.TWOJ * i
            self.de[2, i + 4] = self.TWOJ * i
            p = np.exp(-self.beta * self.de[0, i + 4])
            self.w[0, i + 4] = min(p, 1.)
            self.w[2, i + 4] = min(1. / p, 1.)

    def metropolis(self):
        nchanges = 0

        for n in range(self.N):
            # trial spin change
            # pick a random particle
            i = int(np.random.random() * self.N)

            # change in energy
            iright = self.nn[i, Direction.LEFT]
            ileft = self.nn[i, Direction.RIGHT]
            itop = self.nn[i, Direction.TOP]
            ibottom = self.nn[i, Direction.BOTTOM]

            spin_sum = self.spin[ileft] + self.spin[iright] + self.spin[itop] + self.spin[ibottom]

            s = self.spin[i]
            deltaE = self.de[s + 1, spin_sum + 4]

            if deltaE <= 0. or np.random.random() < self.w[s + 1, spin_sum + 4]:
                self.spin[i] *= -1
                self.Mtot += 2 * (-s)
                self.E += deltaE
                nchanges += 1

        return nchanges


L = 10
Nwarmup = 100
Nsteps = 1000
Ndecorr = 3
Temp = 3
J = 1.

S = Ising(L, J, Temp)
E = np.zeros(Nsteps)
M = np.zeros(Nsteps)

for i in range(Nwarmup):
    S.metropolis()

naccept = 0
for i in range(Nsteps):
    for n in range(Ndecorr):
        naccept += S.metropolis()
    E[i] = S.E
    M[i] = abs(S.Mtot)

E /= S.N
M /= S.N

Et = np.sum(E) / Nsteps
E2t = np.sum(E ** 2) / Nsteps
Mt = np.sum(M) / Nsteps
M2t = np.sum(M ** 2) / Nsteps

print("T = ", Temp)
print("<E>/N = ", Et)
print("<E^2>/N = ", E2t)
print("<M>/N = ", Mt)
print("<M^2>/N = ", M2t)
print("C=", (E2t - Et * Et) / Temp / Temp)
print("chi=", (M2t - Mt * Mt) / Temp)
print("Acceptance ratio = ", float(naccept) / S.N / Nsteps / Ndecorr)

pyplot.plot(np.arange(0, Nsteps, 1), E, ls='-', c='blue')
pyplot.xlabel("Iteration")
pyplot.ylabel("Energy")

pyplot.plot(np.arange(0, Nsteps, 1), M, ls='-', c='red')
pyplot.xlabel("Iteration")
pyplot.ylabel("Magnetization")

T = np.arange(0.2, 8, 0.2)

Mt = np.zeros(T.size)
Et = np.zeros(T.size)
M2t = np.zeros(T.size)
E2t = np.zeros(T.size)

S = Ising(L, J, 0.2)

Nsteps = 1000
Nwarmup = 1000
n = 0
for t in T:
    S.set_temp(t)
    for i in range(Nwarmup):
        S.metropolis()

    for i in range(Nsteps):
        for j in range(Ndecorr):
            S.metropolis()
        Et[n] += S.E
        Mt[n] += abs(S.Mtot)
        E2t[n] += S.E ** 2
        M2t[n] += abs(S.Mtot) ** 2

    print(t, Mt[n] / Nsteps / S.N)
    n += 1

Mt /= float(Nsteps * S.N)
Et /= float(Nsteps * S.N)
E2t /= float(Nsteps * S.N * S.N)
M2t /= float(Nsteps * S.N * S.N)
ErrorE = np.sqrt((E2t - Et ** 2) / Nsteps)
ErrorM = np.sqrt((M2t - Mt ** 2) / Nsteps)
