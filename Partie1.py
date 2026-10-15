
import numpy as np
import matplotlib.pyplot as plt

N = 50
M = 10**2

def g(x):
    return np.cos(np.pi * x / 2)

mu_hat = np.zeros(M)

mu=2/np.pi
var_g = 1/2 - (2/np.pi)**2
# Répéter l'estimation Monte-Carlo M fois
for m in range(M):
    X = np.random.uniform(0, 1, N)
    mu_hat[m] = np.mean(g(X))

# Variance empirique des M estimations
moyenne = np.mean(mu_hat)
var_empirique = np.mean((mu_hat - moyenne)**2)

#I/2
"""print("La moyenne empirique est :", moyenne)
print("La variance empirique est :", var_empirique*M/(M-1))
"""
var_mu_theorique = var_g / N
#print("Variance théorique de mu_hat :", var_mu_theorique)


#I/3
N_values = [10, 20, 50, 100, 200, 500, 1000, 2000]
erreurs = []
erreurs_theoriques = []


for N in N_values:
    X = np.random.uniform(0, 1, (M, N))
    mu_hat = np.mean(g(X), axis=1)

    # Erreur quadratique moyenne empirique
    rmse = np.sqrt(np.mean((mu_hat - mu)**2))
    erreurs.append(rmse)

    # Erreur théorique : écart-type de l'estimateur
    erreurs_theoriques.append(np.sqrt(var_g / N))

"""plt.figure(figsize=(8, 5))
plt.loglog(N_values, erreurs, 'o-', label="RMSE empirique")
plt.loglog(N_values, erreurs_theoriques, '--',
           label="Erreur théorique : 1/sqrt(N)")

plt.xlabel("Nombre de tirages N")
plt.ylabel("Erreur")
plt.title("Vitesse de convergence Monte-Carlo")
plt.legend()
plt.grid(True, which="both")
plt.show()"""

#I/4
N=50
sigma = np.sqrt(var_g / N)
# M estimations, chacune avec N tirages
X = np.random.uniform(0, 1, (M, N))
mu_hat = np.mean(g(X), axis=1)

# Centrage et réduction
Z = (mu_hat - mu) / sigma

"""print("Moyenne empirique de Z :", np.mean(Z))
print("Variance empirique de Z :", np.var(Z, ddof=1))
"""
# Histogramme et loi normale standard
z = np.linspace(-4, 4, 300)
densite = np.exp(-z**2 / 2) / np.sqrt(2 * np.pi)

"""plt.hist(Z, bins=40, density=True, alpha=0.7,
         edgecolor="black", label="Histogramme de Z")
plt.plot(z, densite, label="Loi normale N(0,1)")
plt.xlabel("Valeurs de Z")
plt.ylabel("Densité")
plt.title("Vérification numérique du TCL (N = 50)")
plt.legend()
plt.grid(True)
plt.show()"""


#################################
#II/5
X = np.linspace(0, 1, 50)
gq = np.cos(np.pi * X / 2)
q=np.ones_like(X)
qtilde= 1.5 * (1-X**2)

"""plt.plot(X, q, label="q(x)")
plt.plot(X, gq, label="g(x)q(x)")
plt.plot(X, qtilde, label="q_tilde(x)")

plt.legend()
plt.grid()
plt.show()"""

#III/2 Variance théorique

from scipy.integrate import quad
N=50

def integrande(x):
    if x == 1:
        return 0
    return (2/3) * np.cos(np.pi*x/2)**2 / (1-x**2)

E_H2, _ = quad(integrande, 0, 1)

var_H = E_H2 - mu**2
var_mu = var_H / N
#print("Variance théorique de mu_hat :", var_mu)

# III/3 methode: acceptation et de rejet

M = 10**5

# Tirages selon la loi uniforme
X = np.random.uniform(0, 1, M)
U = np.random.uniform(0, 1, M)

# Acceptation-rejet pour q_tilde(x) = 3/2 * (1 - x**2)
acceptes = U < (1 - X**2)

# Probabilité d'acceptation empirique
P_empirique = np.mean(acceptes)

# Probabilité d'acceptation théorique
P_theorique = 2/3

"""print("Probabilité théorique :", P_theorique)
print("Probabilité empirique :", P_empirique)
print("Écart :", abs(P_empirique - P_theorique))
"""
# Conserver les échantillons acceptés
X_acceptes = X[acceptes]

#print("Nombre de valeurs acceptées :", len(X_acceptes))

## III/4

import numpy as np

M = 10000
N = 50

estimations = []

for _ in range(M):
    X = []

    while len(X) < N:
        x = np.random.uniform(0, 1)
        u = np.random.uniform(0, 1)

        if u < 1 - x**2:
            X.append(x)

    X = np.array(X)

    H = np.cos(np.pi * X / 2) / (1.5 * (1 - X**2))
    mu_hat = np.mean(H)

    estimations.append(mu_hat)

estimations = np.array(estimations)

variance_empirique = np.var(estimations, ddof=1)
"""
print("Moyenne empirique :", np.mean(estimations))
print("Variance empirique :", variance_empirique)
"""


# IV/2 : Validation du TCL pour les deux méthodes

mu = 2 / np.pi
N = 50
M = 10000

var_g = 1/2 - (2/np.pi)**2
var_MC = var_g / N
var_IS = 1.9816618803725205e-05

# Recalculer les estimations Monte-Carlo classiques
X_mc = np.random.uniform(0, 1, (M, N))
mu_hat_MC = np.mean(g(X_mc), axis=1)

# Centrer et réduire
Z_MC = (mu_hat_MC - mu) / np.sqrt(var_MC)
Z_IS = (estimations - mu) / np.sqrt(var_IS)

# Loi normale standard
z = np.linspace(-4, 4, 300)
phi = np.exp(-z**2 / 2) / np.sqrt(2 * np.pi)

"""# Histogramme Monte-Carlo classique
plt.figure()
plt.hist(Z_MC, bins=30, density=True, alpha=0.6,
         label="Histogramme empirique")
plt.plot(z, phi, label="Loi normale N(0,1)")
plt.title("TCL - Monte-Carlo classique")
plt.xlabel("Z")
plt.ylabel("Densité")
plt.legend()
plt.show()

# Histogramme Importance Sampling
plt.figure()
plt.hist(Z_IS, bins=30, density=True, alpha=0.6,
         label="Histogramme empirique")
plt.plot(z, phi, label="Loi normale N(0,1)")
plt.title("TCL - Importance Sampling")
plt.xlabel("Z")
plt.ylabel("Densité")
plt.legend()
plt.show()"""


#IV 3
N = 50
M = 10000

# Inversion de la CDF de q_opt
U = np.random.uniform(0, 1, (M, N))
X_opt = (2 / np.pi) * np.arccos(U)

# Estimateur d'importance sampling
H_opt = (
    np.cos(np.pi * X_opt / 2)
    / ((np.pi / 2) * np.cos(np.pi * X_opt / 2))
)

mu_hat_opt = np.mean(H_opt, axis=1)

print("Moyenne empirique :", np.mean(mu_hat_opt))
print("Variance empirique :", np.var(mu_hat_opt, ddof=1))


# IV/4 : q_tilde(x) = 2(1-x), inversion de la CDF
M = 10000
N = 50
mu = 2 / np.pi

# 1. Variance théorique
def integrande(x):
    if x == 1:
        return 0
    return np.cos(np.pi * x / 2)**2 / (2 * (1 - x))

E_H2, _ = quad(integrande, 0, 1)
var_H = E_H2 - mu**2
var_theorique = var_H / N

# 2. Simulation par inversion de la CDF
estimations_q4 = np.zeros(M)

for m in range(M):
    U = np.random.uniform(0, 1, N)
    X = 1 - np.sqrt(1 - U)

    H = np.cos(np.pi * X / 2) / (2 * (1 - X))
    estimations_q4[m] = np.mean(H)

# 3. Variance empirique
moyenne_empirique = np.mean(estimations_q4)
variance_empirique = np.var(estimations_q4, ddof=1)

print("Moyenne empirique :", moyenne_empirique)
print("Variance théorique :", var_theorique)
print("Variance empirique :", variance_empirique)
print("Variance Monte-Carlo classique :", var_g / N)
