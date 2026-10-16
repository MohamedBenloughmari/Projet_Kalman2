from math import sin, cos, atan2, pi
import time
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parametres generaux (a modifier si besoin)
nSteps = 2000           # longueur de la simulation
nLandmarks = 20         # nombre d'amers
SEED_WORLD = 0          # graine pour la carte, l'odometrie et les mesures
SEED_FILTER = 1         # graine pour les tirages du filtre

QTrue = np.diag([0.02, 0.02, 1 * pi / 180]) ** 2    # bruit odometrie (simulation)
PYTrue = np.diag([0.5, 1 * pi / 180]) ** 2          # bruit mesure (simulation)

x0 = np.array([[1, -50, 0]]).T                      # pose initiale vraie
P0 = np.diag([1, 1, 0.1])                           # ecarts-types du nuage initial

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# Fonctions du code fourni
def angle_wrap(a):
    if a > np.pi:
        a = a - 2 * pi
    elif a < -np.pi:
        a = a + 2 * pi
    return a


def tcomp(tab, tbc):
    """Composition de deux poses : tab (+) tbc"""
    result = angle_wrap(tab[2, 0] + tbc[2, 0])
    s = sin(tab[2, 0])
    c = cos(tab[2, 0])
    tac = tab[0:2] + np.array([[c, -s], [s, c]]) @ tbc[0:2]
    return np.vstack((tac, result))


def get_robot_control(k):
    return np.array([[0, 0.1, 0.1 * np.pi / 180 * sin(3 * np.pi * k / nSteps)]]).T


def observation_model(xVeh, iFeature, Map):
    Delta = Map[0:2, iFeature:(iFeature + 1)] - xVeh[0:2]
    z = np.array([[np.linalg.norm(Delta)],
                  [atan2(Delta[1, 0], Delta[0, 0]) - xVeh[2, 0]]])
    z[1, 0] = angle_wrap(z[1, 0])
    return z


def re_sampling(px, pw, N):
    """Reechantillonnage a faible variance (comme dans le code fourni)"""
    w_cum = np.cumsum(pw / np.sum(pw))
    base = np.arange(0.0, 1.0, 1 / N)[:N]
    re_sample_id = base + np.random.uniform(0, 1 / N)
    indexes = []
    ind = 0
    for ip in range(N):
        while ind < N - 1 and re_sample_id[ip] > w_cum[ind]:
            ind += 1
        indexes.append(ind)
    return px[:, indexes], pw[indexes]


# Simulation du monde (faite une seule fois, puis rejouee)
def simulate_world(outage=None):
    """Simule la carte, la trajectoire vraie, l'odometrie et les mesures.
    outage = (a, b) : pas de mesure pour a < k < b."""
    np.random.seed(SEED_WORLD)
    Map = 120 * np.random.rand(2, nLandmarks) - 60
    xTrue = x0.copy()
    xOdom = x0.copy()
    hxTrue = [xTrue]
    hxOdom = [xOdom]
    U = [None]           # odometrie bruitee u_k
    Z = [None]           # mesures z_k (None si pas de mesure)
    IF = [-1]            # indice de l'amer observe
    for k in range(1, nSteps):
        u = get_robot_control(k)
        xTrue = tcomp(xTrue, u)
        # odometrie bruitee
        uNoise = np.array([np.sqrt(QTrue) @ np.random.randn(3)]).T
        xOdom = tcomp(tcomp(xOdom, u), uNoise)
        U.append(u + uNoise)
        # observation d'un amer au hasard
        iFeature = np.random.randint(0, Map.shape[1] - 1)
        zNoise = np.array([np.sqrt(PYTrue) @ np.random.randn(2)]).T
        z = observation_model(xTrue, iFeature, Map) + zNoise
        z[1, 0] = angle_wrap(z[1, 0])
        if outage is not None and outage[0] < k < outage[1]:
            z, iFeature = None, -1
        Z.append(z)
        IF.append(iFeature)
        hxTrue.append(xTrue)
        hxOdom.append(xOdom)
    return {"Map": Map, "U": U, "Z": Z, "IF": IF,
            "xTrue": np.hstack(hxTrue), "xOdom": np.hstack(hxOdom)}


# Filtre particulaire
def particle_filter(world, N=1000, P0=P0, QEst=2 * QTrue, PYEst=2 * PYTrue,
                    center=x0, init="gauss", theta=None):

    np.random.seed(SEED_FILTER)
    Map = world["Map"]

    # nuage initial
    if init == "uniform":
        xParticles = np.vstack((120 * np.random.rand(2, N) - 60,
                                np.random.uniform(-pi, pi, (1, N))))
    else:
        xParticles = center + P0 @ np.random.randn(3, N)
        for p in range(N):
            xParticles[2, p] = (xParticles[2, p] + pi) % (2 * pi) - pi
    L = np.ones(N) / N

    xEst = np.average(xParticles, axis=1, weights=L)
    hxEst = [xEst]
    hxSTD = [np.sqrt(np.average((xParticles - xEst[:, None]) ** 2, axis=1, weights=L))]
    nResampling = 0

    t0 = time.time()
    for k in range(1, nSteps):
        u = world["U"][k]

        # 1) prediction
        for p in range(N):
            noise = np.sqrt(QEst) @ np.random.randn(3, 1)
            xParticles[:, p] = tcomp(xParticles[:, p:p + 1], u + noise).squeeze()

        # 2) correction
        z = world["Z"][k]
        if z is not None:
            iFeature = world["IF"][k]
            for p in range(N):
                zPred = observation_model(xParticles[:, p:p + 1], iFeature, Map)
                Innov = z - zPred
                Innov[1, 0] = angle_wrap(Innov[1, 0])
                lik = np.exp(-0.5 * Innov.T @ np.linalg.inv(PYEst) @ Innov).item() + 1e-3
                if theta is None:
                    L[p] = lik            # code fourni : poids = vraisemblance
                else:
                    L[p] = L[p] * lik     # SIR : w_k = w_{k-1} * vraisemblance
        if theta is not None:
            L = L / np.sum(L)

        # 3) estimation
        xEst = np.average(xParticles, axis=1, weights=L)
        xSTD = np.sqrt(np.average((xParticles - xEst[:, None]) ** 2, axis=1,
                                  weights=L / np.sum(L)))
        hxEst.append(xEst)
        hxSTD.append(xSTD)

        # 4) reechantillonnage
        if theta is None:
            xParticles, L = re_sampling(xParticles, L, N)
            nResampling += 1
        else:
            Neff = 1 / np.sum(L ** 2)
            if Neff < theta * N:
                xParticles, L = re_sampling(xParticles, L, N)
                L = np.ones(N) / N
                nResampling += 1

    hxEst = np.array(hxEst).T
    hxSTD = np.array(hxSTD).T
    ex = hxEst[0] - world["xTrue"][0]
    return {"xEst": hxEst, "ex": ex, "sx": hxSTD[0],
            "time": time.time() - t0, "nResampling": nResampling}


# Outils d'affichage
def plot_three(results, titles, filename, kmin=0, kmax=None, ylim=(-3, 3),
               outage=None):
    """Trace e_x et +/- 3 sigma_x pour 3 configurations cote a cote."""
    if kmax is None:
        kmax = nSteps
    k = np.arange(kmin, kmax)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, res, title in zip(axs, results, titles):
        if outage is not None:
            ax.axvspan(outage[0], outage[1], color="lightgray", label="panne")
        ax.plot(k, res["ex"][kmin:kmax], "b", label="erreur $e_x$")
        ax.plot(k, 3 * res["sx"][kmin:kmax], "r", label=r"$\pm 3\sigma_x$")
        ax.plot(k, -3 * res["sx"][kmin:kmax], "r")
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_title(title)
        ax.set_xlabel("k")
        ax.grid(True)
    axs[0].set_ylabel("erreur sur x (m)")
    axs[0].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=150)
    plt.close()


def show_table(rows, title):
    df = pd.DataFrame(rows)
    print("\n" + title)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return df


# Etude 0 : scenario
def etude_scenario(world):
    res = particle_filter(world, N=1000)
    plt.figure(figsize=(5, 5))
    plt.plot(world["Map"][0], world["Map"][1], "*k", label="amers")
    plt.plot(world["xTrue"][0], world["xTrue"][1], "k", label="vraie")
    plt.plot(world["xOdom"][0], world["xOdom"][1], "g", label="odometrie")
    plt.plot(res["xEst"][0], res["xEst"][1], "--r", label="filtre (N=1000)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, "scenario.png"), dpi=150)
    plt.close()
    return res


# Etude 1 : nombre de particules
def etude_nparticles(world, res1000=None):
    N_list = [10, 500, 1000, 1500, 2000]
    rows, runs = [], {}
    for N in N_list:
        res = res1000 if (N == 1000 and res1000 is not None) else particle_filter(world, N=N)
        runs[N] = res
        rows.append({"N": N, "moy |e_x| (m)": np.mean(np.abs(res["ex"])),
                     "var e_x (m2)": np.var(res["ex"]), "temps (s)": res["time"]})
        print(f"  N = {N} fini ({res['time']:.1f} s)")
    show_table(rows, "Etude 1 : nombre de particules")
    plot_three([runs[10], runs[1000], runs[2000]], ["N = 10", "N = 1000", "N = 2000"],
               "nparticles.png")
    # erreur et temps en fonction de N
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    a1.plot(N_list, [r["moy |e_x| (m)"] for r in rows], "o-", label="moyenne de |e_x|")
    a1.plot(N_list, [r["var e_x (m2)"] for r in rows], "s--", label="variance de e_x")
    a1.set_xlabel("N")
    a1.legend()
    a1.grid(True)
    a2.plot(N_list, [r["temps (s)"] for r in rows], "o-")
    a2.set_xlabel("N")
    a2.set_ylabel("temps de calcul (s)")
    a2.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "nparticles_resume.png"), dpi=150)
    plt.close()


# Etude 2 : variance initiale
def etude_init(world, res_ref=None):
    """Nuage initial d'ecart-type s * P0, deux scenarios :
    - cas A : centre sur la vraie pose x0 (comme le code fourni) ;
    - cas B : centre sur une estimee fausse x0 + delta ;
    et localisation globale (nuage uniforme)."""
    delta = np.array([[3.0, -3.0, 0.3]]).T
    s_list = [0.1, 1, 10, 50]
    rows, runsA, runsB = [], {}, {}
    for s in s_list:
        # cas A (centre sur x0) ; s = 1 reutilise le resultat de reference
        if s == 1 and res_ref is not None:
            resA = res_ref                     # meme reglage que le code fourni
        else:
            resA = particle_filter(world, N=1000, P0=s * P0, center=x0)
        runsA[s] = resA
        rows.append({"cas": "A", "nuage": f"s = {s}",
                     "moy |e_x| (m)": np.mean(np.abs(resA["ex"])),
                     "max |e_x| (m)": np.max(np.abs(resA["ex"])),
                     "|e_x| final (m)": abs(resA["ex"][-1])})
        # cas B (centre sur x0 + delta)
        resB = particle_filter(world, N=1000, P0=s * P0, center=x0 + delta)
        runsB[s] = resB
        rows.append({"cas": "B", "nuage": f"s = {s}",
                     "moy |e_x| (m)": np.mean(np.abs(resB["ex"])),
                     "max |e_x| (m)": np.max(np.abs(resB["ex"])),
                     "|e_x| final (m)": abs(resB["ex"][-1])})
        print(f"  s = {s} fini")
    plot_three([runsA[0.1], runsA[1], runsA[10]], ["s = 0.1", "s = 1", "s = 10"],
               "init_A.png", kmax=300, ylim=(-6, 6))
    plot_three([runsB[0.1], runsB[1], runsB[10]], ["s = 0.1", "s = 1", "s = 10"],
               "init_B.png", kmax=300, ylim=(-6, 6))
    res = particle_filter(world, N=1000, init="uniform")
    rows.append({"cas": "-", "nuage": "uniforme",
                 "moy |e_x| (m)": np.mean(np.abs(res["ex"])),
                 "max |e_x| (m)": np.max(np.abs(res["ex"])),
                 "|e_x| final (m)": abs(res["ex"][-1])})
    show_table(rows, "Etude 2 : variance initiale (A : centre sur x0, B : centre sur x0 + delta)")


# Etude 3 : bruits du filtre
def etude_bruits(world, res_ref=None):
    alphas = [0.1, 1, 2, 10]
    rows, runsQ, runsR = [], {}, {}
    for a in alphas:
        if a == 2 and res_ref is not None:
            resQ = resR = res_ref
        else:
            resQ = particle_filter(world, N=1000, QEst=a * QTrue)
            resR = particle_filter(world, N=1000, PYEst=a * PYTrue)
        runsQ[a], runsR[a] = resQ, resR
        for name, res in (("alpha_Q", resQ), ("alpha_R", resR)):
            rows.append({"parametre": name, "alpha": a,
                         "moy |e_x| (m)": np.mean(np.abs(res["ex"])),
                         "var e_x (m2)": np.var(res["ex"]),
                         "moy sigma_x (m)": np.mean(res["sx"])})
        print(f"  alpha = {a} fini")
    show_table(rows, "Etude 3 : bruits du filtre (QEst = alpha_Q QTrue, PYEst = alpha_R PYTrue)")
    plot_three([runsQ[0.1], runsQ[1], runsQ[10]],
               [r"$\alpha_Q = 0.1$", r"$\alpha_Q = 1$", r"$\alpha_Q = 10$"], "bruit_Q.png")
    plot_three([runsR[0.1], runsR[1], runsR[10]],
               [r"$\alpha_R = 0.1$", r"$\alpha_R = 1$", r"$\alpha_R = 10$"], "bruit_R.png")


# Etude 4 : panne capteur
def etude_panne(outage=(400, 500)):
    world_p = simulate_world(outage=outage)
    alphas = [0.1, 1, 10]
    rows, runs = [], {}
    a, b = outage
    for al in alphas:
        res = particle_filter(world_p, N=1000, QEst=al * QTrue)
        runs[al] = res
        rows.append({"alpha_Q": al,
                     "max |e_x| panne (m)": np.max(np.abs(res["ex"][a + 1:b])),
                     "3 sigma_x fin panne (m)": 3 * res["sx"][b - 1],
                     "max |e_x| apres (m)": np.max(np.abs(res["ex"][b:b + 200]))})
        print(f"  alpha_Q = {al} fini")
    show_table(rows, f"Etude 4 : panne capteur {a} < k < {b}")
    plot_three([runs[0.1], runs[1], runs[10]],
               [r"$\alpha_Q = 0.1$", r"$\alpha_Q = 1$", r"$\alpha_Q = 10$"],
               "panne.png", kmin=a - 100, kmax=b + 200, ylim=(-3, 3), outage=outage)


# Etude 5 : filtre SIR
def etude_sir(world, res_ref=None):
    thetas = [0, 0.5, None]
    names = ["jamais (theta = 0)", "SIR theta = 0.5", "a chaque pas (code)"]
    rows, runs = [], []
    for th, name in zip(thetas, names):
        res = res_ref if (th is None and res_ref is not None) else particle_filter(world, N=1000, theta=th)
        runs.append(res)
        rows.append({"reechantillonnage": name,
                     "moy |e_x| (m)": np.mean(np.abs(res["ex"])),
                     "max |e_x| (m)": np.max(np.abs(res["ex"])),
                     "nb reech.": res["nResampling"]})
        print(f"  {name} fini")
    show_table(rows, "Etude 5 : filtre SIR")
    plot_three(runs, names, "sir.png", ylim=None)


if __name__ == "__main__":
    t_start = time.time()
    world = simulate_world()
    print("Scenario")
    ref = etude_scenario(world)                 # N = 1000, reglages du code
    print("Etude 1"); etude_nparticles(world, ref)
    print("Etude 2"); etude_init(world, ref)
    print("Etude 3"); etude_bruits(world, ref)
    print("Etude 4"); etude_panne()
    print("Etude 5"); etude_sir(world, ref)
    print(f"\nTermine en {time.time() - t_start:.0f} s, figures dans {FIG_DIR}")
