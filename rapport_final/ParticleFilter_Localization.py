"""
TP particle filter for mobile robots localization

authors: Goran Frehse and David Filliat
"""

from math import sin, cos, atan2, pi, ceil, sqrt
import matplotlib.pyplot as plt
import numpy as np
import time

# ---- Helper functions ----

# Init displays
show_animation = True
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 7))
ax3 = plt.subplot(3, 2, 2)
ax4 = plt.subplot(3, 2, 4)
ax5 = plt.subplot(3, 2, 6)


# fit angle between -pi and pi
def angle_wrap(a):
    if (a > np.pi):
        a = a - 2 * pi
    elif (a < -np.pi):
        a = a + 2 * pi
    return a


# composes two transformations
def tcomp(tab, tbc):
    assert tab.ndim == 2
    assert tbc.ndim == 2

    result = tab[2, 0] + tbc[2, 0]
    result = angle_wrap(result)
    s = sin(tab[2, 0])
    c = cos(tab[2, 0])
    tac = tab[0:2] + np.array([[c, -s], [s, c]]) @ tbc[0:2]
    tac = np.vstack((tac, result))

    return tac


# ---- Simulator functions ----

# return true control at step k
def get_robot_control(k):
    global nSteps
    # generate  sin trajectory
    u = np.array([[0, 0.1,  0.1*np.pi / 180 * sin(3*np.pi * k / nSteps)]]).T
    return u


# simulate new true robot position
def simulate_world(k):
    global xTrue
    u = get_robot_control(k)
    xTrue = tcomp(xTrue, u)
    xTrue[2, 0] = angle_wrap(xTrue[2, 0])


# computes and returns noisy odometry
def get_odometry(k):
    global xOdom  # internal to robot low-level controller
    global QTrue

    u = get_robot_control(k)
    xnow = tcomp(xOdom, u)
    uNoise = np.sqrt(QTrue) @ np.random.randn(3)
    uNoise = np.array([uNoise]).T
    xnow = tcomp(xnow, uNoise)
    xOdom = xnow
    u = u + uNoise
    return xnow, u


# generate a noisy observation of a random feature
def get_observation(k):
    global Map, xTrue, PYTrue
    if (k > 100 and k < 100):
        z = None
        iFeature = -1
    else:
        iFeature = np.random.randint(0, Map.shape[1] - 1)
        zNoise = np.sqrt(PYTrue) @ np.random.randn(2)
        zNoise = np.array([zNoise]).T
        z = observation_model(xTrue, iFeature, Map) + zNoise
        z[1, 0] = angle_wrap(z[1, 0])
    return [z, iFeature]


# ---- model functions ----


# evolution model (f)
def motion_model(x, u):
    xPred = tcomp(x, u)
    xPred[2, 0] = angle_wrap(xPred[2, 0])
    return xPred


# observation model (h)
def observation_model(xVeh, iFeature, Map):
    Delta = Map[0:2, iFeature:(iFeature+1)]-xVeh[0:2]
    z = np.array([[np.linalg.norm(Delta)], [
        atan2(Delta[1, 0], Delta[0, 0]) - xVeh[2, 0]]])
    z[1, 0] = angle_wrap(z[1, 0])
    return z


# ---- particle filter implementation ----


# Particle filter resampling
def re_sampling(px, pw):
    """
    low variance re-sampling
    """

    w_cum = np.cumsum(pw / np.sum(pw))
    base = np.arange(0.0, 1.0, 1 / nParticles)
    re_sample_id = base + np.random.uniform(0, 1 / nParticles)
    indexes = []
    ind = 0
    for ip in range(nParticles):
        while re_sample_id[ip] > w_cum[ind]:
            ind += 1
        indexes.append(ind)

    px = px[:, indexes]
    pw = pw[indexes]

    return px, pw


# Nb of particle in the filter
nParticles = 1000

# Simulation length
nSteps = 1000

# Location of landmarks
Map = 120*np.random.rand(2, 20)-60

# True covariance of errors used for simulating robot movements
QTrue = np.diag([0.02, 0.02, 1*pi/180]) ** 2
PYTrue = np.diag([0.5, 1*pi/180]) ** 2

# Modeled errors used in the Kalman filter process
QEst = 2 * np.eye(3, 3) @ QTrue
PYEst = 2 * np.eye(2, 2) @ PYTrue

# initial conditions
xTrue = np.array([[1, -50, 0]]).T
xOdom = xTrue

# initial conditions: - a point cloud around truth
xParticles = xTrue + np.diag([1, 1, 0.1]) @ np.random.randn(3, nParticles)

# initial conditions: global localization
# xParticles = 120 * np.random.rand(3, nParticles)-60

# initial weights
L = np.ones((nParticles))/nParticles

# initial estimate
xEst = np.average(xParticles, axis=1, weights=L)
xEst = np.expand_dims(xEst, axis=1)
xSTD = np.sqrt(np.average((xParticles-xEst)*(xParticles-xEst),
               axis=1, weights=L))
xSTD = np.expand_dims(xSTD, axis=1)

# Init history matrixes
hxEst = xEst
hxTrue = xTrue
hxOdom = xOdom
err = xEst - xTrue
err[2, 0] = angle_wrap(err[2, 0])
hxError = err
hxSTD = xSTD

for k in range(1, nSteps):
    # Simlulate robot motion
    simulate_world(k)

    # Get odometry measurements
    xOdom, u = get_odometry(k)

    # do prediction
    # for each particle we add control vector AND noise
    for p in range(nParticles):
        xParticles[:, p] = tcomp(xParticles[:, p:p+1], u +
                                 np.sqrt(QEst) @ np.random.randn(3, 1)).squeeze()

    # observe a random feature
    [z, iFeature] = get_observation(k)

    if z is not None:
        for p in range(nParticles):
            # Predict observation from the particle position
            zPred = observation_model(xParticles[:, p:p+1], iFeature, Map)

            # Innovation : perception error
            Innov = z-zPred
            Innov[1] = angle_wrap(Innov[1])

            # Compute particle weight using gaussian model
            L[p] = np.exp(-0.5 * (Innov.T @ np.linalg.inv(PYEst) @ Innov).item()) + 1e-3

    # Compute position as weighted mean of particles
    xEst = np.average(xParticles, axis=1, weights=L)
    xEst = np.expand_dims(xEst, axis=1)

    # Compute particles std deviation
    xSTD = np.sqrt(np.average((xParticles-xEst)*(xParticles-xEst),
                   axis=1, weights=L / np.sum(L)))
    xSTD = np.expand_dims(xSTD, axis=1)

    # Particle resampling
    xParticles, L = re_sampling(xParticles, L)

    # store data history
    hxTrue = np.hstack((hxTrue, xTrue))
    hxOdom = np.hstack((hxOdom, xOdom))
    hxEst = np.hstack((hxEst, xEst))
    err = xEst - xTrue
    err[2, 0] = angle_wrap(err[2, 0])
    hxError = np.hstack((hxError, err))
    hxSTD = np.hstack((hxSTD, xSTD))

    # plot every 20 updates
    if k % 20 == 0:
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

        ax1.cla()

        # Plot true landmark and trajectory
        ax1.plot(Map[0, :], Map[1, :], "*k")
        ax1.plot(hxTrue[0, :], hxTrue[1, :], "-k", label="True")
     #   ax1.plot([xTrue[0], Map[0, iFeature]], [xTrue[1], Map[1, iFeature]], "-b")

        # Plot odometry trajectory
        ax1.plot(hxOdom[0, :], hxOdom[1, :], "-g", label="Odom")

        # Plot estimated trajectory and current particles
        ax1.plot(hxEst[0, :], hxEst[1, :], "-r", label="Part. Filt.")
        ax1.plot(xEst[0], xEst[1], ".r")
        ax1.scatter(xParticles[0, :], xParticles[1, :], s=L*100)
        for i in range(nParticles):
            ax1.arrow(xParticles[0, i], xParticles[1, i], 5*np.cos(xParticles[2, i]+np.pi/2), 5*np.sin(xParticles[2, i]+np.pi/2), color = 'orange')

        ax1.axis([-60, 60, -60, 60])
        ax1.grid(True)
        ax1.legend()

        # plot errors curves
        ax3.plot(hxError[0, :], 'b')
        ax3.plot( 3.0 * hxSTD[0, :], 'r')
        ax3.plot(- 3.0 * hxSTD[0, :], 'r')
        ax3.grid(True)
        ax3.set_ylabel('x')
        ax3.set_title('Real error (blue) and particles covariances (red)')


        ax4.plot(hxError[1, :], 'b')
        ax4.plot( 3.0 * hxSTD[1, :], 'r')
        ax4.plot(- 3.0 * hxSTD[1, :], 'r')
        ax4.grid(True)
        ax4.set_ylabel('y')

        ax5.plot(hxError[2, :], 'b')
        ax5.plot( 3.0 * hxSTD[2, :], 'r')
        ax5.plot(- 3.0 * hxSTD[2, :], 'r')
        ax5.grid(True)
        ax5.set_ylabel(r"$\theta$")

        plt.pause(0.01)


plt.savefig('ParticleFilter_Localization.png')
tErrors = np.sqrt(np.square(hxError[0, :]) + np.square(hxError[1, :]))
print("Mean (var) translation error : {:e} ({:e})".format(np.mean(tErrors), np.var(tErrors)))
print("Press Q in figure to finish...")
plt.show()
