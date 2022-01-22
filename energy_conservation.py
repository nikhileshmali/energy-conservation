from __future__ import division
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import time


start_time = time.time()

num_steps = 100
dt = 3e9         # 100 years in seconds
N = 100
R = 1e15
ran = 1          # range (multiple of R) of axes in plots
G = 6.674e-11
e = 0.1 * R     
rho = 1408       # density of particles
M_sun = 1.99e30

def positions(n, rad):
    """Returns randomised initial positions of 'n' stars in a sphere of radius 'rad'"""
    positions = np.zeros((n, 3))
    r = np.random.uniform(low=-1.0, high=1.0, size=(n)) * rad
    phi = np.random.uniform(low=0, high=2*np.pi, size=(n))
    theta = np.random.uniform(low=0, high=np.pi, size=(n))
    #r = np.random.normal(scale = 0.5, size=(n)) * rad
    for i in range(n):
        x = r[i] * np.sin(theta[i]) * np.cos(phi[i])
        y = r[i] * np.sin(theta[i]) * np.sin(phi[i])
        z = r[i] * np.cos(theta[i])
        positions[i] = np.array((x, y, z)) 
        positions = positions.astype(float)
    return positions

def force(m, mi, separation_vector):   
    """Returns the gravitational force on particle m due to mi (separation_vector must point from mi TO m)"""
    if np.linalg.norm(separation_vector) == 0.0:
        force = 0.0
    else:
        force = (- G * m * mi) * separation_vector /(((np.linalg.norm(separation_vector)**2)+ e**2)**(1.5))
        force = force.astype(float)
    return force
    
def Potential_Energy(m, mi, separation_vector):
    """Returns the gravitational potential energy due to a pair of masses m, mi"""
    if np.linalg.norm(separation_vector) == 0.0:
        PE = 0.0
    else:
        PE = (- G * m * mi) / (np.linalg.norm(separation_vector))
    return PE

def separation_vectors(positions, n):  # 0 vectors are included
    """returns array of the separation vectors of the n particles"""
    separation_vectors = np.zeros((3))
    for i in range(n):
        for j in range(n):
            vect = positions[i] - positions[j] 
            separation_vectors = np.vstack([separation_vectors, vect])   
    separation_vectors = np.delete(separation_vectors, 0, 0) 
    separation_vectors = separation_vectors.astype(float)       
    return separation_vectors

def CoM(pos, m):
    """Returns the CoM coordinate of a system of particles with positions'pos' (array) and masses'm' (array)"""
    CoM = np.zeros(3)
    M_tot = m.sum()
    for i in range(3):
        CoM[i] = (pos[:,i] * m[:,0]).sum() / M_tot
    CoM = CoM.astype(float)
    return CoM

def velocities(masses, rad, n):
    """Returns an array of randomly-generated velocities"""
    masses = np.sum(masses[:,0])
    v_esc = np.sqrt(2*G*masses/rad)
    velocities = np.random.uniform(low=0.0, high=1.0, size=(n, 3)) * v_esc
    return velocities
    
def masses(MIN, MAX, m, n):
    """Returns an array of randomly generated masses within a range MIN*m, and MAX*m"""
    masses = np.random.uniform(low=MIN, high=MAX, size=(n,3)) * m
    return masses
    
M = masses(0.5, 1.5, 2e30, N)
velocities = np.zeros((N, 3))  
positions = positions(N, R)
COM = CoM(positions, M) 
positions = positions - COM
sep_vect = separation_vectors(positions, N)
forces = np.zeros(3)
KE, PE, TE, PCTE = np.zeros(num_steps), np.zeros(num_steps),np.zeros(num_steps), np.zeros(num_steps)

#### INITIAL STAR DISTRIBUTION ####
xs, ys, zs = positions[:,0], positions[:,1], positions[:,2]  
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(2,3,1, projection='3d')
ax.scatter(xs, ys, zs, s=5, c='black', marker = 'o')
plt.title("Initial")
ax.set_xlim3d(-ran*R, ran*R)
ax.set_ylim3d(-ran*R, ran*R)
ax.set_zlim3d(-ran*R, ran*R)

for i in range(num_steps):

    #### UPDATING VELOCITIES ####
    velocities = velocities + forces * dt / M
    
    #### UPDATING POSITIONS ####
    positions = positions + velocities * dt
    COM = CoM(positions, M) 
    positions = positions - COM

    ##### UPDATING SEPARATION VECTORS ####
    sep_vect = separation_vectors(positions, N)
      
    #### UPDATING FORCES ####
    forces = np.zeros(3)
    for j in range(N):
        fvect = np.zeros(3)
        for k in range(N):
            fvect = fvect + force(M[j], M[k], sep_vect[j*N + k])
        forces = np.vstack([forces, fvect])
    forces = np.delete(forces, 0, 0) 
    
    #### UPDATING KE ####
    speeds = np.zeros(N)
    for j in range(N):
        speeds[j] = np.linalg.norm(velocities[j])
    KEi = 0.5 * M[:,0] * speeds ** 2
    KE[i] = np.sum(KEi)
    
    #### UPDATING PE ####
    history = np.zeros(N)
    for j in range(N):      
        PEj = np.zeros(1)
        for k in range(j, N):
            PEj = PEj + Potential_Energy(M[j,0], M[k,0], sep_vect[j*N + k])
        history[j] = PEj
    PE[i] = history.sum() 
     
    #### UPDATING TE ####
    TE[i] = PE[i] + KE[i]
    
    #### UPDATING PCTE ####
    PCTE[i] = ((TE[i] - TE[0]) / TE[0]) * 100
    
    #### PROGRESS ESTIMATE ####
    percent = (float(i+1)/float(num_steps))*100
    estimate = ((time.time() - start_time) * num_steps/(i+1))/(60*60)
    print("Progress: %s / %s (%f percent). Estimate: %f hours " %(i+1, num_steps, percent, estimate))

#### FINAL STAR DISTRIBUTION ####
xs, ys, zs = positions[:,0], positions[:,1], positions[:,2]
ax = fig.add_subplot(2,3,2, projection='3d')
ax.scatter(xs, ys, zs, s=5, c='red', marker = 'o')
plt.title("Final")
ax.set_xlim3d(-ran*R, ran*R)
ax.set_ylim3d(-ran*R, ran*R)
ax.set_zlim3d(-ran*R, ran*R)

#### ENERGY PLOTS ####
time_vals = np.linspace(0,num_steps, num_steps, endpoint=False)
TE = PE + KE
ax = fig.add_subplot(2,3,3)
ax.plot(time_vals, TE, label = "TE")
ax.plot(time_vals, PE, label = "PE")
ax.plot(time_vals, KE, label = "KE")
plt.title("Energy Plots")
ax.legend()

#### PERCENTAGE CHANGE IN ENERGY PLOT ####
ax = fig.add_subplot(2,3,4)
ax.plot(time_vals, PCTE, label = "Percentage Energy Change")
ax.legend()

plt.show()    

print("My program took %f seconds to run" %(time.time() - start_time))
