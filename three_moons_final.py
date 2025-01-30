import numpy as np
import sklearn as sk
import graphlearning as gl
import scipy.sparse as sparse
import LU_order_statistic
import os
import scipy as sp
from scipy.sparse.linalg import eigs

def knn_costum(data, k, kernel='gaussian', eta=None, symmetrize=True, metric='raw', similarity='euclidean', knn_data=None):
    #construction of knn-weight matrix.
    #This method is copied from the Graph Learning Library
    #@software{graphlearning,
    #    author       = {Jeff Calder},
    #    title        = {GraphLearning Python Package},
    #    month        = jan,
    #    year         = 2022,
    #    publisher    = {Zenodo},
    #    doi          = {10.5281/zenodo.5850940},
    #    url          = {https://doi.org/10.5281/zenodo.5850940}
    #}

    #Self is counted in knn data, so add one
    k += 1

    #If knn_data provided
    if knn_data is not None:
        knn_ind, knn_dist = knn_data

    #If data is a string, then load knn from a stored dataset
    elif type(data) is str:
        knn_ind, knn_dist = gl.weightmatrix.load_knn_data(data, metric=metric)

    #Else we have to run a knnsearch
    else:
        knn_ind, knn_dist = gl.weightmatrix.knnsearch(data, k, similarity=similarity, method="kdtree")

    #Restrict to k nearest neighbors
    n = knn_ind.shape[0]
    k = np.minimum(knn_ind.shape[1],k)
    knn_ind = knn_ind[:,:k]
    knn_dist = knn_dist[:,:k]

    #If eta is None, use kernel keyword
    if eta is None:
        if kernel == 'gaussian':
            D = knn_dist*knn_dist
            eps = D[:,k-1]
            weights = np.exp(-4*D/eps[:,None])

        elif kernel == 'symgaussian':
            eps = knn_dist[:,7]
            weights = np.exp(-knn_dist * knn_dist / eps[:,None] / eps[knn_ind])
        elif kernel == 'distance':
            weights = knn_dist
        elif kernel == 'singular':
            weights = knn_dist
            weights[knn_dist==0] = 1
            weights = 1/weights
        else:
            print('Invalid choice of kernel: ' + kernel)
            return

    #Else use user-defined eta
    else:
        D = knn_dist*knn_dist
        eps = D[:,k-1]
        weights = eta(D/eps)

    #Flatten knn data and weights
    knn_ind = knn_ind.flatten()
    weights = weights.flatten()

    #Self indices
    self_ind = np.ones((n,k))*np.arange(n)[:,None]
    self_ind = self_ind.flatten()



    #Construct sparse matrix and convert to Compressed Sparse Row (CSR) format
    W = sparse.coo_matrix((weights, (self_ind, knn_ind)),shape=(n,n)).tocsr()

    if symmetrize:
        if kernel in ['distance','uniform','singular']:
            W = gl.utils.sparse_max(W, W.transpose())
        elif kernel == 'symgaussian':
            W = W + W.T.multiply(W.T > W) - W.multiply(W.T > W)
        else:
            W = (W + W.transpose())/2;   
    W.setdiag(0)
    return W



def diffuse(chi, t, vals, vec):
    #diffusion via appoximated heat kernel
    #t: diffusion time
    #vals: eigenvalues
    #vecs: eigenvectors
    diffused = np.zeros((N,m))
    for i in range(K):
        diffused += np.exp(-t * vals[i]) * vec[i].reshape(N,1).dot(vec[i].dot(chi).reshape(1,m))
    return(diffused)


def diffuse_W(chi, W):
    return W.dot(chi)


def clustering_toonehot(clustering):
    #one hot encoding of clustering
    chi = np.zeros((N,m))
    for i in range(N):
        for j in range(m):
            chi[i,j] = j==clustering[i]
    return chi

def random_phase_init(m,N):
    return np.random.randint(0,m, size=N)


def prepare(chi, diffusion_method, energy):
    #compute diffusion and energy
    chi = chi.astype(np.float32)
    u = diffusion_method(chi)
    energy_before = energy
    energy = 0
    for i in range(m):
        for j in range(N):
            energy += u[j,i]*(1 - chi[j,i])

    relative_energy_change = (energy_before - energy)/energy
    return u, relative_energy_change, energy



def run_MBO(chi, diffusion_function, L=None, U=None):
    #function to run the MBO scheme
    #chi: initial clustering
    #diffusion_function: function that should be used to diffuse the labels
    #L: lower voluem constraints
    #U: upper volume constraints
    #W: weight matrix

    #initial median guess
    median_0 = np.array(m*[1/m])

    #thresholding energy (used as stopping criteria)
    energy= np.inf
    relative_energy_change = np.inf

    #values of the iteration with minimal energy (only used if temperature > 0)
    min_energy = np.inf
    min_chi = None
    min_phase = None

    #mask describing which lables are known
    mask = np.ones(N, dtype=bool)
    mask[fidelity_set_ind] = False

    #clustering
    phase = np.argmax(chi, axis=1)

    count = 0
    #run until stopping criteria (depending on whether temperature is used) is reached
    while((relative_energy_change > 0.0001 and temperature == 0) or (temperature > 0 and count < 50)):
        count+=1

        #diffuse and compute energy
        u, relative_energy_change, energy = prepare(chi, diffusion_function, energy)
    
        #if this iteration has lowest energy, save the clustering
        if(energy < min_energy):
            min_chi = chi.copy()
            min_phase = phase.copy()
            min_energy = energy

        #add temperature if desired
        if(temperature > 0):
            u += np.random.normal(0, temperature, size=u.shape)

        #threshold by calculating the order statistic
        if(L!=None):
            median, phase, P = LU_order_statistic.fit_median(m,L,U,u,median_0)
        else:
            phase = np.argmax(u, axis=1)

        #compute one hot encoding
        chi = clustering_toonehot(phase)
    return min_chi, min_phase


def init_chi(fidelity_set_ind, y, diffusion_function=None, W=None, mode="diffusion", P = None):
    #initialization of clusters with different methods (defined by mode):
    #"bellmand_ford": initialization as voronoi tesselation via the bellman-ford algorithm,for that W needs to be defined
    #"voronoi": initialization as voronoi tesselation via the dijkstra algorithm, for that W needs to be defined
    #"random": random initialization
    #"laguerre": laguerre tesselation initialization, i.e. voronoi with volume constraints, for that W needs to be defined
    #"diffusion": initialization with a diffusion matrix similar to voronoi, for that diffusion_function needs to be defined
    #"diffusion_volume": initialization by diffusion matrix and volume constraints similar to laguerre, again diffusion_function necessary
    if(mode=="bellman_ford"):
        voronoi, phase = bellman_ford_voronoi_initialization(N, fidelity_set_ind, y, W)


    if(mode=="random"):
        phase = random_phase_init(m,N)


    if(mode=="voronoi"):
        q_W = W.copy() + sparse.diags(np.ones(N))
        #q_W.data = -np.log(q_W.data)
        q_graph = gl.graph(q_W)
        distance, phase = q_graph.dijkstra(fidelity_set_ind,return_cp=True)

        phase = y[phase]

    if(mode=="laguerre"):
        q_W = W.copy() + sparse.diags(np.ones(N))
        q_W.data = - np.log(q_W.data)
        q_graph = gl.graph(q_W)
        distances = []
        for index in range(m):
            indices = fidelity_set_ind[y[fidelity_set_ind] == index]
            distance = q_graph.dijkstra(indices,return_cp=False)
            distances.append(distance)
        median_0 = m*[1/m]
        distances = -np.array(distances).transpose()
        distances[distances == -np.inf] = - 10000.
        median, phase, P = LU_order_statistic.fit_median(m,P,P,distances,median_0)
        
        
    if(mode=="diffusion"):
        delta = np.zeros((N,m))
        for i in range(len(fidelity_set_ind)):
            delta[fidelity_set_ind[i], y[fidelity_set_ind[i]]] = N 

        diffused = diffusion_function(delta)
        median_0 = m*[1/m]

        phase = np.argmax(diffused, axis=1)

    if(mode=="diffusion_volume"):
        delta = np.zeros((N,m))
        for i in range(len(fidelity_set_ind)):
            delta[fidelity_set_ind[i], y[fidelity_set_ind[i]]] = N 

        diffused = diffusion_function(delta)
        median_0 = m*[1/m]
        median, phase, P = LU_order_statistic.fit_median(m,P,P,diffused,median_0)


    phase[fidelity_set_ind] = y[fidelity_set_ind]

    chi = clustering_toonehot(phase)
    return chi


def make_3_blops(n_samples=1000, noise=None):
    #generation of samples of three normal distributions
    n_samples_out = n_samples // 3
    n_samples_in =  n_samples_out
    n_samples_out2 = n_samples - n_samples_out - n_samples_in

    outer_circ_x = np.zeros(n_samples_out)
    outer_circ_y = np.zeros(n_samples_out)
    inner_circ_x = np.ones(n_samples_in)
    inner_circ_y = np.zeros(n_samples_in)
    outer_circ_x2 = 0.5*np.ones(n_samples_out2)
    outer_circ_y2 = np.sqrt(3)/2*np.ones(n_samples_out2)

    X = np.vstack(
        [np.append(np.append(outer_circ_x, inner_circ_x), outer_circ_x2), np.append(np.append(outer_circ_y, inner_circ_y), outer_circ_y2)]
    ).T
    y = np.hstack(
        [np.zeros(n_samples_out, dtype=np.intp), np.ones(n_samples_in, dtype=np.intp), 2*np.ones(n_samples_out2, dtype=np.intp)]
    )
    extra_dims = 0
    X = np.pad(X, ((0,0),(0,extra_dims)), 'constant')
    
     
    generator = sk.utils.check_random_state(None)

    X += generator.normal(scale=noise, size=X.shape)
    return X,y


def bellman_ford_voronoi_initialization(N, fidelity, y, W):
    #voronoi tesselation via bellman-ford algorithm
    active = np.zeros(N)
    fixedLabels=fidelity
    labels = np.zeros(N) - 1
    labels[fixedLabels] = y[fixedLabels]
    active[fixedLabels] = True
    voronoiDistances= np.zeros(N)
    for i in range(N):

        if(not active[i]):
            voronoiDistances[i]=np.inf
        

    done=False
    while( not done):
        done=1
        for i in range(N):
            if(active[i]):
                done=0
                for j in range(W.indptr[i], W.indptr[i+1]):
                    index = W.indices[j]
                    if(W.data[j] != 0):
                        dist= W.data[j]
                        current=voronoiDistances[i]
                        if(current+dist<voronoiDistances[index]):
                            voronoiDistances[index]=current+dist
                            active[index]=True
                            labels[index]=labels[i]
                active[i]=False    

    return voronoiDistances, labels




def make_3_moons(n_samples=1000, noise=None):
    #gemerating samples from the three moon dataset
    n_samples_out = n_samples // 3
    n_samples_in =  n_samples_out
    n_samples_out2 = n_samples - n_samples_out - n_samples_in

    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1.5 - 1.5*np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 0.4 - 1.5*np.sin(np.linspace(0, np.pi, n_samples_in)) 
    outer_circ_x2 = np.cos(np.linspace(0, np.pi, n_samples_out2)) + 3.
    outer_circ_y2 = np.sin(np.linspace(0, np.pi, n_samples_out2))

    X = np.vstack(
        [np.append(np.append(outer_circ_x, inner_circ_x), outer_circ_x2), np.append(np.append(outer_circ_y, inner_circ_y), outer_circ_y2)]
    ).T
    y = np.hstack(
        [np.zeros(n_samples_out, dtype=np.intp), np.ones(n_samples_in, dtype=np.intp), 2*np.ones(n_samples_out2, dtype=np.intp)]
    )
    extra_dims = 98
    X = np.pad(X, ((0,0),(0,extra_dims)), 'constant')
    
     
    generator = sk.utils.check_random_state(None)

    X += generator.normal(scale=noise, size=X.shape)
    return X,y





neighbours = 10 #number of neighbours for the knn graph
N=1500 #numbers of node in the sample
m= 3 #number of phases
K =int( 0.5*np.log(N)) #number of eigenvalues for the approximated heat kernel
t = 1.0 #diffusion time 

temperature = 0.0 #temperature as in Auction Dynamics by Jacobs et al. (2018)
points_per_class = 5 #numbers of known labels in every class


#if the weight matrix was already computed before, load the weight matrix
if(os.path.isfile("data/weights_three_moons.npz") and os.path.isfile("data/labels_three_moons.csv")):
    #load (precomputed) weight matrix
    W = sp.sparse.load_npz("data/weights_three_moons.npz")
    #load labels
    y = np.loadtxt("data/labels_three_moons.csv", delimiter=",", dtype=int)
else:  
    #generate 3 moon dataset
    x,y = make_3_moons(N, noise=0.14)

    #optioanl one can also load a dataset sampled from the normal distributions.
    #x,y = make_3_blops(N, noise=0.16)
    

    x = np.array(x)
    y = np.array(y)



    #construct weightmatrix
    W = knn_costum(x, neighbours, metric='euclidean',kernel='gaussian', symmetrize=True)

    #save weightmatrix and labels
    sp.sparse.save_npz('data/weights_three_moons.npz', W)
    np.savetxt("data/labels_three_moons.csv", y, fmt='%i', delimiter=",")



#compute diagonal elements of weightmatrix
D = np.sum(W, axis = 1).A1

#compute the normalized weight matrix D^{-1}W
DW = sp.sparse.diags(1/D,0).dot(W)

#Graph Laplace matrix
Delta = sp.sparse.identity(N) - DW

#first K eigenvalues and eigenvectors of Graph Laplace
vals, vec = eigs(DW, K)
vec = vec.transpose().real
vals = 1 - vals.real

#compute exact volumes of the classes
P=[]
for i in range(m):
    P.append(np.sum(y==i))

#calculate the different diffusion matrices:

#1. taylor expansion of exponential matrix
A_3 = 1/(1+t+t*t/2)*(sp.sparse.identity(N) +t* DW + t*t/2*DW.T.dot(DW))

#2. squared weight matrix
W2 = DW.transpose().dot(DW)

#3. fourth power of weight matrix
W_W2 = W2.dot(W2)

# 4. squared weight matrix minus fraction of identity
A_minus_eig = -0.1*sp.sparse.identity(N) + W2

#define diffusion of labels by the different kernels
#diffusion by the rank K approximation of the heat kernel
diffusion_functioneigen = lambda chi: diffuse(chi, t, vals, vec)

#diffusion by the previously defined matrices
diffusion_function3 = lambda chi: diffuse_W(chi, A_3)
diffusion_functionW2 = lambda chi: diffuse_W(chi, W2)
diffusion_functionW_W2 = lambda chi: diffuse_W(chi, W_W2)
diffusion_functionA4 = lambda chi: diffuse_W(chi, A_minus_eig)

#gererating the results of the paper
#variables for storing the results
s1  = 0
s2  = 0
s3  = 0
s4  = 0
s5  = 0


s1s  = []
s2s  = []
s3s  = []
s4s  = []
s5s  = []


#run 100 trials
for j in range(100):
    #sample random labels that are set as known
    fidelity_set_ind= gl.trainsets.generate(y, rate=points_per_class)
    non_y = np.delete(np.array(range(N)),fidelity_set_ind)

    #initialization with laguerre tesselation
    chi_init_laguerre = init_chi(fidelity_set_ind, y, diffusion_function=diffusion_functioneigen, mode="laguerre", W=W, P=P )
    
    #initialization with approximated heat kernel and volume constraints
    chi_init_diffuse = init_chi(fidelity_set_ind, y, diffusion_function=diffusion_functioneigen, mode="diffusion_volume", W=W, P=P )

    #run MBO with different kernels:
    chi, phase= run_MBO(chi_init_laguerre,diffusion_functionA4,P,P)

    s1s.append(gl.ssl.ssl_accuracy(phase, y, fidelity_set_ind))
    s1 +=  gl.ssl.ssl_accuracy(phase, y, fidelity_set_ind)  
    print(" W^2- r I accuracy with fid after ", j+1, " iterations : ", s1/(j+1))
    
    chi, phase = run_MBO(chi_init_laguerre,diffusion_function3,P,P)

    s2 += gl.ssl.ssl_accuracy(phase, y, fidelity_set_ind)  
    s2s.append(gl.ssl.ssl_accuracy(phase, y, fidelity_set_ind))
    print(" 3rd Taylor Polynom accuracy with fid after ", j+1, " iterations : ", s2/(j+1))

    chi, phase = run_MBO(chi_init_laguerre,diffusion_functionW2,P,P)


    s3 +=  gl.ssl.ssl_accuracy(phase, y, fidelity_set_ind) 
    s3s.append(gl.ssl.ssl_accuracy(phase, y, fidelity_set_ind)) 
    print(" W^2 accuracy with fid after ", j+1, " iterations : ", s3/(j+1))

    chi, phase = run_MBO(chi_init_diffuse,diffusion_functioneigen,P,P)

    s4 += gl.ssl.ssl_accuracy(phase, y, fidelity_set_ind)  
    s4s.append(gl.ssl.ssl_accuracy(phase, y, fidelity_set_ind))
    print(" rank-k approx. of heat accuracy with fid after ", j+1, " iterations : ", s4/(j+1))

    chi, phase = run_MBO(chi_init_laguerre,diffusion_functionW_W2,P,P)

    s5 +=  gl.ssl.ssl_accuracy(phase, y, fidelity_set_ind)  
    s5s.append(gl.ssl.ssl_accuracy(phase, y, fidelity_set_ind))
    print(" W^4 accuracy with fid after ", j+1, " iterations : ", s5/(j+1))


#computing mean and standard deviation for the computed results



means1 = np.mean(s1s)
stds1 = np.std(s1s)
s1s.append(means1)
s1s.append(stds1)
s1s = np.reshape(s1s,(len(s1s), 1))


means2 = np.mean(s2s)
stds2 = np.std(s2s)
s2s.append(means2)
s2s.append(stds2)
s2s = np.reshape(s2s,(len(s2s), 1))


means3 = np.mean(s3s)
stds3 = np.std(s3s)
s3s.append(means3)
s3s.append(stds3)
s3s = np.reshape(s3s,(len(s3s), 1))


means4 = np.mean(s4s)
stds4 = np.std(s4s)
s4s.append(means4)
s4s.append(stds4)
s4s = np.reshape(s4s,(len(s4s), 1))


means5 = np.mean(s5s)
stds5 = np.std(s5s)
s5s.append(means5)
s5s.append(stds5)
s5s = np.reshape(s5s,(len(s5s), 1))


#saving results
np.savetxt("results/three_moons_"+str(temperature)+"_"+str(points_per_class)+"w^2-epsI.csv", s1s, delimiter=",")
np.savetxt("results/three_moons_"+str(temperature)+"_"+str(points_per_class)+"I + h W + h^2 2 W^2.csv", s2s, delimiter=",")
np.savetxt("results/three_moons_"+str(temperature)+"_"+str(points_per_class)+"W^2.csv", s3s, delimiter=",")
np.savetxt("results/three_moons_"+str(temperature)+"_"+str(points_per_class)+"rank K.csv", s4s, delimiter=",")
np.savetxt("results/three_moons_"+str(temperature)+"_"+str(points_per_class)+"W^4.csv", s5s, delimiter=",")



