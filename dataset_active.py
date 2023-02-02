import numpy as np
import string, random, os, time
from scipy import interpolate
import subprocess

# from hdf5storage import savemat
from hdf5storage import loadmat
import h5py

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

# from tqdm.notebook import trange
from tqdm.auto import tqdm, trange
import sobol_seq

from infras.misc import *

# keys used to retrieve the query results
def get_random_alphanumeric_string(length):
    letters_and_digits = string.ascii_letters + string.digits
    result_str = ''.join((random.choice(letters_and_digits) for i in range(length)))
    return result_str

def dump_to_h5f(D, h5fname, comp=None):
 
    try:
        hf = h5py.File(h5fname, 'w')

        # meta info
        hf.create_dataset('Ntrain', data=np.array(D['Ntrain']))
        hf.create_dataset('Ntest', data=np.array(D['Ntest']))
        hf.create_dataset('seed', data=D['trial'])
        hf.create_dataset('domain', data=D['domain'])
        hf.create_dataset('Nf', data=D['Nf'])

        group_X_train = hf.create_group('X_train')
        group_y_train = hf.create_group('y_train')
        group_y_train_ground = hf.create_group('y_train_ground')

        group_X_test = hf.create_group('X_test')
        group_y_test = hf.create_group('y_test')
        group_y_test_ground = hf.create_group('y_test_ground')

        for m in range(D['Nf']):
            key = 'fidelity_' + str(m)
            group_X_train.create_dataset(key, data=D['X_train_list'][m], compression=comp)
            group_y_train.create_dataset(key, data=D['y_train_list'][m], compression=comp)
            group_y_train_ground.create_dataset(key, data=D['y_train_ground_list'][m], compression=comp)

            group_X_test.create_dataset(key, data=D['X_test_list'][m], compression=comp)
            group_y_test.create_dataset(key, data=D['y_test_list'][m], compression=comp)
            group_y_test_ground.create_dataset(key, data=D['y_test_ground_list'][m], compression=comp)
        #
    except:
        print('ERROR occurs when WRITE the h5 object...')
    finally:
        hf.close()


def load_from_h5f(h5fname):
    D = {}
    try:
        hf = h5py.File(h5fname, 'r')

        M = np.array(hf.get('Nf')).tolist()
        domain = np.array(hf.get('domain')).tolist().decode("utf-8")
        Ntrain = np.array(hf.get('Ntrain'))
        Ntest = np.array(hf.get('Ntest'))
        
        D['X_train_list'] = []
        D['y_train_list'] = []
        D['y_train_ground_list'] = []
        D['Nf'] = M
        D['Ntrain'] = Ntrain
        D['Ntest'] = Ntest
        D['domain'] = domain

        D['X_test_list'] = []
        D['y_test_list'] = []
        D['y_test_ground_list'] = []
        
        for m in range(M):

            D['X_train_list'].append(np.array(hf.get('Xtr/f'+str(m))))
            D['y_train_list'].append(np.array(hf.get('ytr/f'+str(m))))
            D['y_train_ground_list'].append(np.array(hf.get('ytr_ground/f'+str(m))))
            
            #print(np.array(hf.get('ytr_ground/f'+str(m))).shape)
                                
            D['X_test_list'].append(np.array(hf.get('Xte/f'+str(m))))
            D['y_test_list'].append(np.array(hf.get('yte/f'+str(m))))
            D['y_test_ground_list'].append(np.array(hf.get('yte_ground/f'+str(m))))
            
#             cprint('b', np.array(hf.get('Xtr/f'+str(m))).shape)
#             cprint('b', np.array(hf.get('ytr/f'+str(m))).shape)
#             cprint('r', np.array(hf.get('Xte/f'+str(m))).shape)
#             cprint('r', np.array(hf.get('yte/f'+str(m))).shape)
        
    except:
        print('ERROR occurs when READ the h5 object')
    finally:
        hf.close()
        return D
       
class Poisson2:
    def __init__(self,):
        self.M = 2
        self.dim = 5
        
        self.fidelity_list = [16,32,128]

        self.bounds = ((0.1, 0.9),(0.1, 0.9),(0.1, 0.9),(0.1, 0.9),(0.1, 0.9))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]
        
    def query(self, X, m):
        y_list = []
        N = X.shape[0]
        for n in range(N):
            y = self.single_query(X[n], m)
            y_list.append(y)
        #
        
        Y = np.array(y_list)
        re_Y = np.reshape(Y, [N,-1])
        
        #print(re_Y.shape)
        
        return re_Y 

    def single_query(self, X, m, interp=False):
        
        X = np.squeeze(X)
        fidelity = self.fidelity_list[m]
        u = self._poisson_solver(self.fidelity_list[m], X[0], X[1], X[2], X[3], X[4])
            
        return u
        
    def interpolate(self, Y, m):
        # low to high
        N = Y.shape[0]
        
        lf = self.fidelity_list[m]
        hf = self.fidelity_list[-1]
        
        x_lf = np.linspace(0,1,lf)
        y_lf = np.linspace(0,1,lf)
        x_hf = np.linspace(0,1,hf)
        y_hf = np.linspace(0,1,hf)
        
        Y_interp_list = []
        
        for n in range(N):
            u = Y[n,:].reshape(lf, lf)
            interp_fn = interpolate.interp2d(x_lf, y_lf, u, kind='linear')
            u_hf = interp_fn(x_hf, y_hf)
            
            Y_interp_list.append(u_hf)
        # 
        
        Y_interp = np.array(Y_interp_list)
        re_Y_interp = np.reshape(Y_interp, [N,-1])
        
        return re_Y_interp

    def ground(self, X):
        y_list = []
        N = X.shape[0]
        for n in range(N):
            y = self.single_ground(X[n])
            y_list.append(y)
        #
        
        Y = np.array(y_list)
        re_Y = np.reshape(Y, [N,-1])
        
        #print(re_Y.shape)
        
        return re_Y 

    def single_ground(self, X):
        X = np.squeeze(X)
        fidelity = self.fidelity_list[-1]
        u_ground = self._poisson_solver(fidelity, X[0], X[1], X[2], X[3], X[4])
        
        return u_ground

    def _poisson_solver(self, fidelity,u_0_x,u_1_x,u_y_0,u_y_1,u_dirac):
        x = np.linspace(0,1,fidelity)
        dx = x[1]-x[0]
        y = np.linspace(0,1,fidelity)
        u = np.zeros((fidelity+2,fidelity+2)) # Initial u used to create the b-vector in Ax = b
        # BC's and dirac delta
        u[0,:] = u_0_x
        u[-1,:] = u_1_x
        u[:,0] = u_y_0
        u[:,-1] = u_y_1
        if fidelity%2 == 0:
            u[int((fidelity+2)/2-1):int((fidelity+2)/2+1),int((fidelity+2)/2-1):int((fidelity+2)/2)+1] = u_dirac
        else:
            u[int((fidelity+1)/2),int((fidelity+1)/2)] = u_dirac

        # 5-point scheme
        A = np.zeros((fidelity**2,fidelity**2))
        for i in range(fidelity**2):
            A[i,i] = 4
            if i < fidelity**2-1:
                if i%fidelity != fidelity-1:
                    A[i,i+1] = -1
                if i%fidelity != 0 & i-1 >= 0:
                    A[i,i-1] = -1
            if i < fidelity**2-fidelity:
                A[i,i+fidelity] = -1
                if i-fidelity >= 0:
                    A[i,i-fidelity] = -1

        # Boundry conditions
        g = np.zeros((fidelity,fidelity))
        for i in range(1,fidelity+1):
            for j in range(1,fidelity+1):
                g[i-1,j-1] = u[i-1,j]+u[i+1,j]+u[i,j-1]+u[i,j+1]

        b = dx**2*g.flatten()
        #x = np.linalg.solve(A,b)
        #u = x.reshape(fidelity,fidelity)

        # Sparse solver
        A_s = csc_matrix(A, dtype=float) # s for sparse
        b_s = csc_matrix(b, dtype=float)
        x_s = spsolve(A_s,b_s.T)
        u_s = x_s.reshape(fidelity,fidelity)

        return u_s
    
class Poisson3:
    def __init__(self,):
        self.M = 3
        self.dim = 5
        
        self.fidelity_list = [16,32,64,128]
#         self.fidelity_list = [16,32,64,72]

        self.bounds = ((0.1, 0.9),(0.1, 0.9),(0.1, 0.9),(0.1, 0.9),(0.1, 0.9))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]

    def query(self, X, m):
        y_list = []
        N = X.shape[0]
        for n in range(N):
            y = self.single_query(X[n], m)
            y_list.append(y)
        #
        
        Y = np.array(y_list)
        re_Y = np.reshape(Y, [N,-1])
        
        #print(re_Y.shape)
        
        return re_Y 
        
    def single_query(self, X, m, interp=False):
        
        X = np.squeeze(X)
        u = self._poisson_solver(self.fidelity_list[m], X[0], X[1], X[2], X[3], X[4])
            
        return u
        
    def interpolate(self, Y, m):
        # low to high
        N = Y.shape[0]
        
        lf = self.fidelity_list[m]
        hf = self.fidelity_list[-1]
        
        x_lf = np.linspace(0,1,lf)
        y_lf = np.linspace(0,1,lf)
        x_hf = np.linspace(0,1,hf)
        y_hf = np.linspace(0,1,hf)
        
        Y_interp_list = []
        
        for n in range(N):
            u = Y[n,:].reshape(lf, lf)
            interp_fn = interpolate.interp2d(x_lf, y_lf, u, kind='linear')
            u_hf = interp_fn(x_hf, y_hf)
            
            Y_interp_list.append(u_hf)
        # 
        
        Y_interp = np.array(Y_interp_list)
        re_Y_interp = np.reshape(Y_interp, [N,-1])
        
        return re_Y_interp
    
    def inv_interpolate(self, Y, m):
        # high to low
        N = Y.shape[0]
        
        lf = self.fidelity_list[-1]
        hf = self.fidelity_list[m]
        
        x_lf = np.linspace(0,1,lf)
        y_lf = np.linspace(0,1,lf)
        x_hf = np.linspace(0,1,hf)
        y_hf = np.linspace(0,1,hf)
        
        Y_interp_list = []
        
        for n in range(N):
            u = Y[n,:].reshape(lf, lf)
            interp_fn = interpolate.interp2d(x_lf, y_lf, u, kind='linear')
            u_hf = interp_fn(x_hf, y_hf)
            
            Y_interp_list.append(u_hf)
        # 
        
        Y_interp = np.array(Y_interp_list)
        re_Y_interp = np.reshape(Y_interp, [N,-1])
        
        return re_Y_interp
        
    def ground(self, X):
        y_list = []
        N = X.shape[0]
        for n in range(N):
            y = self.single_ground(X[n])
            y_list.append(y)
        #
        
        Y = np.array(y_list)
        re_Y = np.reshape(Y, [N,-1])
        
        #print(re_Y.shape)
        
        return re_Y 

    def single_ground(self, X):

        X = np.squeeze(X)
        u_ground = self._poisson_solver(self.fidelity_list[-1], X[0], X[1], X[2], X[3], X[4])
        
        return u_ground

    def _poisson_solver(self, fidelity,u_0_x,u_1_x,u_y_0,u_y_1,u_dirac):
        x = np.linspace(0,1,fidelity)
        dx = x[1]-x[0]
        y = np.linspace(0,1,fidelity)
        u = np.zeros((fidelity+2,fidelity+2)) # Initial u used to create the b-vector in Ax = b
        # BC's and dirac delta
        u[0,:] = u_0_x
        u[-1,:] = u_1_x
        u[:,0] = u_y_0
        u[:,-1] = u_y_1
        if fidelity%2 == 0:
            u[int((fidelity+2)/2-1):int((fidelity+2)/2+1),int((fidelity+2)/2-1):int((fidelity+2)/2)+1] = u_dirac
        else:
            u[int((fidelity+1)/2),int((fidelity+1)/2)] = u_dirac

        # 5-point scheme
        A = np.zeros((fidelity**2,fidelity**2))
        for i in range(fidelity**2):
            A[i,i] = 4
            if i < fidelity**2-1:
                if i%fidelity != fidelity-1:
                    A[i,i+1] = -1
                if i%fidelity != 0 & i-1 >= 0:
                    A[i,i-1] = -1
            if i < fidelity**2-fidelity:
                A[i,i+fidelity] = -1
                if i-fidelity >= 0:
                    A[i,i-fidelity] = -1

        # Boundry conditions
        g = np.zeros((fidelity,fidelity))
        for i in range(1,fidelity+1):
            for j in range(1,fidelity+1):
                g[i-1,j-1] = u[i-1,j]+u[i+1,j]+u[i,j-1]+u[i,j+1]

        b = dx**2*g.flatten()
        #x = np.linalg.solve(A,b)
        #u = x.reshape(fidelity,fidelity)

        # Sparse solver
        A_s = csc_matrix(A, dtype=float) # s for sparse
        b_s = csc_matrix(b, dtype=float)
        x_s = spsolve(A_s,b_s.T)
        u_s = x_s.reshape(fidelity,fidelity)

        return u_s
    
class Heat2:
    def __init__(self,):
        self.M = 2
        self.dim = 3
        
        self.fidelity_list = [16,32,100]

        self.bounds = ((0,1),(-1,0),(0.01,0.1))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]

    def query(self, X, m):
        y_list = []
        N = X.shape[0]
        for n in range(N):
            y = self.single_query(X[n], m)
            y_list.append(y)
        #
        
        Y = np.array(y_list)
        re_Y = np.reshape(Y, [N,-1])
        
        #print(re_Y.shape)
        
        return re_Y
        
    def single_query(self, X, m, interp=False):

        X = np.squeeze(X)
        fidelity = self.fidelity_list[m]
        u, x = self._heat_solver(fidelity, X[2], X[0], X[1])
            
        return u
        
    def interpolate(self, Y, m):
        # low to high
        N = Y.shape[0]
        
        lf = self.fidelity_list[m]
        hf = self.fidelity_list[-1]
        
        x_lf = np.linspace(0,1,lf)
        y_lf = np.linspace(0,1,lf)
        x_hf = np.linspace(0,1,hf)
        y_hf = np.linspace(0,1,hf)
        
        Y_interp_list = []
        
        for n in range(N):
            u = Y[n,:].reshape(lf, lf)
            interp_fn = interpolate.interp2d(x_lf, y_lf, u, kind='linear')
            u_hf = interp_fn(x_hf, y_hf)
            
            Y_interp_list.append(u_hf)
        # 
        
        Y_interp = np.array(Y_interp_list)
        re_Y_interp = np.reshape(Y_interp, [N,-1])
        
        return re_Y_interp

    def ground(self, X):
        y_list = []
        N = X.shape[0]
        for n in range(N):
            y = self.single_ground(X[n])
            y_list.append(y)
        #
        
        Y = np.array(y_list)
        re_Y = np.reshape(Y, [N,-1])
        
        #print(re_Y.shape)
        
        return re_Y 
        
    def single_ground(self, X):
        X = np.squeeze(X)
        u_ground, _ = self._heat_solver(self.fidelity_list[-1], X[2], X[0], X[1])
        
        return u_ground
    
    def _thomas_alg(self, a, b, c, d):
        n = len(b)
        x = np.zeros(n)
        for k in range(1,n):
            q = a[k]/b[k-1]
            b[k] = b[k] - c[k-1]*q
            d[k] = d[k] - d[k-1]*q
        q = d[n-1]/b[n-1]
        x[n-1] = q
        for k in range(n-2,-1,-1):
            q = (d[k]-c[k]*q)/b[k]
            x[k] = q
        return x
    
    def _heat_solver(self, fidelity,alpha,neumann_0,neumann_1):
        x = np.linspace(0,1,fidelity)
        t = np.linspace(0,1,fidelity)
        u = np.zeros((fidelity+1,fidelity+2))
        dx = x[1]-x[0]
        dt = t[1]-t[0]

        # Set heaviside IC
        for i in range(fidelity):
            if i*dx >= 0.25 and i*dx <= 0.75:
                u[0,i+1] = 1

        for n in range(0,fidelity): # temporal loop
            a = np.zeros(fidelity); b = np.zeros(fidelity); c = np.zeros(fidelity); d = np.zeros(fidelity)
            for i in range(1,fidelity+1): # spatial loop
                # Create vectors for a, b, c, d
                a[i-1] = -alpha*dt/dx**2
                b[i-1] = 1+2*alpha*dt/dx**2
                c[i-1] = -alpha*dt/dx**2
                d[i-1] = u[n,i]

            # Neumann coniditions 
            d[0] = (d[0] - ((alpha*dt/dx**2)*2*dx*neumann_0))/2 # Divide by 2 to keep symmetry
            d[-1] = (d[-1] + ((alpha*dt/dx**2)*2*dx*neumann_1))/2
            a[0] = 0
            b[0] = b[0]/2
            c[-1] = 0
            b[-1] = b[-1]/2

            # Solve
            u[n+1,1:-1] = self._thomas_alg(a,b,c,d)
        v = u[1:,1:-1]
        return v, x

class Heat3:
    def __init__(self,):
        self.M = 3
        self.dim = 3
        
        self.fidelity_list = [16,32,64,100]

        self.bounds = ((0,1),(-1,0),(0.01,0.1))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]
        
    def query(self, X, m):
        y_list = []
        N = X.shape[0]
        for n in range(N):
            y = self.single_query(X[n], m)
            y_list.append(y)
        #
        
        Y = np.array(y_list)
        re_Y = np.reshape(Y, [N,-1])
        
        #print(re_Y.shape)
        
        return re_Y 
        
    def single_query(self, X, m, interp=False):

        X = np.squeeze(X)
        fidelity = self.fidelity_list[m]
        u, x = self._heat_solver(fidelity, X[2], X[0], X[1])
            
        return u
        
    def interpolate(self, Y, m):
        # low to high
        N = Y.shape[0]
        
        lf = self.fidelity_list[m]
        hf = self.fidelity_list[-1]
        
        x_lf = np.linspace(0,1,lf)
        y_lf = np.linspace(0,1,lf)
        x_hf = np.linspace(0,1,hf)
        y_hf = np.linspace(0,1,hf)
        
        Y_interp_list = []
        
        for n in range(N):
            u = Y[n,:]
            interp_fn = interpolate.interp2d(x_lf, y_lf, u, kind='linear')
            u_hf = interp_fn(x_hf, y_hf)
            
            Y_interp_list.append(u_hf)
        # 
        
        Y_interp = np.array(Y_interp_list)
        re_Y_interp = np.reshape(Y_interp, [N,-1])
        
        return re_Y_interp
        
    def ground(self, X):
        y_list = []
        N = X.shape[0]
        for n in range(N):
            y = self.single_ground(X[n])
            y_list.append(y)
        #
        
        Y = np.array(y_list)
        re_Y = np.reshape(Y, [N,-1])
        
        #print(re_Y.shape)
        
        return re_Y 

    def single_ground(self, X):
        X = np.squeeze(X)
        u_ground, _ = self._heat_solver(self.fidelity_list[-1], X[2], X[0], X[1])
        
        return u_ground
    
    def _thomas_alg(self, a, b, c, d):
        n = len(b)
        x = np.zeros(n)
        for k in range(1,n):
            q = a[k]/b[k-1]
            b[k] = b[k] - c[k-1]*q
            d[k] = d[k] - d[k-1]*q
        q = d[n-1]/b[n-1]
        x[n-1] = q
        for k in range(n-2,-1,-1):
            q = (d[k]-c[k]*q)/b[k]
            x[k] = q
        return x
    
    def _heat_solver(self, fidelity,alpha,neumann_0,neumann_1):
        x = np.linspace(0,1,fidelity)
        t = np.linspace(0,1,fidelity)
        u = np.zeros((fidelity+1,fidelity+2))
        dx = x[1]-x[0]
        dt = t[1]-t[0]

        # Set heaviside IC
        for i in range(fidelity):
            if i*dx >= 0.25 and i*dx <= 0.75:
                u[0,i+1] = 1

        for n in range(0,fidelity): # temporal loop
            a = np.zeros(fidelity); b = np.zeros(fidelity); c = np.zeros(fidelity); d = np.zeros(fidelity)
            for i in range(1,fidelity+1): # spatial loop
                # Create vectors for a, b, c, d
                a[i-1] = -alpha*dt/dx**2
                b[i-1] = 1+2*alpha*dt/dx**2
                c[i-1] = -alpha*dt/dx**2
                d[i-1] = u[n,i]

            # Neumann coniditions 
            d[0] = (d[0] - ((alpha*dt/dx**2)*2*dx*neumann_0))/2 # Divide by 2 to keep symmetry
            d[-1] = (d[-1] + ((alpha*dt/dx**2)*2*dx*neumann_1))/2
            a[0] = 0
            b[0] = b[0]/2
            c[-1] = 0
            b[-1] = b[-1]/2

            # Solve
            u[n+1,1:-1] = self._thomas_alg(a,b,c,d)
        v = u[1:,1:-1]
        return v, x

class Burgers:
    def __init__(self,):
        self.M = 2
        self.dim = 1
        
        self.fidelity_list = [16,32,128]

        self.bounds = ((0,1))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]
        
    def query(self, X, m):
        if X.ndim == 1:
            X = np.expand_dims(X, 0)
        #
        
        query_key = get_random_alphanumeric_string(77)


        buff_path = os.path.join('data/__buff__', str(query_key)) + '.mat'
    
        if not os.path.exists('data/__buff__'):
            os.makedirs('data/__buff__')
        
        matlab_input = '['
        for i in range(X.shape[0]):
            s = np.array2string(X[i,:], separator=',',floatmode='maxprec',max_line_width=10000000)
            matlab_input += s
            if i < X.shape[0] - 1:
                matlab_input += ';'
            #
        #
        matlab_input += ']'
        
        matlab_cmd = 'addpath(genpath(\'data/Burgers\'));'
        matlab_cmd += 'query_client_burgers(' + matlab_input  + ',' + str(self.fidelity_list[m]) + ', \'' + buff_path + '\'' + ');'
        matlab_cmd += 'quit force'

        process = subprocess.Popen(["matlab", "-nodesktop", "-r", matlab_cmd],
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE)

        process.wait()
        
        retrived_data = loadmat(buff_path, squeeze_me=True, struct_as_record=False, mat_dtype=True)['data']

        Y = retrived_data.Y_interp

        if Y.ndim == 2:
            Y = np.expand_dims(Y, 2)

        N = Y.shape[2]

        Y_tr = np.transpose(Y, [2,0,1])

        re_Y = np.reshape(np.transpose(Y, [2,0,1]), [N,-1])

        return re_Y 
    
    def ground(self, X):
        return self.query(X, self.M)
        
    
    def interpolate(self, Y, m):
        # low to high
        
        N = Y.shape[0]
        
        lf = self.fidelity_list[m]
        hf = self.fidelity_list[-1]
        
        x_lf = np.linspace(0,1,lf)
        y_lf = np.linspace(0,1,lf)
        x_hf = np.linspace(0,1,hf)
        y_hf = np.linspace(0,1,hf)
        
        Y_interp_list = []
        
        for n in range(N):
            u = Y[n,:].reshape(lf, lf)
            interp_fn = interpolate.interp2d(x_lf, y_lf, u, kind='linear')
            u_hf = interp_fn(x_hf, y_hf)
            
            Y_interp_list.append(u_hf)
        # 
        
        Y_interp = np.array(Y_interp_list)
        re_Y_interp = np.reshape(Y_interp, [N,-1])
        
        return re_Y_interp
    
class Lbracket:
    def __init__(self,):
        self.M = 2
        self.dim = 2
        
        self.fidelity_list = [50,76,100]

        self.bounds = ((0,1),(0,1))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]
        
    def query(self, X, m):
        if X.ndim == 1:
            X = np.expand_dims(X, 0)
        #
        
        query_key = get_random_alphanumeric_string(77)


        buff_path = os.path.join('data/__buff__', str(query_key)) + '.mat'
    
        if not os.path.exists('data/__buff__'):
            os.makedirs('data/__buff__')
        
        matlab_input = '['
        for i in range(X.shape[0]):
            s = np.array2string(X[i,:], separator=',',floatmode='maxprec',max_line_width=10000000)
            matlab_input += s
            if i < X.shape[0] - 1:
                matlab_input += ';'
            #
        #
        matlab_input += ']'
        
        matlab_cmd = 'addpath(genpath(\'data/Topology-Optimization\'));'
        matlab_cmd += 'query_client_topopt(' + matlab_input  + ',' + str(self.fidelity_list[m]) + ', \'' + buff_path + '\'' + ');'
        matlab_cmd += 'quit force'
        
        process = subprocess.Popen(["matlab", "-nodesktop", "-r", matlab_cmd],
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE)
        
        
        process.wait()
        
        data = loadmat(buff_path, squeeze_me=True, struct_as_record=False, mat_dtype=True)['data']
        Y = data.Y
        
        if Y.ndim == 2:
            Y = np.expand_dims(Y, 2)
            
        tr_Y = np.transpose(Y, [2,0,1])
        
        N = tr_Y.shape[0]
        
        re_Y = np.reshape(tr_Y, [N,-1])
        
        return re_Y
    
    def ground(self, X):
        return self.query(X, self.M)
    
    def interpolate(self, Y, m):
        # low to high
        
        N = Y.shape[0]
        
        lf = self.fidelity_list[m]
        hf = self.fidelity_list[-1]
        
        x_lf = np.linspace(0,1,lf)
        y_lf = np.linspace(0,1,lf)
        x_hf = np.linspace(0,1,hf)
        y_hf = np.linspace(0,1,hf)
        
        Y_interp_list = []
        
        for n in range(N):
            u = Y[n,:].reshape(lf, lf)
            interp_fn = interpolate.interp2d(x_lf, y_lf, u, kind='cubic')
            u_hf = interp_fn(x_hf, y_hf)
            
            Y_interp_list.append(u_hf)
        # 
        
        Y_interp = np.array(Y_interp_list)
        re_Y_interp = np.reshape(Y_interp, [N,-1])
        
        return re_Y_interp
    
class Navier:
    def __init__(self,):
        self.M = 2
        self.dim = 5
        
        self.fidelity_list = [51,76,101]

        self.bounds = ((0,1),(0,1),(0,1),(0,1),(0,1))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]
        
    def query(self, X, m):
        if X.ndim == 1:
            X = np.expand_dims(X, 0)
        #
        
        query_key = get_random_alphanumeric_string(77)


        buff_path = os.path.join('data/__buff__', str(query_key)) + '.mat'
    
        if not os.path.exists('data/__buff__'):
            os.makedirs('data/__buff__')
        
        matlab_input = '['
        for i in range(X.shape[0]):
            s = np.array2string(X[i,:], separator=',',floatmode='maxprec',max_line_width=10000000)
            matlab_input += s
            if i < X.shape[0] - 1:
                matlab_input += ';'
            #
        #
        matlab_input += ']'
        
        matlab_cmd = 'addpath(genpath(\'data/NS-Solver\'));'
        matlab_cmd += 'query_client_ns(' + matlab_input  + ',' + str(m) + ', \'' + buff_path + '\'' + ');'
        matlab_cmd += 'quit force'
        
        process = subprocess.Popen(["matlab", "-nodesktop", "-r", matlab_cmd])
        stdout, stderr = process.communicate()
        
        process.wait()
        
        PRec = loadmat(buff_path, squeeze_me=True, struct_as_record=False, mat_dtype=True)['URec']
        
        if PRec.ndim == 3:
            PRec = np.expand_dims(PRec, 3)
            
        Nstep = PRec.shape[2]
        Ns = PRec.shape[3]

        tr_PRec = np.transpose(PRec, axes=[3,0,1,2])
        Ns = tr_PRec.shape[0]
        re_PRec = np.reshape(tr_PRec, [Ns, -1])
        
        return re_PRec
    
    def ground(self, X):
        return self.query(X, self.M)
    
    def interpolate(self, Y, m):
        Ns = Y.shape[0]
        lf = self.fidelity_list[m]
        hf = self.fidelity_list[-1]
        
        PRec = np.reshape(Y, [Ns, lf, lf, -1])

        Y_list = []
        
        for n in range(Ns):
            Pn = PRec[n,:,:,:]
            Pt_interp_list = []
            Nstep = PRec.shape[-1]
            for t in range(Nstep):
                Pt = Pn[:,:,t]
            
                x_lf = np.linspace(0,1,lf)
                y_lf = np.linspace(0,1,lf)
                interp_fn = interpolate.interp2d(x_lf, y_lf, Pt, kind='cubic')
                
                x_hf = np.linspace(0,1,hf)
                y_hf = np.linspace(0,1,hf)
            
                Pt_interp = interp_fn(x_hf, y_hf)

                Pt_interp_list.append(Pt_interp.reshape([-1]))
                
            #
            Pn_interp = np.array(Pt_interp_list).reshape([-1])
            
            Y_list.append(Pn_interp)
        #
        
        interp_Y = np.array(Y_list)

        return interp_Y
    

class Dataset:
    def __init__(self, Domain, trial):
        
        self.Domain = Domain
        self.Mfn = {
            'Heat3':Heat3,
            'Heat2':Heat2,
            'Poisson2':Poisson2,
            'Poisson3':Poisson3,
            'Burgers':Burgers,
            'Navier': Navier,
            'Lbracket': Lbracket,
        }[Domain]()
        
        data_path = os.path.join('data/__processed__', Domain)
        data_filename = Domain+'_trial'+str(trial)+'.h5'
        
        raw = load_from_h5f(os.path.join(data_path,data_filename))
        
        self.Ntrain_list = raw['Ntrain']
        self.Ntest_list = raw['Ntest']
        
        self.MF_X_train = raw['X_train_list']
        self.MF_y_train = raw['y_train_list']
        self.MF_y_train_ground = raw['y_train_ground_list']
        
        self.MF_X_test = raw['X_test_list']
        self.MF_y_test = raw['y_test_list']
        self.MF_y_test_ground = raw['y_test_ground_list']

        if raw['domain'] != self.Domain:
            exit(0)
        #

    def random_samples(self, N, m, seed):
        
        rand_state = np.random.get_state()

        try:
            np.random.seed(seed)
            noise = np.random.uniform(0,1,size=[N,self.Mfn.dim])
            scale = (self.Mfn.ub - self.Mfn.lb).reshape([1,-1])
        except:
            print('Errors occured when generating random noise...')
        finally:
            np.random.set_state(rand_state)
        #
        
        X = noise*scale + self.Mfn.lb
        y = self.query(X, m)
        
        return X, y

    
    def get_data(self, m, train=True, normalize=True, noise=0.00):
        if train:
            X = self.MF_X_train[m]
            y = self.MF_y_train[m]
            yg = self.MF_y_train_ground[m]
        else:
            X = self.MF_X_test[m]
            y = self.MF_y_test[m]
            yg = self.MF_y_test_ground[m]
            
        scales = self.get_scales(m, train)
        
        if normalize:
            X = (X-scales['X_mean']) / scales['X_std']
            y = (y-scales['y_mean']) / scales['y_std']
            
            
        y = y + noise*np.random.normal(size=y.shape)
    
        return X,y,yg
    
    def get_scales(self, m, train=True):

        scale_mean_X = np.mean(self.MF_X_test[m])
        scale_std_X = np.std(self.MF_X_test[m])

        scale_mean_y = np.mean(self.MF_y_test[m])
        scale_std_y = np.std(self.MF_y_test[m])
        
        scales = {}
        scales['X_mean'] = scale_mean_X
        scales['X_std'] = scale_std_X
        scales['y_mean'] = scale_mean_y
        scales['y_std'] = scale_std_y
        
        return scales
    
    def query(self, X, m):

        ym = self.Mfn.query(X, m)
        
        return ym
    
    def ground(self, X):
        y_ground = self.Mfn.ground(X)
        
        return y_ground
    
    def interp_to_ground(self, Y, m):
        Y_interp = self.Mfn.interpolate(Y, m)
        return Y_interp
    
    def interp_to_hf(self, Y_ground, m):
        Y_interp = self.Mfn.inv_interpolate(Y_ground, m)
        return Y_interp

    def get_N_bounds(self, m):

        scales = self.get_scales(m)
        X_mean = scales['X_mean']
        X_std = scales['X_std']
        
        N_lb = (self.Mfn.lb - X_mean)/X_std
        N_ub = (self.Mfn.ub - X_mean)/X_std
        
        return N_lb, N_ub
    
    def append(self, N_X_query, m):

        scales = self.get_scales(m)
        X_query = N_X_query*scales['X_std'] + scales['X_mean']
        

        X_query = np.clip(np.squeeze(X_query), self.Mfn.lb, self.Mfn.ub).reshape([1,-1]) #
        
        y_query = self.query(X_query, m)

        y_query_ground = self.ground(X_query)
   
        self.MF_X_train[m] = np.vstack((self.MF_X_train[m], X_query))
        self.MF_y_train[m] = np.vstack((self.MF_y_train[m], y_query))
        self.MF_y_train_ground[m] = np.vstack((self.MF_y_train_ground[m], y_query_ground))
        self.Ntrain_list[m] += X_query.shape[0]
        
