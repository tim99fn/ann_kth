import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def create_data(n,mu_A,mu_B,sigma_A,sigma_B):
    X = np.zeros((2,n*2))

    X[0,:n//2] = np.random.normal(-mu_A[0],sigma_A,n//2)
    X[0,n//2:n] = np.random.normal(mu_A[0],sigma_A,n//2)
    X[1,:n] = np.random.normal(mu_A[1],sigma_A,n)

    X[0,n:] = np.random.normal(mu_B[0],sigma_B,n)
    X[1,n:] = np.random.normal(mu_B[1],sigma_B,n)

    T = np.ones(n*2)
    T[n:] = -1

    p = np.random.permutation(n*2)
    X = X[:,p]
    T = T[p]

    return X, T

def split_data(X,T,flag):

    X_A = X[:,T==1]
    X_B = X[:,T==-1]

    X_train = np.zeros((2,50))
    X_val = np.zeros((2,150))
    T_train = np.ones(50)
    T_val = np.ones(150)
    
    if flag=="25-25":
        i_A_rand = np.random.choice(100,100,replace=False)
        i_A_train = i_A_rand[:25]
        i_A_val = i_A_rand[25:]
        i_B_rand = np.random.choice(100,100,replace=False)
        i_B_train = i_A_rand[:25]
        i_B_val = i_A_rand[25:]
    elif flag=="50-0":
        i_A_rand = np.random.choice(100,100,replace=False)
        i_A_train = i_A_rand[:50]
        i_A_val = i_A_rand[50:]
        i_B_rand = np.random.choice(100,100,replace=False)
        i_B_train = []
        i_B_val = i_B_rand
    elif flag=="80-20":
        i_A_neg = np.where(X_A[0,:]<0)[0]
        i_A_pos = np.where(X_A[0,:]>0)[0]
        i_neg_rand = np.random.choice(50,50,replace=False)
        i_pos_rand = np.random.choice(50,50,replace=False)
        i_A_train = np.hstack(i_A_neg[i_neg_rand[:10]],i_A_pos[i_pos_rand[:40]])
        i_A_val = np.hstack(i_A_neg[i_neg_rand[10:]],i_A_pos[i_pos_rand[40:]])
        i_B_rand = np.random.choice(100,100,replace=False)
        i_B_train = i_B_rand
        i_B_val = []

    X_train[:,:len(i_A_train)] = X_A[:,i_A_train]
    X_train[:,len(i_A_train):] = X_B[:,i_B_train]
    X_val[:,len(i_A_val):] = X_A[:,i_A_val]
    X_val[:,len(i_A_val):] = X_B[:,i_B_val]

    T_train[len(i_A_train):] = -1
    T_val[len(i_A_val):] = -1

    p = np.random.permutation(50)
    X_train = X_train[:,p]
    T_train = T_train[p]

    p = np.random.permutation(150)
    X_val = X_val[:,p]
    T_val = T_val[p]

    return X_train, X_val, T_train, T_val

def phi(x):
    return 2/(1+np.exp(-x))-1

def phi_prime(phi_x):
    return (1+phi_x)*(1-phi_x)/2

def accuracy(W,data,target):
    transformed= W@data
    transformed[transformed<0]=-1
    transformed[transformed>=0]=1
    transformed_neg=transformed[target==-1]
    transformed_pos=transformed[target==1]
    target_pos=target[target==1]
    target_neg=target[target==-1]
    acc_pos=np.count_nonzero(transformed_pos==target_pos)/len(target_pos)
    acc_neg=np.count_nonzero(transformed_neg==target_neg)/len(target_neg)
    return acc_pos, acc_neg

def train_network(X,T,W,V,epochs,eta,alpha):
    delta_W = np.zeros_like(W)
    delta_V = np.zeros_like(V)
    error_mse = np.zeros(epochs)
    error_misclassified = np.zeros(epochs)
    for e in range(epochs):
        # 1) forward pass
        H_in = W@X
        H_out = phi(H_in)
        H_out = np.concatenate((H_out,np.ones((1,H_out.shape[1]))),axis=0)
        O_in = V@H_out
        O_out = phi(O_in)
        # Calculate errors
        pred = np.copy(O_out)
        pred[pred<0] = -1
        pred[pred>=0] = 1
        error_mse[e] = np.mean((O_out - T)**2)
        error_misclassified[e] = 1 - np.mean(pred==T)
        # 2) backward pass
        delta_o = (O_out - T) * phi_prime(O_out)
        delta_h = (V.T @ delta_o) * phi_prime(H_out)
        delta_h = delta_h[:-1,:]
        # 3) weight update
        delta_W = alpha * delta_W - (1-alpha) * delta_h @ X.T
        delta_V = alpha * delta_V - (1-alpha) * delta_o @ H_out.T
        W = W + eta * delta_W
        V = V + eta * delta_V
    return error_mse, error_misclassified

def estimate_boundary():
    pass

if __name__ == "__main__":
    np.random.seed(33)
    epochs = 100
    eta = 0.1
    alpha = 0.9
    hidden_layer_sizes = [2,3,4,5,6,8,10]
    X, T = create_data(n=100,mu_A=[1.0,0.3],mu_B=[0.0,0.1],sigma_A=0.2,sigma_B=0.3)
    X = np.concatenate((X,np.ones((1,X.shape[1]))),axis=0)
    # Task 3.1.1. Question 1
    fig1,axes1 = plt.subplots(1,2,figsize=(15, 5))
    for hidden_layer_size in hidden_layer_sizes:
        W = np.random.normal(size=(hidden_layer_size,3))
        V = np.random.normal(size=(1,hidden_layer_size+1))
        error_mse, error_misclassified = train_network(X,T,W,V,epochs,eta,alpha)
        axes1[0].plot(range(epochs),error_mse,label='k='+str(hidden_layer_size))
        axes1[1].plot(range(epochs),error_misclassified,label='k='+str(hidden_layer_size))
    # fig1.set_title('Backprob learning')
    axes1[0].set_xlabel('Epoch')
    axes1[0].set_ylabel('MSE')
    axes1[0].legend()
    axes1[1].set_xlabel('Epoch')
    axes1[1].set_ylabel('Error rate')
    axes1[1].legend()
    fig1.suptitle(f"Backprob learning for different hidden layer sizes, alpha={alpha}, eta={eta}")
    fig1.tight_layout()
    plt.show()
    fig1.savefig("img/test.png",dpi=300)

    # # Task 3.1.1. Question 1
    # flag = "25-25"
    # X_train, X_val, T_train, T_val = split_data(X,T,flag)
    # for hidden_layer_size in hidden_layer_sizes:
    #     train_network()
    # estimate_boundary()

    