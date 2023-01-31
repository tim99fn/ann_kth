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

    X_train = np.zeros((3,50))
    X_val = np.zeros((3,150))
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
        i_A_train = np.hstack((i_A_neg[i_neg_rand[:10]],i_A_pos[i_pos_rand[:40]]))
        i_A_val = np.hstack((i_A_neg[i_neg_rand[10:]],i_A_pos[i_pos_rand[40:]]))
        i_B_rand = np.random.choice(100,100,replace=False)
        i_B_train = []
        i_B_val = i_B_rand

    X_train[:,:len(i_A_train)] = X_A[:,i_A_train]
    X_train[:,len(i_A_train):] = X_B[:,i_B_train]
    X_val[:,:len(i_A_val)] = X_A[:,i_A_val]
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

def predict_output(X,W,V):
    # Forward pass
    H_in = W@X
    H_out = phi(H_in)
    H_out = np.concatenate((H_out,np.ones((1,H_out.shape[1]))),axis=0)
    O_in = V@H_out
    O_out = phi(O_in)
    # Map to output space
    pred = np.copy(O_out)
    pred[pred<0] = -1
    pred[pred>=0] = 1
    return O_out, pred, H_out

def train_network(X,T,W,V,epochs,eta,alpha):
    delta_W = np.zeros_like(W)
    delta_V = np.zeros_like(V)
    error_mse = np.zeros(epochs)
    error_misclassified = np.zeros(epochs)
    for e in range(epochs):
        # 1) forward pass
        O_out, pred, H_out = predict_output(X,W,V)
        # Calculate errors
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
    return W, V, error_mse, error_misclassified

def train_and_validate(X_train,T_train,X_val,T_val,W,V,epochs,eta,alpha):
    delta_W = np.zeros_like(W)
    delta_V = np.zeros_like(V)
    error_mse = np.zeros((epochs,2))
    error_misclassified = np.zeros((epochs,2))
    for e in range(epochs):
        # 1) forward pass
        O_out_train, pred_train, H_out = predict_output(X_train,W,V)
        # Calculate errors
        error_mse[e,0] = np.mean((O_out_train - T_train)**2)
        error_misclassified[e,0] = 1 - np.mean(pred_train==T_train)
        # 2) Predict outputs for validation
        O_out_val, pred_val, _ = predict_output(X_val,W,V)
        error_mse[e,1] = np.mean((O_out_val - T_val)**2)
        error_misclassified[e,1] = 1 - np.mean(pred_val==T_val)
        # 3) backward pass
        delta_o = (O_out_train - T_train) * phi_prime(O_out_train)
        delta_h = (V.T @ delta_o) * phi_prime(H_out)
        delta_h = delta_h[:-1,:]
        # 4) weight update
        delta_W = alpha * delta_W - (1-alpha) * delta_h @ X_train.T
        delta_V = alpha * delta_V - (1-alpha) * delta_o @ H_out.T
        W = W + eta * delta_W
        V = V + eta * delta_V
    return W, V, error_mse, error_misclassified

def train_and_validate_sequential(X_train,T_train,X_val,T_val,W,V,epochs,eta,alpha):
    delta_W = np.zeros_like(W)
    delta_V = np.zeros_like(V)
    error_mse = np.zeros((epochs,2))
    error_misclassified = np.zeros((epochs,2))
    for e in range(epochs):
        for i in range(X_train.shape[1]):
            x = X_train[:,i].reshape(3,1)
            t = T_train[i]
            # 1) forward pass
            O_out_train, pred_train, H_out = predict_output(x,W,V)
            # Calculate errors
            error_mse[e,0] += (O_out_train - t)**2
            error_misclassified[e,0] += pred_train==t
            # 3) backward pass
            delta_o = (O_out_train - t) * phi_prime(O_out_train)
            delta_h = (V.T @ delta_o) * phi_prime(H_out)
            delta_h = delta_h[:-1,:]
            # 4) weight update
            delta_W = alpha * delta_W - (1-alpha) * delta_h @ x.T
            delta_V = alpha * delta_V - (1-alpha) * delta_o @ H_out.T
            W = W + eta * delta_W
            V = V + eta * delta_V
        error_mse[e,0] = error_mse[e,0]/X_train.shape[1]
        error_misclassified[e,0] = 1 - error_misclassified[e,0]/X_train.shape[1]
        O_out_val, pred_val, _ = predict_output(X_val,W,V)
        error_mse[e,1] = np.mean((O_out_val - T_val)**2)
        error_misclassified[e,1] = 1 - np.mean(pred_val==T_val)
    return W, V, error_mse, error_misclassified

def plot_data(X,T):
    fig,ax = plt.subplots()
    scatter = ax.scatter(X[0,:],X[1,:], c=T)
    legend1 = ax.legend(handles=[scatter.legend_elements()[0][1],scatter.legend_elements()[0][0]], labels=["Class A", "Class B"], loc="lower left")
    ax.add_artist(legend1)
    fig.savefig("img/311_1/data.png", dpi=300)

def estimate_boundary(X_train,T_train,X_val,T_val,W,V,flag,hidden_layer_size):
    fig2, ax2 = plt.subplots()
    # Plot the data
    scatter2 = ax2.scatter(X_val[0,:],X_val[1,:], c=T_val)
    scatter = ax2.scatter(X_train[0,:],X_train[1,:], c=T_train, alpha=0.5, cmap="winter")
    # Set up a mesh
    x = np.arange(-2,2,0.01)
    y = np.arange(-1,1,0.01)
    mesh = np.ones((3,x.size*y.size))
    mesh[0,:] = np.repeat(x,y.size)
    mesh[1,:] = np.tile(y,x.size)
    # Predict class for each x,y pair on the mesh
    _, pred, _ = predict_output(mesh,W,V)
    # pred = np.flipud(np.reshape(pred, (y.size,x.size), order="F"))
    pred = np.reshape(pred, (y.size,x.size), order="F")
    # Plot the resulting contour
    p = ax2.contourf(x,y,pred,alpha=0.1)
    if len(scatter.legend_elements()[0]) == 1:
        legend1 = ax2.legend(handles=[scatter.legend_elements()[0][0],scatter2.legend_elements()[0][1],scatter2.legend_elements()[0][0]], labels=["Class A Training", "Class A Validation","Class B Validation"], loc="lower left")
    elif len(scatter2.legend_elements()[0]) == 1:
        legend1 = ax2.legend(handles=[scatter.legend_elements()[0][1],scatter.legend_elements()[0][0],scatter2.legend_elements()[0][0]], labels=["Class A Training", "Class B Training","Class A Validation"], loc="lower left")
    else:
        legend1 = ax2.legend(handles=[scatter.legend_elements()[0][1],scatter2.legend_elements()[0][1],scatter.legend_elements()[0][0],scatter2.legend_elements()[0][0]], labels=["Class A Training", "Class A Validation", "Class B Training","Class B Validation"], loc="lower left")
    ax2.add_artist(legend1)
    ax2.set_title(f"Estimated decision boundary, subsampling={flag}, hidden layer size={hidden_layer_size}")
    fig2.savefig("img/311_2/estimated_boundary_"+flag+".png",dpi=300)
    # plt.show()

def plot_learning_curves(trials,epochs,hidden_layer_sizes,error_mse,error_misclassified):
    fig1,axes1 = plt.subplots(1,2,figsize=(15, 5))
    for i,hidden_layer_size in enumerate(hidden_layer_sizes):
        mean_mse, var_mse = np.mean(error_mse[:,:,i],axis=0), np.var(error_mse[:,:,i],axis=0)
        mean_misclassified, var_misclassified = np.mean(error_misclassified[:,:,i],axis=0), np.var(error_misclassified[:,:,i],axis=0)
        axes1[0].plot(range(epochs),mean_mse,label='k='+str(hidden_layer_size))
        axes1[0].fill_between(range(epochs), mean_mse-var_mse, mean_mse+var_mse, alpha=.3)
        axes1[1].plot(range(epochs),mean_misclassified,label='k='+str(hidden_layer_size))
        axes1[1].fill_between(range(epochs), mean_misclassified-var_misclassified, mean_misclassified+var_misclassified, alpha=.3)
    axes1[0].set_xlabel('Epoch')
    axes1[0].set_ylabel('MSE')
    axes1[0].set_xlim(0,epochs)
    axes1[0].legend()
    axes1[1].set_xlabel('Epoch')
    axes1[1].set_ylabel('Error rate')
    axes1[1].set_xlim(0,epochs)
    axes1[1].legend()
    fig1.suptitle(f"Backprob learning for different hidden layer sizes, alpha={alpha}, eta={eta}, trials={trials}")
    fig1.tight_layout()
    fig1.savefig("img/311_1/different_layer_sizes.png",dpi=300)

def plot_learning_and_validation_curves(trials,epochs,hidden_layer_sizes,error_mse,error_misclassified,flag):
    fig1,axes1 = plt.subplots(1,2,figsize=(15, 5))
    for i,hidden_layer_size in enumerate(hidden_layer_sizes):
        mean_mse_train, var_mse_train = np.mean(error_mse[:,0,i,:],axis=1), np.var(error_mse[:,0,i,:],axis=1)
        mean_mse_val, var_mse_val = np.mean(error_mse[:,1,i,:],axis=1), np.var(error_mse[:,1,i,:],axis=1)
        mean_misclassified_train, var_misclassified_train = np.mean(error_misclassified[:,0,i,:],axis=1), np.var(error_misclassified[:,0,i,:],axis=1)
        mean_misclassified_val, var_misclassified_val = np.mean(error_misclassified[:,1,i,:],axis=1), np.var(error_misclassified[:,1,i,:],axis=1)
        axes1[0].plot(range(epochs),mean_mse_train,label='Training k='+str(hidden_layer_size))
        axes1[0].fill_between(range(epochs), mean_mse_train-var_mse_train, mean_mse_train+var_mse_train, alpha=.3)
        axes1[0].plot(range(epochs),mean_mse_val,label='Validation k='+str(hidden_layer_size))
        axes1[0].fill_between(range(epochs), mean_mse_val-var_mse_val, mean_mse_val+var_mse_val, alpha=.3)
        axes1[1].plot(range(epochs),mean_misclassified_train,label='Training k='+str(hidden_layer_size))
        axes1[1].fill_between(range(epochs), mean_misclassified_train-var_misclassified_train, mean_misclassified_train+var_misclassified_train, alpha=.3)
        axes1[1].plot(range(epochs),mean_misclassified_val,label='Validation k='+str(hidden_layer_size))
        axes1[1].fill_between(range(epochs), mean_misclassified_val-var_misclassified_val, mean_misclassified_val+var_misclassified_val, alpha=.3)

    axes1[0].set_xlabel('Epoch')
    axes1[0].set_ylabel('MSE')
    axes1[0].set_xlim(0,epochs)
    axes1[0].legend()
    axes1[1].set_xlabel('Epoch')
    axes1[1].set_ylabel('Error rate')
    axes1[1].set_xlim(0,epochs)
    axes1[1].legend()
    fig1.suptitle(f"Backprob learning: alpha={alpha}, eta={eta}, trials={trials}, subsampling={flag}")
    fig1.tight_layout()
    fig1.savefig("img/311_2/training_validation_error_"+flag+".png",dpi=300)

def plot_model_complexity_error(trials,epochs,hidden_layer_sizes,error_mse,error_misclassified,flag):
    fig1,axes1 = plt.subplots(1,2,figsize=(15, 5))
    mean_mse_train, var_mse_train = np.mean(error_mse[epochs-1,0,:,:],axis=1), np.var(error_mse[epochs-1,0,:,:],axis=1)
    mean_mse_val, var_mse_val = np.mean(error_mse[epochs-1,1,:,:],axis=1), np.var(error_mse[epochs-1,1,:,:],axis=1)
    mean_misclassified_train, var_misclassified_train = np.mean(error_misclassified[epochs-1,0,:,:],axis=1), np.var(error_misclassified[epochs-1,0,:,:],axis=1)
    mean_misclassified_val, var_misclassified_val = np.mean(error_misclassified[epochs-1,1,:,:],axis=1), np.var(error_misclassified[epochs-1,1,:,:],axis=1)

    axes1[0].plot(hidden_layer_sizes,mean_mse_train,label='Training')
    axes1[0].fill_between(hidden_layer_sizes, mean_mse_train-var_mse_train, mean_mse_train+var_mse_train,alpha=.3)
    axes1[0].plot(hidden_layer_sizes,mean_mse_val,label='Validation')
    axes1[0].fill_between(hidden_layer_sizes, mean_mse_val-var_mse_val, mean_mse_val+var_mse_val,alpha=.3)

    axes1[1].plot(hidden_layer_sizes,mean_misclassified_train,label='Training')
    axes1[1].fill_between(hidden_layer_sizes, mean_misclassified_train-var_misclassified_train, mean_misclassified_train+var_misclassified_train,alpha=.3)
    axes1[1].plot(hidden_layer_sizes,mean_misclassified_val,label='Validation')
    axes1[1].fill_between(hidden_layer_sizes, mean_misclassified_val-var_misclassified_val, mean_misclassified_val+var_misclassified_val,alpha=.3)
    
    axes1[0].set_xlabel('Hidden layer size')
    axes1[0].set_xticks(hidden_layer_sizes)
    axes1[0].set_ylabel('MSE')
    axes1[0].set_xlim(1,len(hidden_layer_sizes))
    axes1[0].legend()
    axes1[1].set_xlabel('Hidden layer size')
    axes1[1].set_xticks(hidden_layer_sizes)
    axes1[1].set_ylabel('Error rate')
    axes1[1].set_xlim(1,len(hidden_layer_sizes))
    axes1[1].legend()
    fig1.suptitle(f"Backprob learning, error over hidden layer size: alpha={alpha}, eta={eta}, trials={trials}, subsampling={flag}")
    fig1.tight_layout()
    fig1.savefig("img/311_2/model_complexity_error_"+flag+".png",dpi=300)

if __name__ == "__main__":
    np.random.seed(33)
    epochs = 100
    eta = 0.1
    alpha = 0.9
    hidden_layer_sizes = [2,3,5,8,10,15]
    X, T = create_data(n=100,mu_A=[1.0,0.3],mu_B=[0.0,0.1],sigma_A=0.2,sigma_B=0.3)
    plot_data(X,T)
    X = np.concatenate((X,np.ones((1,X.shape[1]))),axis=0)

    # Task 3.1.1. Question 1
    if False:
        trials = 20
        error_mse = np.zeros((trials,epochs,len(hidden_layer_sizes)))
        error_misclassified = np.zeros((trials,epochs,len(hidden_layer_sizes)))
        for i,hidden_layer_size in enumerate(hidden_layer_sizes):
            for j in range(trials):
                W = np.random.normal(size=(hidden_layer_size,3))
                V = np.random.normal(size=(1,hidden_layer_size+1))
                W, V, error_mse[j,:,i], error_misclassified[j,:,i] = train_network(X,T,W,V,epochs,eta,alpha)
        plot_learning_curves(trials,epochs,hidden_layer_sizes,error_mse,error_misclassified)

    # Task 3.1.1. Question 2
    hidden_layer_sizes = np.arange(1,21)
    trials = 20
    if False:
        for flag in ["25-25","50-0","80-20"]:
            X_train, X_val, T_train, T_val = split_data(X,T,flag)
            error_mse = np.zeros((epochs,2,hidden_layer_sizes.size,trials))
            error_misclassified = np.zeros((epochs,2,hidden_layer_sizes.size,trials))
            error_mse_seq = np.zeros((epochs,2,hidden_layer_sizes.size,trials))
            error_misclassified_seq = np.zeros((epochs,2,hidden_layer_sizes.size,trials))
            for i, hidden_layer_size in enumerate(hidden_layer_sizes):
                for j in range(trials):
                    W = np.random.normal(size=(hidden_layer_size,3))
                    V = np.random.normal(size=(1,hidden_layer_size+1))
                    W_seq = np.random.normal(size=(hidden_layer_size,3))
                    V_seq = np.random.normal(size=(1,hidden_layer_size+1))
                    # TODO split accuracy into class A and B -> maybe unnecessary
                    W, V, error_mse[:,:,i,j], error_misclassified[:,:,i,j] = train_and_validate(X_train,T_train,X_val,T_val,W,V,epochs,eta,alpha)
                    _, _, error_mse_seq[:,:,i,j], error_misclassified_seq[:,:,i,j] = train_and_validate_sequential(X_train,T_train,X_val,T_val,W_seq,V_seq,epochs,eta,alpha)
                if hidden_layer_size == 8:
                    estimate_boundary(X_train,T_train,X_val,T_val,W,V,flag,hidden_layer_size)
            plot_learning_and_validation_curves(trials,epochs,hidden_layer_sizes[[2,4,7]],error_mse[:,:,[2,4,7],:],error_misclassified[:,:,[2,4,7],:],flag)
            plot_learning_and_validation_curves(trials,epochs,hidden_layer_sizes[[2,4,7]],error_mse_seq[:,:,[2,4,7],:],error_misclassified_seq[:,:,[2,4,7],:],flag+"-seq")
            plot_model_complexity_error(trials,epochs,hidden_layer_sizes,error_mse,error_misclassified,flag)
            # plt.show()

    # Flipped training and validation sets
    hidden_layer_sizes = np.arange(1,21)
    trials = 20
    if True:
        for flag in ["25-25","50-0","80-20"]:
            X_val, X_train, T_val, T_train = split_data(X,T,flag)
            error_mse = np.zeros((epochs,2,hidden_layer_sizes.size,trials))
            error_misclassified = np.zeros((epochs,2,hidden_layer_sizes.size,trials))
            error_mse_seq = np.zeros((epochs,2,hidden_layer_sizes.size,trials))
            error_misclassified_seq = np.zeros((epochs,2,hidden_layer_sizes.size,trials))
            for i, hidden_layer_size in enumerate(hidden_layer_sizes):
                for j in range(trials):
                    W = np.random.normal(size=(hidden_layer_size,3))
                    V = np.random.normal(size=(1,hidden_layer_size+1))
                    W_seq = np.random.normal(size=(hidden_layer_size,3))
                    V_seq = np.random.normal(size=(1,hidden_layer_size+1))
                    W, V, error_mse[:,:,i,j], error_misclassified[:,:,i,j] = train_and_validate(X_train,T_train,X_val,T_val,W,V,epochs,eta,alpha)
                    _, _, error_mse_seq[:,:,i,j], error_misclassified_seq[:,:,i,j] = train_and_validate_sequential(X_train,T_train,X_val,T_val,W_seq,V_seq,epochs,eta,alpha)
                if hidden_layer_size == 6:
                    estimate_boundary(X_train,T_train,X_val,T_val,W,V,flag,hidden_layer_size)
            plot_learning_and_validation_curves(trials,epochs,hidden_layer_sizes[[2,4,7]],error_mse[:,:,[2,4,7],:],error_misclassified[:,:,[2,4,7],:],flag)
            plot_learning_and_validation_curves(trials,epochs,hidden_layer_sizes[[2,4,7]],error_mse_seq[:,:,[2,4,7],:],error_misclassified_seq[:,:,[2,4,7],:],flag+"-seq")
            plot_model_complexity_error(trials,epochs,hidden_layer_sizes,error_mse,error_misclassified,flag)
            # plt.show()

    # Task 3.1.2 -> skip for now

    # Task 3.1.3




    