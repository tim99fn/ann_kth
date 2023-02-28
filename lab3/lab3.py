import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import itertools

def update(W,x):
    return np.sign(W @ x)

def task_one():
    # Define original input patterns
    x1 = np.array([-1,-1,1,-1,1,-1,-1,1]).reshape(8,1)
    x2 = np.array([-1,-1,-1,-1,-1,1,-1,-1]).reshape(8,1)
    x3 = np.array([-1,1,1,-1,-1,1,-1,1]).reshape(8,1)
    # Calculate weight matrix
    W = x1 @ x1.T + x2 @ x2.T + x3 @ x3.T
    assert W.shape == (8,8)
    # Apply update rule and compare with original input patterns
    assert np.all(x1 == update(W,x1))
    assert np.all(x2 == update(W,x2))
    assert np.all(x3 == update(W,x3))
    print("Success!")
    # Define distorted input patterns
    x1d = np.array([1,-1,1,-1,1,-1,-1,-1]).reshape(8,1)
    x2d = np.array([1,1,-1,-1,-1,1,-1,-1]).reshape(8,1)
    x3d = np.array([1,1,1,-1,1,1,-1,1]).reshape(8,1)
    # Apply update until convergence
    while not np.all(x1d == update(W,x1d)):
        x1d = update(W,x1d)
    while not np.all(x2d == update(W,x2d)):
        x2d = update(W,x2d)
    while not np.all(x3d == update(W,x3d)):
        x3d = update(W,x3d)
    # Compare with original input pattern
    if np.all(x1 == x1d):
        print("x1d converges to x1")
    if np.all(x2 == x2d):
        print("x2d converges to x2")
    if np.all(x3 == x3d):
        print("x3d converges to x3")
    # Find all attractors
    X = np.array([list(i) for i in itertools.product([-1, 1], repeat=8)]).T
    while not np.all(X == update(W,X)):
        X = update(W,X)
    X_unique, counts = np.unique(X, axis=1, return_counts=True)
    print(f"Number of attractors: {X_unique.shape[1]} and their counts: {counts}")
    # Change starting patterns more substantially
    x1d = np.array([1,1,-1,1,-1,-1,-1,1]).reshape(8,1) # Flip first 5 bits
    while not np.all(x1d == update(W,x1d)):
        x1d = update(W,x1d)
    print(x1d)
    print("x1d converges to x1" if np.all(x1 == x1d) else "x1d does not converge to x1")
    
def task_two(p1,p2,p3,p10,p11,W):
    # Display the frist three patterns
    plt.figure()
    plt.subplot(2,3,1)
    plt.imshow(p1.reshape(32,32).T, cmap='gray')
    plt.subplot(2,3,2)
    plt.imshow(p2.reshape(32,32).T, cmap='gray')
    plt.subplot(2,3,3)
    plt.imshow(p3.reshape(32,32).T, cmap='gray')
    # Apply update rule and compare with original input patterns
    plt.subplot(2,3,4)
    plt.imshow(update(W,p1).reshape(32,32).T, cmap='gray')
    plt.subplot(2,3,5)
    plt.imshow(update(W,p2).reshape(32,32).T, cmap='gray')
    plt.subplot(2,3,6)
    plt.imshow(update(W,p3).reshape(32,32).T, cmap='gray')
    plt.savefig("img/update_p1_p2_p3.png", dpi=300, bbox_inches="tight")
    assert np.all(p1 == update(W,p1))
    assert np.all(p2 == update(W,p2))
    assert np.all(p3 == update(W,p3))
    print("p1, p2, p3 stable")
    # Try to restore pattern 10
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(p10.reshape(32,32).T, cmap='gray')
    while not np.all(p10 == update(W,p10)):
        p10 = update(W,p10)
    plt.subplot(2,2,2)
    plt.imshow(p10.reshape(32,32).T, cmap='gray')
    # Try to restore pattern 11
    plt.subplot(2,2,3)
    plt.imshow(p11.reshape(32,32).T, cmap='gray')
    while not np.all(p11 == update(W,p11)):
        p11 = update(W,p11)
    plt.subplot(2,2,4)
    plt.imshow(p11.reshape(32,32).T, cmap='gray')
    plt.savefig("img/restore_p10_p11.png", dpi=300, bbox_inches="tight")
    # Try random pattern
    np.random.seed(124)
    p_prev = np.zeros((1024,1))
    p = np.random.choice([-1,1], size=(1024,1))
    P = np.copy(p)
    while not np.all(p == p_prev):
        p_prev = np.copy(p)
        for i in range(1024):
            p[i] = np.sign(W[i,:] @ p)
            if i%32 == 0 or i == 1023:
                P = np.hstack((P,p))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    def update_plot(i):
        ax.clear()
        ax.imshow(P[:,i].reshape(32,32).T, cmap='gray')
    a = anim.FuncAnimation(fig, func=update_plot, frames=P.shape[1], repeat=False)
    a.save(f"img/restore_random.gif", fps=5)

def calculate_energy(p,W):
    return (- p.T @ W @ p)[0][0]

def random_energy_iteration(p,W,name):
    p_prev = np.zeros((1024,1))
    E = np.zeros(1)
    E[0] = calculate_energy(p,W)
    while not np.all(p == p_prev) and E.shape[0] < 10245: # At most ten full iterations
        p_prev = np.copy(p)
        for i in range(1024):
            p[i] = np.sign(W[i,:] @ p)
            E = np.hstack((E,np.array([calculate_energy(p,W)])))
    plt.figure()
    plt.plot(E[:-1024])
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title("Energy of random pattern")
    plt.savefig(f"img/{name}.png", dpi=300, bbox_inches="tight")

def task_three(p1,p2,p3,p10,p11,W):
    # Calculate energy of patterns
    E_p1 = calculate_energy(p1,W)
    E_p2 = calculate_energy(p2,W)
    E_p3 = calculate_energy(p3,W)
    E_p10 = calculate_energy(p10,W)
    E_p11 = calculate_energy(p11,W)
    print(f"Energy of p1: {E_p1}, p2: {E_p2}, p3: {E_p3}, p10: {E_p10}, p11: {E_p11}")
    # Try random pattern
    np.random.seed(124)
    p = np.random.choice([-1,1], size=(1024,1))
    random_energy_iteration(p,W,"energy_random")
    W2 = np.random.normal(size=(1024,1024))
    random_energy_iteration(p,W2,"energy_random_gaussian_W")
    random_energy_iteration(p,0.5*(W2+W2.T),"energy_random_symmetric_gaussian_W")

def add_noise_and_restore(p,f,W):
    p_noise = np.copy(p)
    id = np.random.choice(1024, size=int(f*1024), replace=False)
    p_noise[id] = -p_noise[id]
    while not np.all(p == p_noise) and not np.all(p_noise == update(W,p_noise)):
        p_noise = update(W,p_noise)
    if np.all(p == p_noise):
        return 1,p_noise
    else:
        return 0

def task_four(p1,p2,p3,W):
    f_split = np.arange(0,1.03,0.05)
    np.random.seed(82)
    # Add noise to p1 and try to restore
    restored = np.zeros((3,11,1000))
    patterncount=np.zeros((1024,3300))
    for i in range(1000):
        for j, f in enumerate(f_split):
            restored[0,j,i],final = add_noise_and_restore(p1,f,W)
            restored[1,j,i],final1= add_noise_and_restore(p2,f,W)
            restored[2,j,i],final2= add_noise_and_restore(p3,f,W)
            patterncount[:,i*11+j*3]=final.reshape(1024)
            patterncount[:,i*11+j*3+1]=final1.reshape(1024)
            patterncount[:,i*11+j*3+2]=final2.reshape(1024)
             
            
    x_unique, x_counts = np.unique(patterncount,axis=1, return_counts=True)  
    for i in range(x_unique.shape[1]):
        plt.figure()     
        plt.imshow(x_unique[:,i].reshape(32,32).T,cmap='gray')
        plt.title("Pattern "+str(i)+", count: "+str(x_counts[i]))
        plt.savefig("img/attractor_"+str(i)+".png", dpi=300, bbox_inches="tight")
    
    plt.figure()
    plt.plot(f_split, np.mean(restored[0,:,:],axis=1), label="p1")
    plt.plot(f_split, np.mean(restored[1,:,:],axis=1), label="p2")
    plt.plot(f_split, np.mean(restored[2,:,:],axis=1), label="p3")
    plt.xlabel("Noise level")
    plt.ylabel("Restored")
    plt.xticks(np.arange(0,1.1,0.1))
    # plt.yticks([0,1],["False","True"])
    plt.title(f"Average restoration performance for different noise levels and {trials} trials")
    plt.legend()
    plt.savefig("img/restore_3_noisy_patterns.png", dpi=300, bbox_inches="tight")

def task_five(p1,p2,p3,p4,p5,p6,p7,p8,p9,W,trials):
    f_split = np.arange(0,1.03,0.05)
    # Add noise and try to restore
    W += p4 @ p4.T
    restored = np.zeros((4,21,trials))
    for i in range(trials):
        for j, f in enumerate(f_split):
            restored[0,j,i] = add_noise_and_restore(p1,f,W)
            restored[1,j,i] = add_noise_and_restore(p2,f,W)
            restored[2,j,i] = add_noise_and_restore(p3,f,W)
            restored[3,j,i] = add_noise_and_restore(p4,f,W)
    plt.figure()
    plt.plot(f_split, np.mean(restored[0,:,:],axis=1), label="p1")
    plt.plot(f_split, np.mean(restored[1,:,:],axis=1), label="p2")
    plt.plot(f_split, np.mean(restored[2,:,:],axis=1), label="p3")
    plt.plot(f_split, np.mean(restored[3,:,:],axis=1), label="p4")
    plt.xlabel("Noise level")
    plt.ylabel("Restored")
    plt.xticks(np.arange(0,1.1,0.1))
    # plt.yticks([0,1],["False","True"])
    plt.title(f"Average restoration performance for different noise levels and {trials} trials")
    plt.legend()
    plt.savefig("img/restore_4_noisy_patterns.png", dpi=300, bbox_inches="tight")

    # Add noise and try to restore
    W += p5 @ p5.T
    restored = np.zeros((5,21,trials))
    for i in range(trials):
        for j, f in enumerate(f_split):
            restored[0,j,i] = add_noise_and_restore(p1,f,W)
            restored[1,j,i] = add_noise_and_restore(p2,f,W)
            restored[2,j,i] = add_noise_and_restore(p3,f,W)
            restored[3,j,i] = add_noise_and_restore(p4,f,W)
            restored[4,j,i] = add_noise_and_restore(p5,f,W)
    plt.figure()
    plt.plot(f_split, np.mean(restored[0,:,:],axis=1), label="p1")
    plt.plot(f_split, np.mean(restored[1,:,:],axis=1), label="p2")
    plt.plot(f_split, np.mean(restored[2,:,:],axis=1), label="p3")
    plt.plot(f_split, np.mean(restored[3,:,:],axis=1), label="p4")
    plt.plot(f_split, np.mean(restored[4,:,:],axis=1), label="p5")
    plt.xlabel("Noise level")
    plt.ylabel("Restored")
    plt.xticks(np.arange(0,1.1,0.1))
    # plt.yticks([0,1],["False","True"])
    plt.title(f"Average restoration performance for different noise levels and {trials} trials")
    plt.legend()
    plt.savefig("img/restore_5_noisy_patterns.png", dpi=300, bbox_inches="tight")

    # Add noise and try to restore
    W += p6 @ p6.T
    restored = np.zeros((6,21,trials))
    for i in range(trials):
        for j, f in enumerate(f_split):
            restored[0,j,i] = add_noise_and_restore(p1,f,W)
            restored[1,j,i] = add_noise_and_restore(p2,f,W)
            restored[2,j,i] = add_noise_and_restore(p3,f,W)
            restored[3,j,i] = add_noise_and_restore(p4,f,W)
            restored[4,j,i] = add_noise_and_restore(p5,f,W)
            restored[5,j,i] = add_noise_and_restore(p6,f,W)
    plt.figure()
    plt.plot(f_split, np.mean(restored[0,:,:],axis=1), label="p1")
    plt.plot(f_split, np.mean(restored[1,:,:],axis=1), label="p2")
    plt.plot(f_split, np.mean(restored[2,:,:],axis=1), label="p3")
    plt.plot(f_split, np.mean(restored[3,:,:],axis=1), label="p4")
    plt.plot(f_split, np.mean(restored[4,:,:],axis=1), label="p5")
    plt.plot(f_split, np.mean(restored[5,:,:],axis=1), label="p6")
    plt.xlabel("Noise level")
    plt.ylabel("Restored")
    plt.xticks(np.arange(0,1.1,0.1))
    # plt.yticks([0,1],["False","True"])
    plt.title(f"Average restoration performance for different noise levels and {trials} trials")
    plt.legend()
    plt.savefig("img/restore_6_noisy_patterns.png", dpi=300, bbox_inches="tight")

    # Add noise and try to restore
    W += p7 @ p7.T
    restored = np.zeros((7,21,trials))
    for i in range(trials):
        for j, f in enumerate(f_split):
            restored[0,j,i] = add_noise_and_restore(p1,f,W)
            restored[1,j,i] = add_noise_and_restore(p2,f,W)
            restored[2,j,i] = add_noise_and_restore(p3,f,W)
            restored[3,j,i] = add_noise_and_restore(p4,f,W)
            restored[4,j,i] = add_noise_and_restore(p5,f,W)
            restored[5,j,i] = add_noise_and_restore(p6,f,W)
            restored[6,j,i] = add_noise_and_restore(p7,f,W)
    plt.figure()
    plt.plot(f_split, np.mean(restored[0,:,:],axis=1), label="p1")
    plt.plot(f_split, np.mean(restored[1,:,:],axis=1), label="p2")
    plt.plot(f_split, np.mean(restored[2,:,:],axis=1), label="p3")
    plt.plot(f_split, np.mean(restored[3,:,:],axis=1), label="p4")
    plt.plot(f_split, np.mean(restored[4,:,:],axis=1), label="p5")
    plt.plot(f_split, np.mean(restored[5,:,:],axis=1), label="p6")
    plt.plot(f_split, np.mean(restored[6,:,:],axis=1), label="p7")
    plt.xlabel("Noise level")
    plt.ylabel("Restored")
    plt.xticks(np.arange(0,1.1,0.1))
    # plt.yticks([0,1],["False","True"])
    plt.title(f"Average restoration performance for different noise levels and {trials} trials")
    plt.legend()
    plt.savefig("img/restore_7_noisy_patterns.png", dpi=300, bbox_inches="tight")

    # Add noise and try to restore
    W += p8 @ p8.T
    restored = np.zeros((8,21,trials))
    for i in range(trials):
        for j, f in enumerate(f_split):
            restored[0,j,i] = add_noise_and_restore(p1,f,W)
            restored[1,j,i] = add_noise_and_restore(p2,f,W)
            restored[2,j,i] = add_noise_and_restore(p3,f,W)
            restored[3,j,i] = add_noise_and_restore(p4,f,W)
            restored[4,j,i] = add_noise_and_restore(p5,f,W)
            restored[5,j,i] = add_noise_and_restore(p6,f,W)
            restored[6,j,i] = add_noise_and_restore(p7,f,W)
            restored[7,j,i] = add_noise_and_restore(p8,f,W)
    plt.figure()
    plt.plot(f_split, np.mean(restored[0,:,:],axis=1), label="p1")
    plt.plot(f_split, np.mean(restored[1,:,:],axis=1), label="p2")
    plt.plot(f_split, np.mean(restored[2,:,:],axis=1), label="p3")
    plt.plot(f_split, np.mean(restored[3,:,:],axis=1), label="p4")
    plt.plot(f_split, np.mean(restored[4,:,:],axis=1), label="p5")
    plt.plot(f_split, np.mean(restored[5,:,:],axis=1), label="p6")
    plt.plot(f_split, np.mean(restored[6,:,:],axis=1), label="p7")
    plt.plot(f_split, np.mean(restored[7,:,:],axis=1), label="p8")
    plt.xlabel("Noise level")
    plt.ylabel("Restored")
    plt.xticks(np.arange(0,1.1,0.1))
    # plt.yticks([0,1],["False","True"])
    plt.title(f"Average restoration performance for different noise levels and {trials} trials")
    plt.legend()
    plt.savefig("img/restore_8_noisy_patterns.png", dpi=300, bbox_inches="tight")
        
    # Add noise and try to restore
    W += p9 @ p9.T
    restored = np.zeros((9,21,trials))
    for i in range(trials):
        for j, f in enumerate(f_split):
            restored[0,j,i] = add_noise_and_restore(p1,f,W)
            restored[1,j,i] = add_noise_and_restore(p2,f,W)
            restored[2,j,i] = add_noise_and_restore(p3,f,W)
            restored[3,j,i] = add_noise_and_restore(p4,f,W)
            restored[4,j,i] = add_noise_and_restore(p5,f,W)
            restored[5,j,i] = add_noise_and_restore(p6,f,W)
            restored[6,j,i] = add_noise_and_restore(p7,f,W)
            restored[7,j,i] = add_noise_and_restore(p8,f,W)
            restored[8,j,i] = add_noise_and_restore(p9,f,W)
    plt.figure()
    plt.plot(f_split, np.mean(restored[0,:,:],axis=1), label="p1")
    plt.plot(f_split, np.mean(restored[1,:,:],axis=1), label="p2")
    plt.plot(f_split, np.mean(restored[2,:,:],axis=1), label="p3")
    plt.plot(f_split, np.mean(restored[3,:,:],axis=1), label="p4")
    plt.plot(f_split, np.mean(restored[4,:,:],axis=1), label="p5")
    plt.plot(f_split, np.mean(restored[5,:,:],axis=1), label="p6")
    plt.plot(f_split, np.mean(restored[6,:,:],axis=1), label="p7")
    plt.plot(f_split, np.mean(restored[7,:,:],axis=1), label="p8")
    plt.plot(f_split, np.mean(restored[8,:,:],axis=1), label="p9")
    plt.xlabel("Noise level")
    plt.ylabel("Restored")
    plt.xticks(np.arange(0,1.1,0.1))
    # plt.yticks([0,1],["False","True"])
    plt.title(f"Average restoration performance for different noise levels and {trials} trials")
    plt.legend()
    plt.savefig("img/restore_9_noisy_patterns.png", dpi=300, bbox_inches="tight")

if __name__ == '__main__':
    # task_one()

    # Load data
    pict = np.fromfile('data.csv', sep=',').reshape(11,1024)
    p1 = pict[0,:].reshape(1024,1)
    p2 = pict[1,:].reshape(1024,1)
    p3 = pict[2,:].reshape(1024,1)
    p4 = pict[3,:].reshape(1024,1)
    p5 = pict[4,:].reshape(1024,1)
    p6 = pict[5,:].reshape(1024,1)
    p7 = pict[6,:].reshape(1024,1)
    p8 = pict[7,:].reshape(1024,1)
    p9 = pict[8,:].reshape(1024,1)
    p10 = pict[9,:].reshape(1024,1)
    p11 = pict[10,:].reshape(1024,1)

    # Calculate weight matrix for the first three patterns
    W = p1 @ p1.T + p2 @ p2.T + p3 @ p3.T
    assert W.shape == (1024,1024)

    # task_two(p1,p2,p3,p10,p11,W)

    # task_three(p1,p2,p3,p10,p11,W)

    trials = 100

    task_four(p1,p2,p3,W,trials)

    task_five(p1,p2,p3,p4,p5,p6,p7,p8,p9,W,trials)
