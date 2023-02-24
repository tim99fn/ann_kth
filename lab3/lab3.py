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

    task_three(p1,p2,p3,p10,p11,W)
