import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def update_neighbourhood(n):
    if n > 20:
        return n - 4
    elif n > 5:
        return n - 2
    elif n > 0:
        return n - 1
    else:
        return 0

def order_animals(eta=0.2,epochs=20):
    props = np.array(pd.read_csv('data_lab2/animals.dat', sep=",", header=None)).reshape(32,84)
    animalnames = np.loadtxt('data_lab2/animalnames.txt',dtype=str)
    weights = np.random.uniform(low=0.,high=1.,size=(100,84))

    n = 50
    for e in range(epochs):
        print(f"Epoch {e}, neighbourhood {n}")
        for animal in props:
            animal = animal.reshape(1,84)
            winner = np.argmin(np.linalg.norm(weights - animal,axis=1))
            lower = winner - n if winner - n >= 0 else 0
            upper = winner + n + 1 if winner + n + 1 <= 100 else 100
            weights[lower:upper,:] = weights[lower:upper,:] + eta * (animal - weights[lower:upper,:])
        # n = update_neighbourhood(n)
        n = round(50-(e+1)*50/19)
    
    pos = np.zeros(32,dtype=int)
    for i, animal in enumerate(props):
        pos[i] = np.argmin(np.linalg.norm(weights - animal.reshape(1,84),axis=1))
    for j in np.argsort(pos):
        print(pos[j],animalnames[j])

def find_update_set(winner,n):
    ids = []
    for id in range(winner-n,winner+n+1):
        if id < 0:
            ids.append(id+10)
        elif id > 9:
            ids.append(id-10)
        else:
            ids.append(id)
    return ids

def cyclic_tour(k,eta=0.2,epochs=20):
    cities = np.array(pd.read_csv('data_lab2/cities.dat', sep=",", header=None)).reshape(10,2)
    weights = np.random.uniform(low=0.,high=1.,size=(10,2))

    n = 2
    for e in range(epochs):
        print(f"Epoch {e}, neighbourhood {n}")
        for city in cities:
            city = city.reshape(1,2)
            winner = np.argmin(np.linalg.norm(weights - city,axis=1))
            id_update = find_update_set(winner,n)
            weights[id_update,:] = weights[id_update,:] + eta * (city - weights[id_update,:])
        n = round(2-(e+1)*2/19)

    fig,ax = plt.subplots()
    ax.scatter(cities[:,0], cities[:,1],marker='x',label="Cities")
    ax.plot(weights[:,0],weights[:,1],'ko-',label="Tour")
    ax.plot([weights[-1,0],weights[0,0]],[weights[-1,1],weights[0,1]],'ko-')
    ax.legend()
    ax.set_title(f"Cylic Tour via SOM, eta={eta}, epochs={epochs}")
    fig.savefig(f"img/cyclic_tour_{k}.png",dpi=300)
    # plt.show()

def find_neighbourhood_pairs(winner,n):
    ids = []
    for i in range(10):
        for j in range(10):
            if abs(winner[0]-i) + abs(winner[1]-j) <= n:
                ids.append([i,j])
    return ids

def map_to(winners,category):
    w = np.zeros_like(winners)
    for i in range(10):
        for j in range(10):
            w[i,j] = category[winners[i,j],0]
    return w

def vote_clustering(eta=0.9,epochs=20):
    votes = np.array(pd.read_csv('data_lab2/votes.dat', sep=",", header=None)).reshape(349,31)
    party = np.array(pd.read_csv('data_lab2/mpparty.dat', sep="\n", header=None)).reshape(349,1) # Coding: 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
    gender = np.array(pd.read_csv('data_lab2/mpsex.dat', sep="\n", header=None)).reshape(349,1) # 0 male, 1 female
    district = np.array(pd.read_csv('data_lab2/mpdistrict.dat', sep="\n", header=None)).reshape(349,1)
    # names = np.loadtxt('data_lab2/mpnames.txt',dtype=str)

    weights = np.random.uniform(low=0.,high=1.,size=(10,10,31))

    n = 5
    for e in range(epochs):
        print(f"Epoch {e}, neighbourhood {n}")
        for vote in votes:
            vote = vote.reshape(1,1,31)
            diff = np.linalg.norm(weights - vote,axis=2)
            winner = np.unravel_index(np.argmin(diff, axis=None), diff.shape)
            ids = find_neighbourhood_pairs(winner,n)
            for id in ids:
                weights[id[0],id[1],:] = weights[id[0],id[1],:] + eta * (vote - weights[id[0],id[1],:])
        n = round(5-(e+1)*5/(epochs-1))
        eta = eta/1.5
    
    map = np.zeros(shape=(10,10,349),dtype=float)
    for i, vote in enumerate(votes):
        vote = vote.reshape(1,1,31)
        map[:,:,i] = np.linalg.norm(weights - vote,axis=2)
        winners = np.argmax(map,axis=2)

    winners_party = map_to(winners,party)
    winners_gender = map_to(winners,gender)
    winners_district = map_to(winners,district)

    fig, ax = plt.subplots(1,3)
    ax[0].imshow(winners_party)
    ax[0].title.set_text('MPs mapped to parties')
    ax[1].imshow(winners_gender)
    ax[1].title.set_text('MPs mapped to gender')
    ax[2].imshow(winners_district)
    ax[2].title.set_text('MPs mapped to districts')
    plt.show()


if __name__ == "__main__":
    np.random.seed(40)
    eta = 0.2

    # Task 4.1
    # order_animals()

    # # Task 4.2
    # for k in range(5):
    #     cyclic_tour(k)
    
    # Task 4.3
    vote_clustering()
    