import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

def update_neighbourhood(n,e,epochs,factor=5):
    # return round(n-(e+1)*n/(epochs-1))
    return round(n*np.exp(-factor*(e+1)/epochs))

def update_learning_rate(eta,e,epochs):
    return eta*np.exp(-(e+1)/epochs)

def order_animals(eta_init=0.2,epochs=50,n_init=50):
    props = np.array(pd.read_csv('data_lab2/animals.dat', sep=",", header=None)).reshape(32,84)
    animalnames = np.loadtxt('data_lab2/animalnames.txt',dtype=str)
    weights = np.random.uniform(low=0.,high=1.,size=(100,84))

    n = n_init
    eta = eta_init
    for e in range(epochs):
        print(f"Epoch {e}, neighbourhood {n}")
        for animal in props:
            animal = animal.reshape(1,84)
            winner = np.argmin(np.linalg.norm(weights - animal,axis=1))
            lower = winner - n if winner - n >= 0 else 0
            upper = winner + n + 1 if winner + n + 1 <= 100 else 100
            weights[lower:upper,:] = weights[lower:upper,:] + eta * (animal - weights[lower:upper,:])
        n = update_neighbourhood(n_init,e,epochs)
        eta = update_learning_rate(eta_init,e,epochs)
    
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

def find_update_set2(winner,n):
    ids = []
    for id in range(winner-n,winner+n+1):
        if id < 0:
            ids.append(id+20)
        elif id > 19:
            ids.append(id-20)
        else:
            ids.append(id)
    return ids

def cyclic_tour(k,eta_init=0.2,epochs=30,n_init=4):
    cities = np.array(pd.read_csv('data_lab2/cities.dat', sep=",", header=None)).reshape(10,2)
    weights = np.random.uniform(low=0.,high=1.,size=(20,2))

    n = n_init
    eta = eta_init
    fig,ax = plt.subplots()
    for e in range(epochs):
        ax.clear()
        ax.scatter(cities[:,0], cities[:,1],marker='x',label="Cities")
        ax.plot(weights[:,0],weights[:,1],'ko-',label="Tour")
        ax.plot([weights[-1,0],weights[0,0]],[weights[-1,1],weights[0,1]],'ko-')
        ax.legend()
        ax.set_title(f"Cylic Tour via SOM, epoch={e}")
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        fig.savefig(f"img/cyclic_tour_{k}_{e}.png",dpi=300)

        print(f"Epoch {e}, neighbourhood {n}")
        for city in cities:
            city = city.reshape(1,2)
            winner = np.argmin(np.linalg.norm(weights - city,axis=1))
            id_update = find_update_set2(winner,n)
            weights[id_update,:] = weights[id_update,:] + eta * (city - weights[id_update,:])
        n = update_neighbourhood(n_init,e,epochs,factor=4)
        eta = update_learning_rate(eta_init,e,epochs)

    ax.clear()
    ax.scatter(cities[:,0], cities[:,1],marker='x',label="Cities")
    ax.plot(weights[:,0],weights[:,1],'ko-',label="Tour")
    ax.plot([weights[-1,0],weights[0,0]],[weights[-1,1],weights[0,1]],'ko-')
    ax.legend()
    ax.set_title(f"Cylic Tour via SOM, epoch={e+1}")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    fig.savefig(f"img/cyclic_tour_{k}_{e+1}.png",dpi=300)
    save_images_as_gif(epochs,k)
    # plt.show()

def save_images_as_gif(epochs,k):
    import imageio
    images = []
    for e in range(epochs+1):
        images.append(imageio.imread(f"img/cyclic_tour_{k}_{e}.png"))
    imageio.mimsave(f'img/cyclic_tour_{k}.gif', images, duration=0.25)

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

def gender_name(value):
    if value == 0:
        return "Male"
    elif value == 1:
        return "Female"

def party_name(value):
    if value == 0:
        return "No party"
    elif value == 1:
        return "M"
    elif value == 2:
        return "FP"
    elif value == 3:
        return "S"
    elif value == 4:
        return "V"
    elif value == 5:
        return "MP"
    elif value == 6:
        return "KD"
    elif value == 7:
        return "C"

def vote_clustering(eta_init=0.2,epochs=100,n_init=5):
    votes = np.array(pd.read_csv('data_lab2/votes.dat', sep=",", header=None)).reshape(349,31)
    party = np.array(pd.read_csv('data_lab2/mpparty.dat', sep="\n", header=None)).reshape(349,1) # Coding: 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
    gender = np.array(pd.read_csv('data_lab2/mpsex.dat', sep="\n", header=None)).reshape(349,1) # 0 male, 1 female
    district = np.array(pd.read_csv('data_lab2/mpdistrict.dat', sep="\n", header=None)).reshape(349,1)
    # names = np.loadtxt('data_lab2/mpnames.txt',dtype=str)

    weights = np.random.uniform(low=0.,high=1.,size=(10,10,31))

    n = n_init
    eta = eta_init
    for e in range(epochs):
        print(f"Epoch {e}, neighbourhood {n}")
        for vote in votes:
            vote = vote.reshape(1,1,31)
            diff = np.linalg.norm(weights - vote,axis=2)
            winner = np.unravel_index(np.argmin(diff, axis=None), diff.shape)
            ids = find_neighbourhood_pairs(winner,n)
            for id in ids:
                weights[id[0],id[1],:] = weights[id[0],id[1],:] + eta * (vote - weights[id[0],id[1],:])
        n = update_neighbourhood(n_init,e,epochs,6)
        eta = update_learning_rate(eta_init,e,epochs)
    
    map = np.zeros(shape=(10,10,349),dtype=float)
    for i, vote in enumerate(votes):
        vote = vote.reshape(1,1,31)
        map[:,:,i] = np.linalg.norm(weights - vote,axis=2)
        winners = np.argmin(map,axis=2)

    winners_party = map_to(winners,party)
    winners_gender = map_to(winners,gender)
    winners_district = map_to(winners,district)

    # male/female cluster
    map_male = np.zeros(shape=(10,10),dtype=float)
    map_female = np.zeros(shape=(10,10),dtype=float)
    for i, vote in enumerate(votes):
        vote = vote.reshape(1,1,31)
        if gender[i] == 0:
            map_male += np.linalg.norm(weights - vote,axis=2)
        elif gender[i] == 1:
            map_female += np.linalg.norm(weights - vote,axis=2)
    map_male = map_male/186
    map_female = map_female/163

    # party cluster
    map_no = np.zeros(shape=(10,10),dtype=float)
    map_m = np.zeros(shape=(10,10),dtype=float)
    map_fp = np.zeros(shape=(10,10),dtype=float)
    map_s = np.zeros(shape=(10,10),dtype=float)
    map_v = np.zeros(shape=(10,10),dtype=float)
    map_mp = np.zeros(shape=(10,10),dtype=float)
    map_kd = np.zeros(shape=(10,10),dtype=float)
    map_c = np.zeros(shape=(10,10),dtype=float)

    for i, vote in enumerate(votes):
        vote = vote.reshape(1,1,31)
        if party[i] == 0:
            map_no += np.linalg.norm(weights - vote,axis=2)
        elif party[i] == 1:
            map_m += np.linalg.norm(weights - vote,axis=2)
        elif party[i] == 2:
            map_fp += np.linalg.norm(weights - vote,axis=2)
        elif party[i] == 3:
            map_s += np.linalg.norm(weights - vote,axis=2)
        elif party[i] == 4:
            map_v += np.linalg.norm(weights - vote,axis=2)
        elif party[i] == 5:
            map_mp += np.linalg.norm(weights - vote,axis=2)
        elif party[i] == 6:
            map_kd += np.linalg.norm(weights - vote,axis=2)
        elif party[i] == 7:
            map_c += np.linalg.norm(weights - vote,axis=2)
    map_m = map_m/55
    map_fp = map_fp/48
    map_s = map_s/144
    map_v = map_v/29
    map_mp = map_mp/17
    map_kd = map_kd/33
    map_c = map_c/22


    fig0, ax0 = plt.subplots()
    im0 = ax0.imshow(winners_party,cmap="tab10")
    ax0.title.set_text('MPs mapped to parties')
    values = np.unique(winners_party.ravel())
    colors = [im0.cmap(im0.norm(value)) for value in values]
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label=party_name(values[i]) ) for i in range(len(values)) ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    fig0.savefig("img/vote_mapped_to_parties.png",dpi=300)
    
    fig1, ax1 = plt.subplots()
    im1 = ax1.imshow(winners_gender,vmin=-0.25, vmax=1.25,cmap="RdBu")
    ax1.title.set_text('MPs mapped to gender')
    values = np.unique(winners_gender.ravel())
    colors = [im1.cmap(im1.norm(value)) for value in values]
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label=gender_name(values[i]) ) for i in range(len(values)) ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    fig1.savefig("img/vote_mapped_to_gender.png",dpi=300)

    fig2, ax2 = plt.subplots()
    im2 = ax2.imshow(winners_district,cmap="cividis")
    ax2.title.set_text('MPs mapped to districts')
    values = np.unique(winners_district.ravel())
    colors = [im2.cmap(im2.norm(value)) for value in values]
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label=str(values[i])) for i in range(len(values)) ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    fig2.savefig("img/vote_mapped_to_district.png",dpi=300)

    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].imshow(map_male,cmap='Blues')
    ax[0].title.set_text('Male cluster')
    ax[1].imshow(map_female,cmap='Reds')
    ax[1].title.set_text('Female cluster')
    fig.savefig("img/vote_male_female.png",dpi=300)
    # plt.show()

    fig, ax = plt.subplots()
    ax.imshow(map_no,cmap='Greys')
    ax.title.set_text('No party cluster')
    fig.savefig("img/vote_party/vote_no_party.png",dpi=300)
    ax.clear()
    ax.imshow(map_m,cmap='Blues')
    ax.title.set_text('M cluster')
    fig.savefig("img/vote_party/vote_m.png",dpi=300)
    ax.clear()
    ax.imshow(map_fp,cmap='Oranges')
    ax.title.set_text('FP cluster')
    fig.savefig("img/vote_party/vote_fp.png",dpi=300)
    ax.clear()
    ax.imshow(map_s,cmap='Reds')
    ax.title.set_text('SD cluster')
    fig.savefig("img/vote_party/vote_s.png",dpi=300)
    ax.clear()
    ax.imshow(map_v,cmap='Purples')
    ax.title.set_text('V cluster')
    fig.savefig("img/vote_party/vote_v.png",dpi=300)
    ax.clear()
    ax.imshow(map_mp,cmap='Greens')
    ax.title.set_text('MP cluster')
    fig.savefig("img/vote_party/vote_mp.png",dpi=300)
    ax.clear()
    ax.imshow(map_kd,cmap='BuGn')
    ax.title.set_text('KD cluster')
    fig.savefig("img/vote_party/vote_kd.png",dpi=300)
    ax.clear()
    ax.imshow(map_c,cmap='YlOrBr')
    ax.title.set_text('C cluster')
    fig.savefig("img/vote_party/vote_c.png",dpi=300)

if __name__ == "__main__":
    np.random.seed(40)
    eta = 0.2

    # Task 4.1
    # order_animals()

    # # Task 4.2
    # for k in range(1):
    #     cyclic_tour(k)
    
    # Task 4.3
    vote_clustering()