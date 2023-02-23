from util import *
from rbm import RestrictedBoltzmannMachine 
from dbn import DeepBeliefNet

if __name__ == "__main__":

    image_size = [28,28]
    n_train = 60000
    batch_size = 20
    n_iterations = n_train//batch_size
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=n_train, n_test=10000)

    # ''' restricted boltzmann machine '''
    
    epochs = 20
    hidden_sizes = [200,300,400,500]
    hidden_sizes = [500]
    
    # loss_hidden_sizes = np.zeros((4,epochs+1))
    # for i, hidden_size in enumerate(hidden_sizes):
    #     print ("\nStarting a Restricted Boltzmann Machine...")
    #     rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
    #                                     ndim_hidden=hidden_size,
    #                                     is_bottom=True,
    #                                     image_size=image_size,
    #                                     is_top=False,
    #                                     n_labels=10,
    #                                     batch_size=batch_size
    #     )
        
    #     print (f"\nTrain RBM for {epochs} epochs...")
    #     loss_hidden_sizes[i,:] = rbm.cd1(visible_trainset=train_imgs, n_iterations=n_iterations, epochs=epochs)
    #     pause = input("Press any key to continue...")

    # plot_recon_loss(loss_hidden_sizes,hidden_sizes)
    
    ''' deep- belief net '''

    print ("\nStarting a Deep Belief Net..")
    
    dbn = DeepBeliefNet(sizes={"vis":image_size[0]*image_size[1], "hid":500, "pen":500, "top":2000, "lbl":10},
                        image_size=image_size,
                        n_labels=10,
                        batch_size=batch_size
    )
    
    ''' greedy layer-wise training '''

    dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=n_iterations, epochs=epochs)

    #dbn.recognize(train_imgs, train_lbls)
    
    #dbn.recognize(test_imgs, test_lbls)

    for digit in range(10):
        digit_1hot = np.zeros(shape=(1,10))
        digit_1hot[0,digit] = 1
        dbn.generate(digit_1hot, name="rbms")

    # ''' fine-tune wake-sleep training '''

    # dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10000)

    # dbn.recognize(train_imgs, train_lbls)
    
    # dbn.recognize(test_imgs, test_lbls)
    
    # for digit in range(10):
    #     digit_1hot = np.zeros(shape=(1,10))
    #     digit_1hot[0,digit] = 1
    #     dbn.generate(digit_1hot, name="dbn")
