import imageio

images = []
for e in range(21):
    images.append(imageio.imread("rbm_results/500_nodes/rf.epoch%02d.png"%e))
imageio.mimsave(f'rbm_results/500_nodes/rf500.gif', images, duration=0.25)