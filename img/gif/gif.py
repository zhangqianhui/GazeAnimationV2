import imageio, glob
images = []

base_path = './gif12/'
image_path = base_path + '/*.jpg'
list_file = []
for file in glob.glob(image_path):
        list_file.append(file)

list_file.sort()

for filename in list_file:
    images.append(imageio.imread(filename))

imageio.mimsave(base_path + '/age.gif', images, duration=0.8)