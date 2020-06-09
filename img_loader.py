import numpy as np
from PIL import Image
from matplotlib import pyplot

path = "datasets/3/observations-master/experiements/data/with_mask/0-with-mask.jpg"

image = np.asarray(Image.open(path).resize((200, 200)))

#image = np.array(Image.open(path).convert('LA'))

#image = np.mean(image, axis = 2)

new_p = Image.fromarray(image)
new_p = new_p.convert("L")

new_p.save("your_file.png")

pyplot.imshow(new_p)
pyplot.show()

"""
image = image.imread(path)

print(image.dtype, image.shape)

pyplot.imshow(image)
pyplot.show()
"""
