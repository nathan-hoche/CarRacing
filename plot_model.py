# print model
from networks import CNN, FullyConnected, SelectiveCNN, SelectiveKMNN

from keras.utils.vis_utils import plot_model

plot_model(CNN.brain("test").model, to_file='img/CNN.png', show_shapes=True, show_layer_activations=True)
plot_model(FullyConnected.brain("test").model, to_file='img/FullyConnected.png', show_shapes=True, show_layer_activations=True)
plot_model(SelectiveCNN.brain("test").model, to_file='img/SelectiveCNN.png', show_shapes=True, show_layer_activations=True)
plot_model(SelectiveKMNN.brain("test").model, to_file='img/SelectiveKMNN.png', show_shapes=True, show_layer_activations=True)


