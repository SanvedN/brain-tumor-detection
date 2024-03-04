import visualkeras
from keras.models import load_model
from keras.utils.vis_utils import plot_model

model = load_model('cnn_model.h5')

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
