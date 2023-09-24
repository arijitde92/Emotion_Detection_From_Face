from torchview import draw_graph
from Model import Net
import matplotlib.pyplot as plt

model = Net(dropout=0.2, num_classes=7)
print("Creating Model")
input_size = (512, 3, 48, 48)
model_graph = draw_graph(model, input_size=input_size, device='meta')
model_graph.visual_graph.render('model_architecture', view=True)
