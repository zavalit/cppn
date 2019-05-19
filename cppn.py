import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse





def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight)


class NN(nn.Module):

    def __init__(self, activation=nn.Tanh, num_neurons=16, num_layers=10):
        super(NN, self).__init__()
        layers = [nn.Linear(2, num_neurons, bias=True), activation()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(num_neurons, num_neurons, bias=False), activation()]
        layers += [nn.Linear(num_neurons, 3, bias=False), nn.Sigmoid()]
        self.layers = nn.Sequential(*layers)

    def forward(self, pixel):
        return self.layers(pixel)


def plot_colors(colors, fig_size=4):
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(colors, interpolation='nearest', vmin=0, vmax=1)


def save_colors(colors):
    filename = str(np.random.randint(100000)) + ".png"
    print("save to file %s" % filename)
    plt.imsave(filename, colors)

def main(neurons, layers, size, png):
    print("create image of %s neurons %s layers of size %sx%s" % (neurons, layers, size, size))
    net = NN(num_neurons=neurons, num_layers=layers)
    net.apply(init_normal)
    x = np.arange(0, size, 1)
    y = np.arange(0, size, 1)
    colors = np.zeros((size, size, 2))
    for i in x:
        for j in y:
            colors[i][j] = np.array([float(i)/size - 0.5, float(j)/size + 0.5])
    colors = colors.reshape((size*size, 2))
    img = net(torch.tensor(colors).type(torch.FloatTensor)).detach().numpy()
    img = img.reshape(size, size, 3)
    if png:
        save_colors(img)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CPPN demo')
    parser.add_argument('--png', action='store_true', help='save pictures in png formt')
    parser.add_argument('-n', '--num_neurons', default=16, help='number of neurons')
    parser.add_argument('-l', '--num_layers', default=10, help='number of layers')
    parser.add_argument('-s', '--size', default=128, help='image side size')
    args = parser.parse_args()
    main(int(args.num_neurons), int(args.num_layers), int(args.size), args.png)
