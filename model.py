import torch
import torch.nn as nn


class DQN(nn.Module):

    def __init__(self, width, input_dim, output_dim, path=None, checkpoint=1):
        super(DQN, self).__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._width = width

        self.model_definition()

        if path is not None:
            print(path)
            # load the model from a saved checkpoint
            self.layers = torch.load(path + str(checkpoint))

    def model_definition(self):

        """
        Define the neural network for tye DQN agent.
        """

        self.layers = nn.Sequential(
            nn.Linear(self._input_dim, self._width),
            nn.ReLU(),
            nn.Linear(self._width, self._width),
            nn.ReLU(),
            nn.Linear(self._width, self._width),
            nn.ReLU(),
            nn.Linear(self._width, self._width),
            nn.ReLU(),
            nn.Linear(self._width, self._width),
            nn.ReLU(),
            nn.Linear(self._width, self._output_dim)
        )

    def forward(self, x):
        """
        Execute the forward pass through the neural network
        """
        return self.layers(x)

