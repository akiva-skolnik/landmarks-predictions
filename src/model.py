import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 50, dropout: float = 0.2) -> None:
        super().__init__()

        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        layers = []
        x, y, z = 3, 224, 224
        # 6 layers of Conv2d, BatchNorm2d, ReLU and MaxPool2d
        n_layers = 6
        for i in range(n_layers):
            out_channels = 2 ** (i + 4)  # 16, 32, 64, 128, 256, 512
            in_channels = 3 if i == 0 else out_channels // 2
            # Input tensor = (x, y, z), conv2d doubles x (for i>0), while MaxPool2d halves y and z.
            # So, the output tensor is (16 * 2^5, y / 2^6, z / 2^6)
            # We start with (3, 224, 224) and end with (512, 4, 4)
            layers.extend([
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels, kernel_size=3, padding=1),  # padding=1 to keep the size
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            x = out_channels
            y //= 2
            z //= 2

        self.layers = layers
        self.features = nn.Sequential(*layers)

        in_features = x * y * z  # 512 * 3 * 3 = 4608
        hidden_nodes = in_features // 9  # 512
        logger.info(f"Number of input features to the classifier: {in_features}, hidden nodes: {hidden_nodes}")
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_features, out_features=hidden_nodes),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, num_classes)
        )
        logger.info(f"Model has {sum(p.numel() for p in self.parameters() if p.requires_grad): ,} parameters")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # process the input tensor through the feature extractor,
        # the pooling and the final linear layers
        x = self.features(x)
        x = self.classifier(x)
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):
    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
