import torch.nn.functional as F


def convert_to_simplex(label, simplex_value, num_classes):
    return 2 * simplex_value * F.one_hot(label, num_classes) - simplex_value