import torch
import torch.nn.functional as F
from collections.abc import Mapping


def scale(inputs, scale_value):
    return inputs / scale_value

def convert_to_simplex(label, simplex_value, num_classes):
    return 2 * simplex_value * F.one_hot(label, num_classes) - simplex_value

def logits_projection(logits, simplex_value):
    probs = F.softmax(logits, dim=-1)
    pred_labels = torch.argmax(probs, dim=-1)
    return convert_to_simplex(pred_labels, simplex_value, logits.size(-1))

def self_condition_preds(self_condition, logits, logits_projection=None):
    if self_condition in ["logits", "logits_addition", "logits_mean", "logits_max", "logits_multiply"]:
        previous_pred = logits.detach()
    elif self_condition in ["logits_with_projection", "logits_with_projection_addition"]:
        previous_pred = logits_projection(logits.detach())
    else:
        assert NotImplementedError(f"{self_condition} is not implemented.")
    return previous_pred

def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_detach(t) for k, t in tensors.items()})
    return tensors.detach()