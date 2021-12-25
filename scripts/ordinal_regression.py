"""
Ordinal regression is a special type of classification where the class labels are ordered. A straightforward way to
model the order relation between classes is to encode the classes in the following way: given N classes, each class
is represented as a binary vector of length N-1, where component i is 1 if the sample is greater than class i. For
example, for 3-class ordinal regression, class 0 = [0, 0], 1 = [1, 0], 2 = [1, 1].

This modelling of ordinal regression is great because it only requires slight adaptation on the inputs and outputs of
the model, but any classification model still works. Although this method is popular in
"""
from typing import List, Union, Dict, Any

import numpy as np
import torch


def encode_ordinal(labels: List[int], num_labels: int) -> torch.Tensor:
    """
    Given a list of class labels, transforms them into an ordinal tensor.

    :param labels: list of class labels to transform.
    :param num_labels: number of class labels. It is not inferred directly from parameter `labels` because this function
    could be called in batches.
    :return: an ordinal tensor from the labels passed.
    """
    return torch.tensor(
        [
            ([1] * label + [0] * (num_labels - label - 1))
            for label in labels
        ]
    ).float()


def decode_ordinal(logits: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
    """
    Given some predicted logits in the ordinal regression space, transforms them to predictions in the original space
    of class labels.

    :param logits: a tensor/array of logits in the ordinal regression space.
    :return: a dictionary containing the predictions in the original space of class labels, and the predicted
    probabilities for each output in the ordinal space.
    """
    probabilities = torch.nn.Sigmoid()(torch.Tensor(logits))
    binary_predictions = (probabilities > 0.5).int().numpy()  # Calculate predicted outputs binarizing probabilities
    predictions = np.sum(binary_predictions, axis=1).tolist()  # Calculate the predicted label from predicted outputs

    return {
        "prediction": predictions,
        "probability": probabilities,
    }


if __name__ == "__main__":
    """
    An example of encoding labels to ordinal and decoding ordinal logits.
    """
    labels = [2, 0, 1]
    ordinal_labels = encode_ordinal(labels, 3)

    print("Real labels:")
    print(labels)
    print("Ordinal labels:")
    print(ordinal_labels)
    print()

    logits = np.array([[-0.05, -0.61],
                       [0.02, -0.2],
                       [0.25, 0.10]])
    output_dict = decode_ordinal(logits)

    print("Ordinal logits:")
    print(logits)
    print("Real predictions:")
    print(output_dict["prediction"])
