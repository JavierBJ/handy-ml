import random
from typing import Dict, Any, Optional, Callable

from sklearn.metrics import precision_score, recall_score


class OptimizeSatisfyEarlyStopping:
    """
    This class implements a multi-objective early stopping criterion for machine learning algorithms, based on the
    optimize/satisfy principle: optimize a certain metric and at the same time ensure other metrics satisfy a certain
    threshold. It assumes the metrics are scores (higher is better).
    """
    def __init__(
        self,
        optimize_metric: Callable,
        satisfy_metrics: Optional[Dict[Callable, float]] = None,
    ):
        """
        Creates an optimize/satisfy early stopping criterion.

        :param optimize_metric: a function that receives a dictionary of evaluation scores and returns the value of the
        metric to optimize.
        :param satisfy_metrics: a dictionary where the keys are functions, each receives a dictionary of evaluation
        scores and returns the value of a metric to satisfy; and the values are the minimum threshold to be satisfied
        by each metric.
        """
        self.optimize_metric = optimize_metric
        self.satisfy_metrics = {} if satisfy_metrics is None else satisfy_metrics
        self.best_dict = {}

    def evaluate_criteria(self, score_dict: Dict[str, Any]) -> bool:
        """
        Evaluates the state of a training algorithm in a certain step given its dictionary of scores, returning True
        if training must stop considering the criterion defined. This must be called after every iteration because it
        preserves the current state in order to check for improvement in the next call.

        :param score_dict: a dictionary containing the key-value pairs needed to assess the optimize/satisfy criteria.
        :return: True, if all satisfaction metrics are satisfied and the optimization criterion stopped improving from
        the previous evaluation; False otherwise.
        """
        if self._metric_decreases(score_dict) and self._metrics_satisfied():
            return True

        self.best_dict = score_dict
        return False

    def _metric_decreases(self, score_dict: Dict[str, Any]) -> bool:
        if self.best_dict == {}:
            return False
        return self.optimize_metric(score_dict) < self.optimize_metric(self.best_dict)

    def _metrics_satisfied(self) -> bool:
        return all(
            satisfy_metric(self.best_dict) >= value
            for satisfy_metric, value in self.satisfy_metrics.items()
        )


if __name__ == "__main__":
    """
    An execution example. Predictions at each step are calculated randomly.
    """
    MAX_ITER = 9999     # Max iterations in the simulation

    y_true = [0] * 50 + [1] * 50
    criterion = OptimizeSatisfyEarlyStopping(optimize_metric=lambda x: x["recall"],
                                             satisfy_metrics={lambda x: x["precision"]: 0.5})

    for i in range(MAX_ITER):
        y_pred = [random.randint(0, 1) for _ in range(100)]  # Your training step
        score_dict = {"precision": precision_score(y_true, y_pred),
                      "recall": recall_score(y_true, y_pred)}

        print(f"Step {i+1}: {score_dict}")

        if criterion.evaluate_criteria(score_dict):
            break
