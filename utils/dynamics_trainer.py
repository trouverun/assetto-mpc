import torch
from evotorch import Problem
from evotorch.algorithms import CMAES
from evotorch.logging import StdOutLogger


def median_loss(outputs, targets):
    return torch.mean(torch.sum(torch.abs(outputs - targets), axis=1))


class EvoTorchWrapper:
    def __init__(self, data_x, data_y, torch_model):
        self.data_x = data_x.cuda()
        self.data_y = data_y.cuda()
        self.torch_model = torch_model.cuda()
        self.variable_scalers = self.torch_model.get_variable_scalers().cuda()
        self.output_weights = self.torch_model.get_output_weights().cuda()
        self.loss_fn = torch.nn.MSELoss(reduction="mean")

    def solve(self, popsize=100, std=0.1, iters=500):
        problem = Problem(
            "min",
            self._evaluate,
            initial_bounds=(-1, 1),
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            solution_length=len(self.variable_scalers),
            # Evaluation is vectorized
            vectorized=False,
            # Higher-than-default precision
            dtype=torch.float64,
        )

        searcher = CMAES(problem, popsize=popsize, stdev_init=std,
                         center_init=self.torch_model.extract_params().cuda() / self.variable_scalers)
        logger = StdOutLogger(searcher, interval=50)

        searcher.run(iters)
        best_discovered_solution = searcher.status["center"]
        self.torch_model.setup_params(best_discovered_solution.clone() * self.variable_scalers)

    def _evaluate(self, x):
        self.torch_model.setup_params(x.clone() * self.variable_scalers)
        with torch.no_grad():
            outputs = self.torch_model(self.data_x)
            loss = self.loss_fn(self.output_weights*outputs, self.output_weights*self.data_y)
            loss += self.torch_model.get_constraint_costs()

        return loss
