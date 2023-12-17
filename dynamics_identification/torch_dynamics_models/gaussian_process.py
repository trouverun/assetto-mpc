import gpytorch
import torch


class BatchIndependentGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, output_size):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([output_size]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=5, batch_shape=torch.Size([output_size])),
            batch_shape=torch.Size([output_size])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


class GPModel(torch.nn.Module):
    def __init__(self, train_x, train_y, device="cuda:0", train_lr=1e-1, train_epochs=150):
        super().__init__()
        train_x = train_x.to(device)
        train_y = train_y.to(device)
        self.output_size = train_y.shape[1]
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.output_size, has_global_noise=False).to(device)
        self.model = BatchIndependentGPModel(train_x, train_y, self.likelihood, self.output_size).to(device)
        self.likelihood.train()
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=train_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=33, gamma=1e-1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for i in range(train_epochs):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            print("GP epoch %d, loss: %.8f" % (i, loss))

        self.likelihood.eval()
        self.model.eval()

        print(f'Actual outputscale: {self.model.covar_module.outputscale}')
        print(f'Actual covar: {self.model.covar_module.base_kernel.lengthscale}')
        print(f'Task noise: {torch.nn.functional.softplus(self.likelihood.raw_task_noises)}')

    def forward(self, x):
        with gpytorch.settings.fast_pred_var():
            return self.likelihood(self.model(x))