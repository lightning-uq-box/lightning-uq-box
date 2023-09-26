from sklearn.preprocessing import StandardScaler
import torch
import os
import numpy as np
import abc

class Dataset(object):
    def __init__(self, seed, n_samples):
        self.seed = seed
        self.n_samples = n_samples
        self.eps_samples = None
    
    @abc.abstractmethod
    def create_train_test_dataset(self):
        pass

    def create_noises(self, noise_dict):
        """
        Ref: https://pytorch.org/docs/stable/distributions.html
        :param noise_dict: {"noise_type": "norm", "loc": 0., "scale": 1.}
        """
        print("Create noises using the following parameters:")
        print(noise_dict)
        noise_type = noise_dict.get("noise_type", "norm")
        if noise_type == "t":
            dist = torch.distributions.studentT.StudentT(df=noise_dict.get("df", 10.), loc=noise_dict.get("loc", 0.0),
                                                         scale=noise_dict.get("scale", 1.0))
        elif noise_type == "unif":
            dist = torch.distributions.uniform.Uniform(low=noise_dict.get("low", 0.), high=noise_dict.get("high", 1.))
        elif noise_type == "Chi2":
            dist = torch.distributions.chi2.Chi2(df=noise_dict.get("df", 10.))
        elif noise_type == "Laplace":
            dist = torch.distributions.laplace.Laplace(loc=noise_dict.get("loc", 0.), scale=noise_dict.get("scale", 1.))
        else:  # noise_type == "norm"
            dist = torch.distributions.normal.Normal(loc=noise_dict.get("loc", 0.), scale=noise_dict.get("scale", 1.))

        self.eps_samples = dist.sample((self.n_samples, 1))

class DatasetWithOneX(Dataset):
    def __init__(self, n_samples, seed, x_dict, noise_dict, normalize_x=False, normalize_y=False):
        super(DatasetWithOneX, self).__init__(seed=seed, n_samples=n_samples)
        self.x_dict = x_dict
        self.x_samples = self.sample_x(self.x_dict)
        self.dim_x = self.x_samples.shape[1]  # dimension of data input
        self.y = self.create_y_from_one_x(noise_dict=noise_dict)
        self.dim_y = self.y.shape[1]  # dimension of regression output
        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None
        self.train_n_samples, self.test_n_samples = None, None
        self.train_dataset, self.test_dataset = None, None
        self.normalize_x = normalize_x
        self.normalize_y = normalize_y
        self.scaler_x, self.scaler_y = None, None

    def sample_x(self, x_dict):
        """
        :param x_dict: {"dist_type": "unif", "low": 0., "high": 1.}
        """
        print("Create x using the following parameters:")
        print(x_dict)
        dist_type = x_dict.get("dist_type", "unif")
        if dist_type == "norm":
            dist = torch.distributions.normal.Normal(loc=x_dict.get("loc", 0.), scale=x_dict.get("scale", 1.))
        else:
            dist = torch.distributions.uniform.Uniform(low=x_dict.get("low", 0.), high=x_dict.get("high", 1.))

        return dist.sample((self.n_samples, 1))

    def create_y_from_one_x(self, noise_dict):
        if self.eps_samples is None:
            n_samples_temp = self.n_samples
            if type(noise_dict.get("scale", 1.)) == torch.Tensor:
                self.n_samples = 1
            self.create_noises(noise_dict)
            if self.n_samples == 1:
                self.eps_samples = self.eps_samples[0]
                self.n_samples = n_samples_temp

    def create_train_test_dataset(self, train_ratio=0.8):
    
        data_idx = np.arange(self.n_samples)
        np.random.shuffle(data_idx)
        train_size = int(self.n_samples * train_ratio)
        self.x_train, self.y_train, self.x_test, self.y_test = \
            self.x_samples[data_idx[:train_size]], self.y[data_idx[:train_size]], \
            self.x_samples[data_idx[train_size:]], self.y[data_idx[train_size:]]
        self.train_n_samples = self.x_train.shape[0]
        self.test_n_samples = self.x_test.shape[0]
        # standardize x and y if needed
        if self.normalize_x:
            self.normalize_train_test_x()
        if self.normalize_y:
            self.normalize_train_test_y()
        self.train_dataset = torch.cat((self.x_train, self.y_train), dim=1)
        # sort x for easier plotting purpose during test time
        if self.dim_x == 1:
            sorted_idx = torch.argsort(self.x_test, dim=0).squeeze()
            self.x_test = self.x_test[sorted_idx]
            self.y_test = self.y_test[sorted_idx]
        self.test_dataset = torch.cat((self.x_test, self.y_test), dim=1)

    def normalize_train_test_x(self):
        self.scaler_x = StandardScaler(with_mean=True, with_std=True)
        self.x_train = torch.from_numpy(
            self.scaler_x.fit_transform(self.x_train).astype(np.float32))
        self.x_test = torch.from_numpy(
            self.scaler_x.transform(self.x_test).astype(np.float32))

    def normalize_train_test_y(self):
        self.scaler_y = StandardScaler(with_mean=True, with_std=True)
        self.y_train = torch.from_numpy(
            self.scaler_y.fit_transform(self.y_train).astype(np.float32))
        self.y_test = torch.from_numpy(
            self.scaler_y.transform(self.y_test).astype(np.float32))
        
    
class LogLinearDatasetWithOneX(DatasetWithOneX):
    def __init__(self, a, b, n_samples, seed, x_dict, noise_dict, normalize_x=False, normalize_y=False):
        self.a = a
        self.b = b
        super(LogLinearDatasetWithOneX, self).__init__(
            n_samples=n_samples, seed=seed, x_dict=x_dict, noise_dict=noise_dict,
            normalize_x=normalize_x, normalize_y=normalize_y)

    def create_y_from_one_x(self, noise_dict):
        # log(y) = a + b * x + eps
        super().create_y_from_one_x(noise_dict)
        logy = self.a + self.b * self.x_samples + self.eps_samples
        return logy.exp()
    

class CircleDatasetWithOneX(DatasetWithOneX):
    def __init__(self, r, n_samples, seed, x_dict, noise_dict, normalize_x=False, normalize_y=False):
        self.r = r
        self.theta = torch.rand((n_samples, 1)) * 2 * np.pi
        x_dict.update(r=r)
        super(CircleDatasetWithOneX, self).__init__(
            n_samples=n_samples, seed=seed, x_dict=x_dict, noise_dict=noise_dict,
            normalize_x=normalize_x, normalize_y=normalize_y
        )

    def create_y_from_one_x(self, noise_dict):
        super().create_y_from_one_x(noise_dict)
        r_samples = self.r + self.eps_samples
        self.x_samples, y = r_samples * torch.cos(self.theta), r_samples * torch.sin(self.theta)
        self.dim_x = self.x_samples.shape[1]
        return y


class SinusoidDatasetWithOneX(DatasetWithOneX):
    """
    Both "Snelson" Dataset and "OAT-1D" Dataset are sinusoid curve.
    Currently use function from Fig. 4 of Mixture Density Networks.
    """

    def __init__(
        self, n_samples, seed, x_dict, noise_dict, normalize_x=False, normalize_y=False
    ):
        super(SinusoidDatasetWithOneX, self).__init__(
            n_samples=n_samples,
            seed=seed,
            x_dict=x_dict,
            noise_dict=noise_dict,
            normalize_x=normalize_x,
            normalize_y=normalize_y,
        )

    def create_y_from_one_x(self, noise_dict):
        # y = x + 0.3 * sin(2 * pi * x) + eps
        super().create_y_from_one_x(noise_dict)
        y = (
            self.x_samples
            + 0.3 * torch.sin(2 * np.pi * self.x_samples)
            + self.eps_samples
        )
        return y

    def invert_xy(self):
        """
        Swap x and y to have a one-to-many mapping, like in MDN paper.
        """
        temp_x = self.y
        temp_dim_x = self.dim_y
        self.y = self.x_samples
        self.dim_y = self.dim_x
        self.x_samples = temp_x
        self.dim_x = temp_dim_x
