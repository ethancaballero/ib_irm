# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import json
import random
import numpy as np
import utils
from torch.autograd import grad


class Model(torch.nn.Module):
    def __init__(self, args, in_features, out_features, bias, task, hparams="default"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.task = task

        # network architecture
        self.network = torch.nn.Linear(in_features, out_features, bias)

        # loss
        if self.task == "regression":
            self.loss = torch.nn.MSELoss()
        else:
            self.loss = torch.nn.BCEWithLogitsLoss()

        # hyper-parameters
        if hparams == "default":
            self.hparams = {k: v[0] for k, v in self.HPARAMS.items()}
        elif hparams == "random":
            self.hparams = {k: v[1] for k, v in self.HPARAMS.items()}
        else:
            self.hparams = json.loads(hparams)

        # callbacks
        self.callbacks = {}
        for key in ["errors"]:
            self.callbacks[key] = {
                "train": [],
                "validation": [],
                "test": []
            }


class ERM(Model):
    def __init__(self, args, in_features, out_features, bias, task, hparams="default"):
        self.HPARAMS = {}
        self.HPARAMS["lr"] = (1e-3, 10**random.uniform(-4, -2))
        self.HPARAMS['wd'] = (0., 10**random.uniform(-6, -2))
        #self.HPARAMS['l1'] = (0., 10**random.uniform(-6, -2))
        self.HPARAMS['l1'] = (0., 0.)

        super().__init__(args, in_features, out_features, bias, task, hparams)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["wd"])

    def fit(self, envs, num_iterations, callback=False):
        x = torch.cat([xe for xe, ye in envs["train"]["envs"]])
        y = torch.cat([ye for xe, ye in envs["train"]["envs"]])

        for epoch in range(num_iterations):
            if self.bias:
                params = torch.cat([[_ for _ in self.network.parameters()][-2].squeeze(), [_ for _ in self.network.parameters()][-1]])
            else:
                params = [_ for _ in self.network.parameters()][-1]
            l1_penalty = torch.norm(params, 1)
            self.optimizer.zero_grad()
            (self.loss(self.network(x), y) + self.hparams["l1"] * l1_penalty).backward()
            self.optimizer.step()

            if callback:
                # compute errors
                utils.compute_errors(self, envs)

    def predict(self, x):
        return self.network(x)


class IB_ERM(Model):
    def __init__(self, args, in_features, out_features, bias, task, hparams="default"):
        self.args = args
        self.HPARAMS = {}
        self.HPARAMS["lr"] = (1e-3, 10**random.uniform(-4, -2))
        self.HPARAMS['wd'] = (0., 10**random.uniform(-6, -2))
        #self.HPARAMS['l1'] = (0., 10**random.uniform(-6, -2))
        self.HPARAMS['l1'] = (0., 0.)
        """
        if args["new_hparam_interval"]:
            self.HPARAMS['ib_lambda'] = (0.9, 1 - 10**random.uniform(-3, 0.))
        else:
            self.HPARAMS['ib_lambda'] = (0.9, 1 - 10**random.uniform(-3, -.3))
        #"""
        self.HPARAMS['ib_lambda'] = (0.1, 1 - 10**random.uniform(-.05, 0.))
        self.HPARAMS['ib_on'] = (True, random.choice([True, False]))

        super().__init__(args, in_features, out_features, bias, task, hparams)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["wd"])

    def fit(self, envs, num_iterations, callback=False):
        x = torch.cat([xe for xe, ye in envs["train"]["envs"]])
        y = torch.cat([ye for xe, ye in envs["train"]["envs"]])

        for epoch in range(num_iterations):
            if self.bias:
                params = torch.cat([[_ for _ in self.network.parameters()][-2].squeeze(), [_ for _ in self.network.parameters()][-1]])
            else:
                params = [_ for _ in self.network.parameters()][-1]
            l1_penalty = torch.norm(params, 1)
            self.optimizer.zero_grad()
            logits = self.network(x)
            loss = self.loss(logits, y) + self.hparams["l1"] * l1_penalty

            if self.hparams['ib_on'] or (not self.args["ib_bool"]):
                loss += self.hparams["ib_lambda"] * logits.var(0).mean()

            loss.backward()
            self.optimizer.step()

            if callback:
                # compute errors
                utils.compute_errors(self, envs)

    def predict(self, x):
        return self.network(x)

class REx(Model):
    """
    Abstract class for REx
    """

    def __init__(
            self, args, in_features, out_features, bias, task, hparams="default", version=1):
        self.HPARAMS = {}
        self.HPARAMS["lr"] = (1e-3, 10**random.uniform(-4, -2))
        self.HPARAMS['wd'] = (0., 10**random.uniform(-6, -2))
        #self.HPARAMS['l1'] = (0., 10**random.uniform(-6, -2))
        self.HPARAMS['l1'] = (0., 0.)
        self.HPARAMS['rex_lambda'] = (0.9, 1 - 10**random.uniform(-3, -.3))

        super().__init__(args, in_features, out_features, bias, task, hparams)
        self.version = version

        self.network = self.IRMLayer(self.network)
        self.net_parameters, self.net_dummies = self.find_parameters(
            self.network)

        self.optimizer = torch.optim.Adam(
            self.net_parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["wd"])

    def find_parameters(self, network):
        """
        Alternative to network.parameters() to separate real parameters
        from dummmies.
        """
        parameters = []
        dummies = []

        for name, param in network.named_parameters():
            if "dummy" in name:
                dummies.append(param)
            else:
                parameters.append(param)
        return parameters, dummies

    class IRMLayer(torch.nn.Module):
        """
        Add a "multiply by one and sum zero" dummy operation to
        any layer. Then you can take gradients with respect these
        dummies. Often applied to Linear and Conv2d layers.
        """

        def __init__(self, layer):
            super().__init__()
            self.layer = layer
            self.dummy_mul = torch.nn.Parameter(torch.Tensor([1.0]))
            self.dummy_sum = torch.nn.Parameter(torch.Tensor([0.0]))

        def forward(self, x):
            return self.layer(x) * self.dummy_mul + self.dummy_sum

    def fit(self, envs, num_iterations, callback=False):
        for epoch in range(num_iterations):
            losses_env = []
            gradients_env = []
            for x, y in envs["train"]["envs"]:
                losses_env.append(self.loss(self.network(x), y))
                gradients_env.append(grad(
                    losses_env[-1], self.net_dummies, create_graph=True))

            # Average loss across envs
            losses_avg = sum(losses_env) / len(losses_env)
            penalty = torch.stack([lo for lo in losses_env]).var()
            """
            gradients_avg = grad(
                losses_avg, self.net_dummies, create_graph=True)

            penalty = 0
            for gradients_this_env in gradients_env:
                for g_env, g_avg in zip(gradients_this_env, gradients_avg):
                    if self.version == 1:
                        penalty += g_env.pow(2).sum()
                    else:
                        raise NotImplementedError
            #"""

            if self.bias:
                params = torch.cat([[_ for _ in self.network.parameters()][-2].squeeze(), [_ for _ in self.network.parameters()][-1]])
            else:
                params = [_ for _ in self.network.parameters()][-1]
            l1_penalty = torch.norm(params, 1)

            obj = (1 - self.hparams["rex_lambda"]) * losses_avg
            obj += self.hparams["rex_lambda"] * penalty
            obj += self.hparams["l1"] * l1_penalty

            self.optimizer.zero_grad()
            obj.backward()
            self.optimizer.step()

            if callback:
                # compute errors
                utils.compute_errors(self, envs)

    def predict(self, x):
        return self.network(x)


class IRM(Model):
    """
    Abstract class for IRM
    """

    def __init__(
            self, args, in_features, out_features, bias, task, hparams="default", version=1):
        self.args = args
        self.HPARAMS = {}
        self.HPARAMS["lr"] = (1e-3, 10**random.uniform(-4, -2))
        self.HPARAMS['wd'] = (0., 10**random.uniform(-6, -2))
        #self.HPARAMS['l1'] = (0., 10**random.uniform(-6, -2))
        self.HPARAMS['l1'] = (0., 0.)
        """
        if args["new_hparam_interval"]:
            self.HPARAMS['irm_lambda'] = (0.9, 1 - 10**random.uniform(-3, 0.))
        else:
            self.HPARAMS['irm_lambda'] = (0.9, 1 - 10**random.uniform(self.args.irm_lambda_l, self.args.irm_lambda_r))
        #"""
        self.HPARAMS['irm_lambda'] = (0.9, 1 - 10**random.uniform(self.args.irm_lambda_l, self.args.irm_lambda_r))

        super().__init__(args, in_features, out_features, bias, task, hparams)
        self.version = version

        self.network = self.IRMLayer(self.network)
        self.net_parameters, self.net_dummies = self.find_parameters(
            self.network)

        self.optimizer = torch.optim.Adam(
            self.net_parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["wd"])

    def find_parameters(self, network):
        """
        Alternative to network.parameters() to separate real parameters
        from dummmies.
        """
        parameters = []
        dummies = []

        for name, param in network.named_parameters():
            if "dummy" in name:
                dummies.append(param)
            else:
                parameters.append(param)
        return parameters, dummies

    class IRMLayer(torch.nn.Module):
        """
        Add a "multiply by one and sum zero" dummy operation to
        any layer. Then you can take gradients with respect these
        dummies. Often applied to Linear and Conv2d layers.
        """

        def __init__(self, layer):
            super().__init__()
            self.layer = layer
            self.dummy_mul = torch.nn.Parameter(torch.Tensor([1.0]))
            self.dummy_sum = torch.nn.Parameter(torch.Tensor([0.0]))

        def forward(self, x):
            return self.layer(x) * self.dummy_mul + self.dummy_sum

    def fit(self, envs, num_iterations, callback=False):
        for epoch in range(num_iterations):
            losses_env = []
            gradients_env = []
            for x, y in envs["train"]["envs"]:
                losses_env.append(self.loss(self.network(x), y))
                gradients_env.append(grad(
                    losses_env[-1], self.net_dummies, create_graph=True))

            # Average loss across envs
            losses_avg = sum(losses_env) / len(losses_env)
            gradients_avg = grad(
                losses_avg, self.net_dummies, create_graph=True)

            penalty = 0
            for gradients_this_env in gradients_env:
                for g_env, g_avg in zip(gradients_this_env, gradients_avg):
                    if self.version == 1:
                        penalty += g_env.pow(2).sum()
                    else:
                        raise NotImplementedError

            if self.bias:
                params = torch.cat([[_ for _ in self.network.parameters()][-2].squeeze(), [_ for _ in self.network.parameters()][-1]])
            else:
                params = [_ for _ in self.network.parameters()][-1]
            l1_penalty = torch.norm(params, 1)

            """
            if self.args["new_hparam_interval"]:
                penalty /= len(envs["train"]["envs"])
            #"""

            obj = (1 - self.hparams["irm_lambda"]) * losses_avg
            obj += self.hparams["irm_lambda"] * penalty
            obj += self.hparams["l1"] * l1_penalty

            self.optimizer.zero_grad()
            obj.backward()
            self.optimizer.step()

            if callback:
                # compute errors
                utils.compute_errors(self, envs)

    def predict(self, x):
        return self.network(x)


class IRMv1(IRM):
    """
    IRMv1 with penalty \sum_e \| \nabla_{w|w=1} \mR_e (\Phi \circ \vec{w}) \|_2^2
    From https://arxiv.org/abs/1907.02893v1 
    """

    def __init__(self, args, in_features, out_features, bias, task, hparams="default"):
        super().__init__(args, in_features, out_features, bias, task, hparams, version=1)


class IB_IRM(Model):
    """
    Abstract class for IRM
    """

    def __init__(
            self, args, in_features, out_features, bias, task, hparams="default", version=1):
        self.args = args
        self.HPARAMS = {}
        self.HPARAMS["lr"] = (1e-3, 10**random.uniform(-4, -2))
        self.HPARAMS['wd'] = (0., 10**random.uniform(-6, -2))
        #self.HPARAMS['l1'] = (0., 10**random.uniform(-6, -2))
        self.HPARAMS['l1'] = (0., 0.)
        """
        if args["new_hparam_interval"]:
            self.HPARAMS['irm_lambda'] = (0.9, 1 - 10**random.uniform(-3, 0.))
            self.HPARAMS['ib_lambda'] = (0.9, 1 - 10**random.uniform(-3, 0.))
        else:
            self.HPARAMS['irm_lambda'] = (0.9, 1 - 10**random.uniform(self.args.irm_lambda_l, self.args.irm_lambda_r))
            self.HPARAMS['ib_lambda'] = (0.9, 1 - 10**random.uniform(-3, -.3))
        #"""
        self.HPARAMS['irm_lambda'] = (0.9, 1 - 10**random.uniform(self.args.irm_lambda_l, self.args.irm_lambda_r))
        self.HPARAMS['ib_lambda'] = (0.1, 1 - 10**random.uniform(-.05, 0.))
        self.HPARAMS['ib_on'] = (True, random.choice([True, False]))

        super().__init__(args, in_features, out_features, bias, task, hparams)
        self.version = version

        self.network = self.IRMLayer(self.network)
        self.net_parameters, self.net_dummies = self.find_parameters(
            self.network)

        self.optimizer = torch.optim.Adam(
            self.net_parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["wd"])

    def find_parameters(self, network):
        """
        Alternative to network.parameters() to separate real parameters
        from dummmies.
        """
        parameters = []
        dummies = []

        for name, param in network.named_parameters():
            if "dummy" in name:
                dummies.append(param)
            else:
                parameters.append(param)
        return parameters, dummies

    class IRMLayer(torch.nn.Module):
        """
        Add a "multiply by one and sum zero" dummy operation to
        any layer. Then you can take gradients with respect these
        dummies. Often applied to Linear and Conv2d layers.
        """

        def __init__(self, layer):
            super().__init__()
            self.layer = layer
            self.dummy_mul = torch.nn.Parameter(torch.Tensor([1.0]))
            self.dummy_sum = torch.nn.Parameter(torch.Tensor([0.0]))

        def forward(self, x):
            return self.layer(x) * self.dummy_mul + self.dummy_sum

    def fit(self, envs, num_iterations, callback=False):
        for epoch in range(num_iterations):
            losses_env = []
            gradients_env = []
            logits_env = []
            for x, y in envs["train"]["envs"]:
                logits = self.network(x)
                logits_env.append(logits)
                losses_env.append(self.loss(self.network(x), y))
                gradients_env.append(grad(
                    losses_env[-1], self.net_dummies, create_graph=True))

            # penalty per env
            logit_penalty = torch.stack(logits_env).var(1).mean()

            # single
            #logit_penalty = torch.cat(logits_env).var(0).mean()

            # Average loss across envs
            losses_avg = sum(losses_env) / len(losses_env)
            gradients_avg = grad(
                losses_avg, self.net_dummies, create_graph=True)

            penalty = 0
            for gradients_this_env in gradients_env:
                for g_env, g_avg in zip(gradients_this_env, gradients_avg):
                    if self.version == 1:
                        penalty += g_env.pow(2).sum()
                    else:
                        raise NotImplementedError

            if self.bias:
                params = torch.cat([[_ for _ in self.network.parameters()][-2].squeeze(), [_ for _ in self.network.parameters()][-1]])
            else:
                params = [_ for _ in self.network.parameters()][-1]
            l1_penalty = torch.norm(params, 1)

            if self.args["new_hparam_interval"]:
                penalty /= len(envs["train"]["envs"])

            obj = (1 - self.hparams["irm_lambda"]) * losses_avg
            obj += self.hparams["irm_lambda"] * penalty

            if self.hparams['ib_on'] or (not self.args["ib_bool"]):
                obj += self.hparams["ib_lambda"] * logit_penalty

            obj += self.hparams["l1"] * l1_penalty

            self.optimizer.zero_grad()
            obj.backward()
            self.optimizer.step()

            if callback:
                # compute errors
                utils.compute_errors(self, envs)

    def predict(self, x):
        return self.network(x)

class AndMask(Model):
    """
    AndMask: Masks the grqdients features for which 
    the gradients signs across envs disagree more than 'tau'
    From https://arxiv.org/abs/2009.00329
    """

    def __init__(self, args, in_features, out_features, bias, task, hparams="default"):
        self.HPARAMS = {}
        self.HPARAMS["lr"] = (1e-3, 10**random.uniform(-4, 0))
        self.HPARAMS['wd'] = (0., 10**random.uniform(-5, 0))
        ##self.HPARAMS['l1'] = (0., 10**random.uniform(-6, -2))
        self.HPARAMS['l1'] = (0., 0.)
        self.HPARAMS["tau"] = (0.9, random.uniform(0.8, 1))
        super().__init__(args, in_features, out_features, bias, task, hparams)

    def fit(self, envs, num_iterations, callback=False):
        for epoch in range(num_iterations):
            """
            if self.bias:
                params = torch.cat([[_ for _ in self.network.parameters()][-2].squeeze(), [_ for _ in self.network.parameters()][-1]])
            else:
                params = [_ for _ in self.network.parameters()][-1]
            l1_penalty = torch.norm(params, 1)
            losses = [self.loss(self.network(x), y) + self.hparams["l1"] * l1_penalty
            #"""
            losses = [self.loss(self.network(x), y)
                      for x, y in envs["train"]["envs"]]
            self.mask_step(
                losses, list(self.parameters()),
                tau=self.hparams["tau"],
                wd=self.hparams["wd"],
                lr=self.hparams["lr"]
            )

            if callback:
                # compute errors
                utils.compute_errors(self, envs)

    def predict(self, x):
        return self.network(x)

    def mask_step(self, losses, parameters, tau=0.9, wd=0.1, lr=1e-3):
        with torch.no_grad():
            gradients = []
            for loss in losses:
                gradients.append(list(torch.autograd.grad(loss, parameters)))
                gradients[-1][0] = gradients[-1][0] / gradients[-1][0].norm()

            for ge_all, parameter in zip(zip(*gradients), parameters):
                # environment-wise gradients (num_environments x num_parameters)
                ge_cat = torch.cat(ge_all)

                # treat scalar parameters also as matrices
                if ge_cat.dim() == 1:
                    ge_cat = ge_cat.view(len(losses), -1)

                # creates a mask with zeros on weak features
                mask = (torch.abs(torch.sign(ge_cat).sum(0))
                        > len(losses) * tau).int()

                # mean gradient (1 x num_parameters)
                g_mean = ge_cat.mean(0, keepdim=True)

                # apply the mask
                g_masked = mask * g_mean

                # update
                parameter.data = parameter.data - lr * g_masked \
                    - lr * wd * parameter.data


class IB_AndMask(Model):
    """
    AndMask: Masks the grqdients features for which 
    the gradients signs across envs disagree more than 'tau'
    From https://arxiv.org/abs/2009.00329
    """

    def __init__(self, args, in_features, out_features, bias, task, hparams="default"):
        self.HPARAMS = {}
        self.HPARAMS["lr"] = (1e-3, 10**random.uniform(-4, 0))
        self.HPARAMS['wd'] = (0., 10**random.uniform(-5, 0))
        ##self.HPARAMS['l1'] = (0., 10**random.uniform(-6, -2))
        self.HPARAMS['l1'] = (0., 0.)
        self.HPARAMS["tau"] = (0.9, random.uniform(0.8, 1))
        self.HPARAMS['ib_lambda'] = (0.9, 1 - 10**random.uniform(-3, -.3))
        super().__init__(args, in_features, out_features, bias, task, hparams)

    def fit(self, envs, num_iterations, callback=False):
        for epoch in range(num_iterations):
            logits = []
            losses = []
            for x, y in envs["train"]["envs"]:
                """
                if self.bias:
                    params = torch.cat([[_ for _ in self.network.parameters()][-2].squeeze(), [_ for _ in self.network.parameters()][-1]])
                else:
                    params = [_ for _ in self.network.parameters()][-1]
                l1_penalty = torch.norm(params, 1)
                #"""
                logit = self.network(x)
                #logits.append(logit)
                #loss = self.loss(logit, y) + self.hparams["ib_lambda"] * logit.var(0).mean() + self.hparams["l1"] * l1_penalty
                loss = self.loss(logit, y) + self.hparams["ib_lambda"] * logit.var(0).mean()
                losses.append(loss)
            self.mask_step(
                losses, list(self.parameters()),
                tau=self.hparams["tau"],
                wd=self.hparams["wd"],
                lr=self.hparams["lr"]
            )

            if callback:
                # compute errors
                utils.compute_errors(self, envs)

    def predict(self, x):
        return self.network(x)

    def mask_step(self, losses, parameters, tau=0.9, wd=0.1, lr=1e-3):
        with torch.no_grad():
            gradients = []
            for loss in losses:
                gradients.append(list(torch.autograd.grad(loss, parameters)))
                gradients[-1][0] = gradients[-1][0] / gradients[-1][0].norm()

            for ge_all, parameter in zip(zip(*gradients), parameters):
                # environment-wise gradients (num_environments x num_parameters)
                ge_cat = torch.cat(ge_all)

                # treat scalar parameters also as matrices
                if ge_cat.dim() == 1:
                    ge_cat = ge_cat.view(len(losses), -1)

                # creates a mask with zeros on weak features
                mask = (torch.abs(torch.sign(ge_cat).sum(0))
                        > len(losses) * tau).int()

                # mean gradient (1 x num_parameters)
                g_mean = ge_cat.mean(0, keepdim=True)

                # apply the mask
                g_masked = mask * g_mean

                # update
                parameter.data = parameter.data - lr * g_masked \
                    - lr * wd * parameter.data


class IGA(Model):
    """
    Inter-environmental Gradient Alignment
    From https://arxiv.org/abs/2008.01883v2
    """

    def __init__(self, args, in_features, out_features, bias, task, hparams="default"):
        self.HPARAMS = {}
        self.HPARAMS["lr"] = (1e-3, 10**random.uniform(-4, -2))
        self.HPARAMS['wd'] = (0., 10**random.uniform(-6, -2))
        #self.HPARAMS['l1'] = (0., 10**random.uniform(-6, -2))
        self.HPARAMS['l1'] = (0., 0.)
        self.HPARAMS['penalty'] = (1000, 10**random.uniform(1, 5))
        super().__init__(args, in_features, out_features, bias, task, hparams)

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["wd"])

    def fit(self, envs, num_iterations, callback=False):
        for epoch in range(num_iterations):
            if self.bias:
                params = torch.cat([[_ for _ in self.network.parameters()][-2].squeeze(), [_ for _ in self.network.parameters()][-1]])
            else:
                params = [_ for _ in self.network.parameters()][-1]
            l1_penalty = torch.norm(params, 1)
            losses = [self.loss(self.network(x), y)
                      for x, y in envs["train"]["envs"]]
            gradients = [
                grad(loss, self.parameters(), create_graph=True)
                for loss in losses
            ]
            # average loss and gradients
            avg_loss = sum(losses) / len(losses)
            avg_gradient = grad(avg_loss, self.parameters(), create_graph=True)

            # compute trace penalty
            penalty_value = 0
            for gradient in gradients:
                for gradient_i, avg_grad_i in zip(gradient, avg_gradient):
                    penalty_value += (gradient_i - avg_grad_i).pow(2).sum()

            self.optimizer.zero_grad()
            (avg_loss + self.hparams['penalty'] * penalty_value + self.hparams["l1"] * l1_penalty).backward()
            self.optimizer.step()

            if callback:
                # compute errors
                utils.compute_errors(self, envs)

    def predict(self, x):
        return self.network(x)


class MLP(torch.nn.Module):
    def __init__(self, args, in_features, out_features):
        super(MLP, self).__init__()
        #hidden_dim = 128
        hidden_dim = 32
        self.lin1 = torch.nn.Linear(in_features, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = torch.nn.Linear(hidden_dim, out_features)
        for lin in [self.lin1, self.lin2, self.lin3]:
            torch.nn.init.xavier_uniform_(lin.weight)
            torch.nn.init.zeros_(lin.bias)
        #self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)

    def forward(self, x):
        #import pdb; pdb.set_trace()
        #x = x.view(x.shape[0], self.input_size)
        x1 = x = self.lin1(x)
        x2 = x = torch.nn.functional.elu(x)
        x3 = x = self.lin2(x)
        x4 = x = torch.nn.functional.elu(x)
        x5 = x = self.lin3(x)
        #out = self._main(out)
        #import pdb; pdb.set_trace()
        return x, x1, x2, x3, x4, x5

class IB_IRM_NN(Model):
    """
    Abstract class for IRM
    """
    def __init__(
            self, args, in_features, out_features, bias, task, hparams="default", version=1):
        self.args = args
        self.HPARAMS = {}
        self.HPARAMS["lr"] = (1e-3, 10**random.uniform(-4, -2))
        self.HPARAMS['wd'] = (0., 10**random.uniform(-6, -2))
        #self.HPARAMS['l1'] = (0., 10**random.uniform(-6, -2))
        self.HPARAMS['l1'] = (0., 0.)
        """
        if args["new_hparam_interval"]:
            self.HPARAMS['irm_lambda'] = (0.9, 1 - 10**random.uniform(-3, 0.))
            self.HPARAMS['ib_lambda'] = (0.9, 1 - 10**random.uniform(-3, 0.))
        else:
            self.HPARAMS['irm_lambda'] = (0.9, 1 - 10**random.uniform(self.args.irm_lambda_l, self.args.irm_lambda_r))
            self.HPARAMS['ib_lambda'] = (0.9, 1 - 10**random.uniform(-3, -.3))
        """
        self.HPARAMS['irm_lambda'] = (0.9, 1 - 10**random.uniform(self.args.irm_lambda_l, self.args.irm_lambda_r))
        self.HPARAMS['ib_lambda'] = (0.1, 1 - 10**random.uniform(-.05, 0.))
        super().__init__(args, in_features, out_features, bias, task, hparams)
        self.version = version
        self.network = MLP(in_features, out_features)
        self.network = self.IRMLayer(self.network)
        self.net_parameters, self.net_dummies = self.find_parameters(
            self.network)
        self.optimizer = torch.optim.Adam(
            self.net_parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["wd"])

    def find_parameters(self, network):
        """
        Alternative to network.parameters() to separate real parameters
        from dummmies.
        """
        parameters = []
        dummies = []
        for name, param in network.named_parameters():
            if "dummy" in name:
                dummies.append(param)
            else:
                parameters.append(param)
        return parameters, dummies

    class IRMLayer(torch.nn.Module):
        """
        Add a "multiply by one and sum zero" dummy operation to
        any layer. Then you can take gradients with respect these
        dummies. Often applied to Linear and Conv2d layers.
        """
        def __init__(self, layer):
            super().__init__()
            self.layer = layer
            self.dummy_mul = torch.nn.Parameter(torch.Tensor([1.0]))
            self.dummy_sum = torch.nn.Parameter(torch.Tensor([0.0]))

        def forward(self, x):
            x, x1, x2, x3, x4, x5 = self.layer(x)
            return x * self.dummy_mul + self.dummy_sum, x1, x2, x3, x4, x5

    def fit(self, envs, num_iterations, callback=False):
        for epoch in range(num_iterations):
            losses_env = []
            gradients_env = []
            logits_env = []
            acts1_env = []
            acts2_env = []
            acts3_env = []
            acts4_env = []
            acts5_env = []
            for x, y in envs["train"]["envs"]:
                logits, a1, a2, a3, a4, a5 = self.network(x)
                logits_env.append(logits)
                acts1_env.append(a1)
                acts2_env.append(a2)
                acts3_env.append(a3)
                acts4_env.append(a4)
                acts5_env.append(a5)
                losses_env.append(self.loss(logits, y))
                gradients_env.append(grad(
                    losses_env[-1], self.net_dummies, create_graph=True))
            # penalty per env
            logit_penalty = torch.stack(logits_env).var(1).mean()
            #import pdb; pdb.set_trace()
            acts1_penalty = torch.stack(acts1_env).var(1).mean()
            acts2_penalty = torch.stack(acts2_env).var(1).mean()
            acts3_penalty = torch.stack(acts3_env).var(1).mean()
            acts4_penalty = torch.stack(acts4_env).var(1).mean()
            acts5_penalty = torch.stack(acts5_env).var(1).mean()
            # single
            #logit_penalty = torch.cat(logits_env).var(0).mean()
            # Average loss across envs
            losses_avg = sum(losses_env) / len(losses_env)
            gradients_avg = grad(
                losses_avg, self.net_dummies, create_graph=True)
            penalty = 0
            for gradients_this_env in gradients_env:
                for g_env, g_avg in zip(gradients_this_env, gradients_avg):
                    if self.version == 1:
                        penalty += g_env.pow(2).sum()
                    else:
                        raise NotImplementedError
            if self.bias:
                params = torch.cat([[_ for _ in self.network.parameters()][-2].squeeze(), [_ for _ in self.network.parameters()][-1]])
            else:
                params = [_ for _ in self.network.parameters()][-1]
            l1_penalty = torch.norm(params, 1)

            if self.args["new_hparam_interval"]:
                penalty /= len(envs["train"]["envs"])

            obj = (1 - self.hparams["irm_lambda"]) * losses_avg
            obj += self.hparams["irm_lambda"] * penalty
            #obj += self.hparams["ib_lambda"] * logit_penalty
            obj += self.hparams["ib_lambda"] * (logit_penalty + acts1_penalty + acts2_penalty + acts3_penalty + acts4_penalty + acts5_penalty)
            #obj += self.hparams["ib_lambda"] * (logit_penalty + acts1_penalty + acts2_penalty + acts3_penalty + acts4_penalty + acts5_penalty) / 6
            #obj += self.hparams["ib_lambda"] * (logit_penalty + (acts1_penalty + acts2_penalty + acts3_penalty + acts4_penalty + acts5_penalty) / (acts1_penalty + acts2_penalty + acts3_penalty + acts4_penalty + acts5_penalty).detach())
            obj += self.hparams["l1"] * l1_penalty
            self.optimizer.zero_grad()
            obj.backward()
            self.optimizer.step()
            if callback:
                # compute errors
                utils.compute_errors_nonlinear(self, envs)

    def predict(self, x):
        return self.network(x)


MODELS = {
    "ERM": ERM,
    "REx": REx,
    "IB_ERM": IB_ERM,
    "IRMv1": IRMv1,
    "IB_IRM": IB_IRM,
    "ANDMask": AndMask,
    "IB_ANDMask": IB_AndMask,
    "IGA": IGA,
    "Oracle": ERM,

    "IB_IRM_NN": IB_IRM_NN
}
