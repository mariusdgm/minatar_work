"""Implementation credits: Florin"""


import math
import torch
import torch.nn as nn

__all__ = ["ReDo", "apply_redo_parametrization"]


class ReDo:
    def __init__(
        self,
        module,
        inbound,
        outbound,
        tau=0.005,
        beta=0.1,
        selection_option=None,
        verbose=False,
    ) -> None:
        self.module = module
        self.inbound = inbound
        self.outbound = outbound
        self.tau = tau
        self.beta = beta
        self.running_avg = None
        self.selection_option = selection_option

    def __call__(self, m, _, output):
        """Keep a running average of abs(activation)"""
        dims = [0] if output.ndim == 2 else [0, -2, -1]
        x = output.detach().abs().mean(dim=dims)
        m.running_avg = (1 - self.beta) * m.running_avg + self.beta * x
        m.running_avg_cnt = m.running_avg_cnt + 1

    def get_score(self):
        return self.module.running_avg / self.module.running_avg.sum()

    def redo(self):
        """Computes a mask of activations that are dormant based on the score
        and uses it to reset the inbound and outbound weights of both
        nn.Linear and nn.Conv2d layers.
        """
        score = self.get_score()
        idxs = (score <= self.tau).nonzero().flatten()

        if self.selection_option in ["intersect", "union"]:
            smallest_weights_idx = torch.argsort(
                self.inbound.weight.data.abs().mean(dim=1)
            )
            smallest_weights_idx = smallest_weights_idx[: len(idxs)]

            if self.selection_option == "intersect":
                idxs = torch.tensor(
                    list(set(idxs.tolist()) & set(smallest_weights_idx.tolist())),
                    device=idxs.device,
                )
            elif self.selection_option == "union":
                idxs = torch.tensor(
                    list(set(idxs.tolist()) | set(smallest_weights_idx.tolist())),
                    device=idxs.device,
                )

        # reinitialize inbound weights
        w, bias = self._reinit()

        self.inbound.weight.data[idxs] = w[idxs]
        self.inbound.bias.data[idxs] = bias[idxs]

        # set outbound weights to zero
        self.outbound.weight.data[:, idxs] = 0

        # reset the running average for reseted neurons
        with torch.no_grad():
            # Detach and clone the tensor to break from any computation graph
            running_avg_updated = self.module.running_avg.detach().clone()
            running_avg_updated[idxs] = 0
            self.module.running_avg = running_avg_updated

        return idxs

    def _reinit(self):
        w = self.inbound.weight.data.clone()
        bias = self.inbound.bias.data.clone() if self.inbound.bias is not None else None
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        if self.inbound.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(bias, -bound, bound)
        return w, bias

    def get_ratio(self):
        """Return fraction of dormant neurons."""
        score = self.get_score()
        mask = score <= self.tau
        return mask.sum() / mask.nelement()

    def get_avg_running_avg(self):
        return self.module.running_avg.mean()

    @staticmethod
    def hook(module, inbound, outbound, tau=0.025, beta=0.1, selection_option=None):
        assert isinstance(module, nn.ReLU), "ReDo applies on activation layers."
        assert isinstance(inbound, (nn.Linear, nn.Conv2d)), "Inbound type not right."
        assert isinstance(outbound, (nn.Linear, nn.Conv2d)), "Outbound type not right."

        # register activation running average
        for attr in ("out_features", "out_channels"):
            act_shape = getattr(inbound, attr, None)
            if act_shape is not None:
                break

        device = inbound.weight.device
        module.register_buffer("running_avg", torch.zeros(act_shape, device=device))
        module.register_buffer("running_avg_cnt", torch.zeros(1, device=device))

        # register hook
        fn = ReDo(module, inbound, outbound, tau=tau, beta=beta)
        module.register_forward_hook(fn)
        return fn


def apply_redo_parametrization(
    net, tau=0.025, beta=0.1, selection_option=None, verbose=False
):
    """Assumes the modules are properly ordered."""
    supported_layers = (nn.ReLU, nn.LayerNorm, nn.Linear, nn.Conv2d)
    layers = [(k, v) for k, v in net.named_modules() if isinstance(v, supported_layers)]
    hndlrs = []
    ratios = []
    scores = []
    for i, (_, module) in enumerate(layers):
        if isinstance(module, nn.ReLU):
            inbound, outbound = layers[i - 1], layers[i + 1]
            hook = ReDo.hook(
                module,
                inbound[1],
                outbound[1],
                tau=tau,
                beta=beta,
                selection_option=selection_option,
            )
            hndlrs.append(hook)
            ratios.append((module, hook))
            scores.append((module, hook))
            if verbose:
                print(f"Hooking {inbound[1]} -> {outbound[1]}")

    # monkey-patch the estimator **instance** by bounding the method below it.
    # This way we can access the handlers.
    def get_dormant_ratios(self):
        ratios = [h.get_ratio().item() for h in hndlrs]
        mus = [h.get_avg_running_avg().item() for h in hndlrs]
        return ratios, mus

    def get_dormant_scores(self):
        scores = [h.get_score() for h in hndlrs]
        return scores

    def apply_redo(self):
        reset_indexes = [h.redo() for h in hndlrs]
        return reset_indexes

    net.get_dormant_ratios = get_dormant_ratios.__get__(net)
    net.get_dormant_scores = get_dormant_scores.__get__(net)
    net.apply_redo = apply_redo.__get__(net)

    return net


def _test_register():
    net = nn.Sequential(
        nn.Conv2d(1, 8, 3, 1, 1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * 5**2, 6),
        nn.ReLU(),
        nn.Linear(6, 6),
        nn.ReLU(),
        nn.Linear(6, 2),
    )

    net = apply_redo_parametrization(net, tau=0.1)

    D = 5
    for i in range(100):
        x = torch.rand((32, 1, D, D))
        net(x)
        if i % 10 == 0:
            print(i, net.get_dormant_ratios())


if __name__ == "__main__":
    _test_register()
    net = nn.Sequential(
        nn.Conv2d(1, 8, 3, 1, 1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * 5**2, 6),
        nn.ReLU(),
        nn.Linear(6, 6),
        nn.ReLU(),
        nn.Linear(6, 2),
    )

    net = apply_redo_parametrization(net, tau=0.1)

    D = 5
    for i in range(100):
        x = torch.rand((32, 1, D, D))
        net(x)
        if i % 10 == 0:
            print(i, net.get_dormant_ratios())

if __name__ == "__main__":
    _test_register()
