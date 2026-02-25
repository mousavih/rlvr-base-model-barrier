import torch


def plot_cdf(p_sorteds, all_steps, filename="cdf.pdf"):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # Use Times New Roman and enable LaTeX rendering
    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"""
    \usepackage{bm}
    \usepackage{amsmath}
    \usepackage{amssymb}
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    norm = mpl.colors.Normalize(vmin=min(all_steps), vmax=max(all_steps))
    cmap = mpl.cm.viridis
    for p_sorted, steps in zip(p_sorteds, all_steps):
        n = p_sorted.numel()
        y = torch.linspace(0, 1, n)
        ax.plot(p_sorted.numpy(), y.numpy(), color=cmap(norm(steps)), label=steps.item(), linewidth=2.5)
    ax.set_xlabel(r"$\alpha$", fontsize=24)
    ax.set_ylabel(r"Likelihood CDF - $\mathbb{P}[q(\bm{y}^*|\bm{x}) \leq \alpha]$", fontsize=24)
    ax.tick_params(axis="both", labelsize=20)
    ax.grid(True)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def plot_alpha_tail(p_sorteds, all_steps, filename="alpha_tail.pdf"):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # Use Times New Roman and enable LaTeX rendering
    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"""
    \usepackage{bm}
    \usepackage{amsmath}
    \usepackage{amssymb}
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    norm = mpl.colors.Normalize(vmin=min(all_steps), vmax=max(all_steps))
    cmap = mpl.cm.viridis
    for p_sorted, steps in zip(p_sorteds, all_steps):
        n = p_sorted.numel()
        y = torch.linspace(0, 1, n)
        tail = 1.0 - y
        ax.plot(
            p_sorted.numpy(),
            (p_sorted * tail).numpy(),
            color=cmap(norm(steps)),
            label=steps.item(),
            linewidth=2.5,
        )
    ax.set_xlabel(r"$\alpha$", fontsize=24)
    ax.set_ylabel(r"$\alpha \, \mathbb{P}[q(\bm{y}^*|\bm{x}) > \alpha]$", fontsize=24)
    ax.tick_params(axis="both", labelsize=20)
    ax.grid(True)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, shrink=0.9)
    cbar.set_label(r"Steps", fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def plot_quantile(p_sorteds, all_steps, filename="quantile.pdf", eps=1e-3):
    """Plot inverse CDF (quantile function) for one or more models.

    Args:
        p_sorteds: list of sorted p(y|x) tensors
        all_steps: tensor/list of step labels for coloring (same length as p_sorteds)
        eps: threshold for drawing vertical line on the largest-step curve
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # Use Times New Roman and enable LaTeX rendering
    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"""
    \usepackage{bm}
    \usepackage{amsmath}
    \usepackage{amssymb}
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    norm = mpl.colors.Normalize(vmin=min(all_steps), vmax=max(all_steps))
    cmap = mpl.cm.viridis
    for p_sorted, steps in zip(p_sorteds, all_steps):
        n = p_sorted.numel()
        y = torch.linspace(0, 1, n)

        # Quantile function: map quantile -> value
        ax.plot(y.numpy(), p_sorted.numpy(), color=cmap(norm(steps)), linewidth=2.5)

    # Vertical line at first eps-crossing for the largest step curve
    if len(p_sorteds) > 0:
        step_values = [s.item() if torch.is_tensor(s) else float(s) for s in all_steps]
        max_idx = int(torch.tensor(step_values).argmax().item())
        p_max = p_sorteds[max_idx].cpu()
        n = p_max.numel()
        y = torch.linspace(0, 1, n)
        above = torch.nonzero(p_max > eps, as_tuple=False)
        if above.numel() > 0:
            q_at_eps = y[int(above[0].item())].item()
            ax.axvline(x=q_at_eps, linestyle="--", color="red", label="The Barrier", linewidth=2.0)

    ax.set_xlabel(r"$\varepsilon$", fontsize=24)
    ax.set_ylabel(r"Likelihood Quantile - $\mathcal{Q}_q(\varepsilon)$", fontsize=24)
    ax.tick_params(axis="both", labelsize=20)
    ax.grid(True)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, shrink=0.9)
    cbar.set_label(r"Steps", fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    ax.legend(fontsize=18, loc="best", frameon=False)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def plot_likelihood_over_time(
    likelihood_history,
    filename="likelihood_vs_iters.pdf",
    track_every=10,
    quantiles=None,
    include_colorbar=False,
):
    """Plot how likelihood changes over training steps for tracked samples.

    Args:
        likelihood_history: list of tensors, each of shape (num_tracked_samples,)
        labels: optional list of labels for each tracked sample
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # Use Times New Roman and enable LaTeX rendering
    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"""
    \usepackage{bm}
    """

    likelihood_history = torch.stack(likelihood_history, dim=0).detach().cpu().numpy()  # (num_steps, num_samples)
    num_steps, num_samples = likelihood_history.shape

    # Color by initial likelihood (step 0)
    initial = likelihood_history[0]
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = mpl.cm.plasma

    fig, ax = plt.subplots(figsize=(7.6, 6))
    for i in range(num_samples):
        color = cmap(norm(initial[i]))
        label = (r"$\mathcal{Q}_q$" + f"({quantiles[i]:.2f})") if quantiles is not None else f"Sample {i}"
        ax.plot(
            torch.arange(len(likelihood_history)) * track_every,
            likelihood_history[:, i],
            label=label,
            color=color,
            linewidth=2.5,
        )
    ax.tick_params(axis="both", labelsize=20)

    ax.set_xlabel(r"PG Step", fontsize=24)
    ax.set_ylabel(r"Likelihood - $p_{\bm{w}_t}\big(\bm{y}^*|\bm{x}\big)$", fontsize=24)
    if include_colorbar:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.04, shrink=0.9)
        cbar.set_label(r"Initial Likelihood - $q(\bm{y}^*|\bm{x})$", fontsize=18)
        cbar.ax.tick_params(labelsize=16)
    # ax.legend(title="Initial Likelihood", title_fontsize=14, loc='best', fontsize=18)
    ax.grid(True)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def plot_multi_likelihoods_over_time(likelihood_histories, filename="likelihood_vs_iters.pdf", titles=None):
    """Plot likelihood over time for one or more runs side by side.

    Args:
        likelihood_histories: list of lists (or a single list) of tensors,
                              each tensor shape (num_tracked_samples,)
        labels: optional list of labels for each tracked sample
        titles: optional list of titles for each subplot
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # Use Times New Roman and enable LaTeX rendering
    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["text.usetex"] = True

    # Normalize input to list of histories
    if len(likelihood_histories) > 0 and not isinstance(likelihood_histories[0], (list, tuple)):
        likelihood_histories = [likelihood_histories]

    num_runs = len(likelihood_histories)
    if num_runs == 0:
        return

    # Fixed color scale and colormap
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = mpl.cm.plasma

    fig, axes = plt.subplots(1, num_runs, figsize=(6 * num_runs, 5), sharey=True, gridspec_kw={"wspace": 0.10})
    if num_runs == 1:
        axes = [axes]

    for run_idx, history in enumerate(likelihood_histories):
        arr = torch.stack(history, dim=0).cpu().numpy()  # (num_steps, num_samples)
        _, num_samples = arr.shape
        initial = arr[0]
        ax = axes[run_idx]
        for i in range(num_samples):
            color = cmap(norm(initial[i]))
            ax.plot(arr[:, i], color=color)
        if titles is not None and run_idx < len(titles):
            ax.set_title(titles[run_idx], fontsize=24)
        else:
            ax.set_title(r"Likelihood over training", fontsize=24)
        ax.set_xlabel(r"PG Step", fontsize=24)
        ax.tick_params(axis="both", labelsize=20)
        ax.grid(True)
        if run_idx == 0:
            ax.set_ylabel(r"Likelihood - $p_{\bm{w}_t}\big(\bm{y}^*|\bm{x}\big)$", fontsize=24)

    # Single colorbar next to the rightmost subplot
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="right", fraction=0.046, pad=0.04, shrink=0.8)
    cbar.set_label(r"Initial Likelihood - $q(\bm{y}^*|\bm{x})$", fontsize=22)
    cbar.ax.tick_params(labelsize=20)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def plot_expected_error_over_time(pg_errors, filename="expected_error_vs_iters.pdf", test_every=50):
    """Plot expected error over time from pg_errors (test set averages).

    Args:
        pg_errors: list of test-set sequence errors (0-1) from policy_gradient_train
        test_every: evaluation interval used to generate pg_errors
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"""
    \usepackage{bm}
    \usepackage{amsmath}
    \usepackage{amssymb}
    """

    errors = torch.as_tensor(pg_errors, dtype=torch.float32)
    steps = torch.arange(len(errors)) * test_every

    fig, ax = plt.subplots(figsize=(7.6, 6))
    ax.plot(steps.numpy(), errors.numpy(), linewidth=2.5)
    ax.set_xlabel(r"PG Step", fontsize=24)
    ax.set_ylabel(r"Expected Error - $\mathbb{P}[\bm{y} \neq \bm{y}^*]$", fontsize=24)
    ax.tick_params(axis="both", labelsize=20)
    ax.grid(True)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
