import torch


def _nice_tick_step(span: float, target_ticks: int = 6) -> int:
    """Return a human-friendly integer tick step for a given span."""
    import math

    if span <= 0:
        return 1
    raw = span / max(1, target_ticks - 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = m * magnitude
        if step >= raw:
            return max(1, int(round(step)))
    return max(1, int(round(10 * magnitude)))


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
    ax.set_xlabel(r"$\alpha$", fontsize=30)
    ax.set_ylabel(r"Likelihood CDF - $\mathbb{P}[q(\bm{y}^*|\bm{x}) \leq \alpha]$", fontsize=24)
    ax.tick_params(axis="both", labelsize=22)
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
def plot_quantile(p_sorteds, all_steps, filename="quantile.pdf"):
    """Plot inverse CDF (quantile function) for one or more models.

    Args:
        p_sorteds: list of sorted p(y|x) tensors
        all_steps: tensor/list of step labels for coloring (same length as p_sorteds)
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

    ax.set_xlabel(r"$\varepsilon$", fontsize=30)
    ax.set_ylabel(r"Likelihood Quantile - $\mathcal{Q}_q(\varepsilon)$", fontsize=30)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(torch.linspace(0, 1, 5).tolist())
    ax.tick_params(axis="both", labelsize=22)
    ax.grid(True)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, shrink=0.9)
    cbar.set_label(r"Steps", fontsize=28)
    cbar.ax.tick_params(labelsize=22)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def plot_likelihood_over_time(
    likelihood_history,
    filename="likelihood_vs_iters.pdf",
    track_every=10,
    quantiles=None,
    include_colorbar=False,
    ema_beta=0.0,
    title: str | None = None,
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

    likelihood_history = torch.stack(likelihood_history, dim=0).detach().cpu().numpy()
    beta = float(ema_beta)
    if not (0.0 <= beta < 1.0):
        raise ValueError(f"ema_beta must be in [0, 1), got {ema_beta}")
    if beta > 0.0 and likelihood_history.shape[0] > 1:
        for t in range(1, likelihood_history.shape[0]):
            likelihood_history[t] = beta * likelihood_history[t - 1] + (1.0 - beta) * likelihood_history[t]
    num_steps, num_samples = likelihood_history.shape

    # Color by initial likelihood (step 0)
    initial = likelihood_history[0]
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = mpl.cm.plasma

    fig, ax = plt.subplots(figsize=(7.6, 6))
    steps = torch.arange(len(likelihood_history)) * track_every + 1
    for i in range(num_samples):
        color = cmap(norm(initial[i]))
        label = (r"$\mathcal{Q}_q$" + f"({quantiles[i]:.2f})") if quantiles is not None else f"Sample {i}"
        ax.plot(
            steps,
            likelihood_history[:, i],
            label=label,
            color=color,
            linewidth=1.5,
        )
    ax.tick_params(axis="both", labelsize=24)
    ax.set_xscale("log")

    ax.set_xlabel(r"PG Step", fontsize=30)
    ax.set_ylabel(r"Likelihood - $p_{\bm{w}_t}\big(\bm{y}^*|\bm{x}\big)$", fontsize=30)
    if include_colorbar:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.04, shrink=0.9)
        cbar.set_label("Initial Likelihood", fontsize=28)
        cbar.ax.tick_params(labelsize=20)
    # ax.legend(title="Initial Likelihood", title_fontsize=14, loc='best', fontsize=18)
    ax.grid(True)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def plot_average_likelihood_over_time(
    likelihood_history,
    filename="average_likelihood_vs_iters.pdf",
    track_every=10,
):
    """Plot the mean tracked likelihood across samples over PG steps."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"""
    \usepackage{bm}
    \usepackage{amsmath}
    \usepackage{amssymb}
    """

    history = torch.stack(likelihood_history, dim=0).detach().cpu()
    mean_likelihood = history.mean(dim=1)
    steps = torch.arange(history.shape[0]) * track_every + 1

    fig, ax = plt.subplots(figsize=(7.6, 6))
    ax.plot(steps.numpy(), mean_likelihood.numpy(), linewidth=2.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"PG Step", fontsize=24)
    ax.set_ylabel(r"Average Likelihood", fontsize=22)
    ax.tick_params(axis="both", labelsize=20)
    ax.grid(True)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def plot_compare_average_likelihood_over_time(
    outcome_likelihood_history,
    process_likelihood_history,
    filename="compare_outcome_process_off_support.pdf",
    outcome_track_every=10,
    process_track_every=10,
    show_legend: bool = True,
):
    """Plot outcome vs process average tracked likelihood on one axis."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"""
    \usepackage{bm}
    \usepackage{amsmath}
    \usepackage{amssymb}
    """

    outcome_hist = torch.stack(outcome_likelihood_history, dim=0).detach().cpu()
    process_hist = torch.stack(process_likelihood_history, dim=0).detach().cpu()
    outcome_mean = outcome_hist.mean(dim=1)
    process_mean = process_hist.mean(dim=1)
    outcome_steps = torch.arange(outcome_hist.shape[0]) * outcome_track_every + 1
    process_steps = torch.arange(process_hist.shape[0]) * process_track_every + 1

    fig, ax = plt.subplots(figsize=(7.6, 6))
    ax.plot(outcome_steps.numpy(), outcome_mean.numpy(), linewidth=2.5, label="Outcome Reward")
    ax.plot(process_steps.numpy(), process_mean.numpy(), linewidth=2.5, label="Process Reward")
    ax.set_xscale("log")
    ax.set_xlabel(r"PG Step", fontsize=30)
    ax.set_ylabel(r"Avg. Likelihood of Off-Support Samples", fontsize=24, y=0.46)
    ax.tick_params(axis="both", labelsize=22)
    ax.grid(True)
    if show_legend:
        ax.legend(fontsize=24, frameon=False)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def plot_compare_expected_error_over_time(
    outcome_pg_errors,
    process_pg_errors,
    filename="compare_outcome_process_err.pdf",
    outcome_test_every=50,
    process_test_every=50,
    offset=0,
    show_legend: bool = True,
):
    """Plot outcome vs process expected error on one axis."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.ticker import MaxNLocator

    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"""
    \usepackage{bm}
    \usepackage{amsmath}
    \usepackage{amssymb}
    """

    outcome_errors = torch.as_tensor(outcome_pg_errors, dtype=torch.float32).clone()
    process_errors = torch.as_tensor(process_pg_errors, dtype=torch.float32).clone()
    outcome_start = max(0, (int(offset) + outcome_test_every - 1) // outcome_test_every)
    process_start = max(0, (int(offset) + process_test_every - 1) // process_test_every)
    outcome_steps = torch.arange(outcome_start, outcome_errors.shape[0]) * outcome_test_every
    process_steps = torch.arange(process_start, process_errors.shape[0]) * process_test_every
    outcome_errors = outcome_errors[outcome_start:]
    process_errors = process_errors[process_start:]

    fig, ax = plt.subplots(figsize=(7.6, 6))
    ax.plot(outcome_steps.numpy(), outcome_errors.numpy(), linewidth=2.5, label="Outcome Reward")
    ax.plot(process_steps.numpy(), process_errors.numpy(), linewidth=2.5, label="Process Reward")
    offset = int(offset)
    ax.set_xlim(left=offset)
    x_max = max(
        float(outcome_steps.max().item()) if outcome_steps.numel() > 0 else float(offset),
        float(process_steps.max().item()) if process_steps.numel() > 0 else float(offset),
    )
    tick_step = _nice_tick_step(max(0.0, x_max - float(offset)), target_ticks=6)
    major_ticks = [offset]
    first_aligned = ((offset + tick_step - 1) // tick_step) * tick_step
    major_ticks.extend(range(first_aligned, int(x_max) + 1, tick_step))
    ax.set_xticks(sorted(set(major_ticks)))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_xlabel(r"PG Step", fontsize=30)
    ax.set_ylabel(r"Test Error", fontsize=30)
    ax.tick_params(axis="both", labelsize=24)
    ax.minorticks_off()
    ax.grid(True)
    if show_legend:
        ax.legend(fontsize=24, frameon=False)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def plot_expected_error_over_time(
    pg_errors,
    filename="expected_error_vs_iters.pdf",
    test_every=50,
    offset=0,
    title: str | None = None,
):
    """Plot expected error over time from pg_errors (test set averages).

    Args:
        pg_errors: list of test-set sequence errors (0-1) from policy_gradient_train
        test_every: evaluation interval used to generate pg_errors
        offset: minimum PG step to include (default: 0)
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.ticker import MaxNLocator

    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"""
    \usepackage{bm}
    \usepackage{amsmath}
    \usepackage{amssymb}
    """

    errors = torch.as_tensor(pg_errors, dtype=torch.float32).clone()
    start = max(0, (int(offset) + test_every - 1) // test_every)
    steps = torch.arange(start, len(errors)) * test_every
    errors = errors[start:]

    fig, ax = plt.subplots(figsize=(7.6, 6))
    ax.plot(steps.numpy(), errors.numpy(), linewidth=2.5)
    offset = int(offset)
    ax.set_xlim(left=offset)
    x_max = float(steps.max().item()) if steps.numel() > 0 else float(offset)
    tick_step = _nice_tick_step(max(0.0, x_max - float(offset)), target_ticks=6)
    major_ticks = [offset]
    first_aligned = ((offset + tick_step - 1) // tick_step) * tick_step
    major_ticks.extend(range(first_aligned, int(x_max) + 1, tick_step))
    ax.set_xticks(sorted(set(major_ticks)))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_xlabel(r"PG Step", fontsize=30)
    ax.set_ylabel(r"Test Error", fontsize=30)
    if title:
        ax.set_title(title, fontsize=30)
    ax.tick_params(axis="both", labelsize=24)
    ax.minorticks_off()
    ax.grid(True)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def plot_likelihood_histogram(
    likelihoods,
    filename="track_set_likelihood_histogram.pdf",
    bins=80,
    title: str | None = None,
):
    """Plot a histogram of sequence likelihoods."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"""
    \usepackage{bm}
    \usepackage{amsmath}
    \usepackage{amssymb}
    """

    likelihoods = torch.as_tensor(likelihoods, dtype=torch.float32).detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(7.6, 6))
    ax.hist(
        likelihoods,
        bins=max(10, int(bins)),
        color="#1f77b4",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xlabel(r"Likelihood - $q(\bm{y}^*|\bm{x})$", fontsize=24)
    ax.set_ylabel(r"Count", fontsize=24)
    if title:
        ax.set_title(title, fontsize=20)
    ax.tick_params(axis="both", labelsize=20)
    ax.grid(True, alpha=0.3)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
