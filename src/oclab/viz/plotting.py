import matplotlib.pyplot as plt

def plot_alpha_running(mu, a1, a2, a3):
    fig = plt.figure()
    plt.loglog(mu, a1, label='alpha1')
    plt.loglog(mu, a2, label='alpha2')
    plt.loglog(mu, a3, label='alpha3')
    plt.xlabel('mu [GeV]'); plt.ylabel('alpha(mu)')
    plt.legend(); plt.title('Gauge couplings running')
    return fig

def plot_eta(k, eta):
    fig = plt.figure()
    plt.loglog(k, abs(eta))
    plt.xlabel('k [GeV]'); plt.ylabel('|eta_A(k)|')
    plt.title('FRG anomalous dimension')
    return fig

def plot_eta_with_freeze(k, eta, k_star=None):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.loglog(k, [abs(e) for e in eta], label="|eta_A(k)|")
    if k_star:
        plt.axvline(k_star, linestyle="--", label=f"k_*={k_star:.3g}")
    plt.xlabel("k [GeV]"); plt.ylabel("|eta_A(k)|")
    plt.title("FRG anomalous dimension (freeze annotated)")
    plt.legend()
    return fig
