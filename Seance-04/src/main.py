#coding:utf8

import numpy as np
import pandas as pd
import scipy
import scipy.stats

#https://docs.scipy.org/doc/scipy/reference/stats.html


dist_names = ['norm', 'beta', 'gamma', 'pareto', 't', 'lognorm', 'invgamma', 'invgauss',  'loggamma', 'alpha', 'chi', 'chi2', 'bradford', 'burr', 'burr12', 'cauchy', 'dweibull', 'erlang', 'expon', 'exponnorm', 'exponweib', 'exponpow', 'f', 'genpareto', 'gausshyper', 'gibrat', 'gompertz', 'gumbel_r', 'pareto', 'pearson3', 'powerlaw', 'triang', 'weibull_min', 'weibull_max', 'bernoulli', 'betabinom', 'betanbinom', 'binom', 'geom', 'hypergeom', 'logser', 'nbinom', 'poisson', 'poisson_binom', 'randint', 'zipf', 'zipfian']

print(dist_names)
#Première étape
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
# Dossier img
IMG_DIR = "img"
os.makedirs(IMG_DIR, exist_ok=True)
#Code
def plot_dirac(k0=0, kmin=None, kmax=None, save="img/dirac.png"):
    if kmin is None: kmin = k0 - 5
    if kmax is None: kmax = k0 + 5
    k = np.arange(kmin, kmax + 1)
    pmf = (k == k0).astype(int)

    plt.figure(figsize=(8,5))
    plt.stem(k, pmf, use_line_collection=True, basefmt=" ")
    plt.title(f"Loi de Dirac (k₀={k0})")
    plt.xlabel("k"); plt.ylabel("p(k)")
    plt.grid(True, alpha=0.3)
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

# Utilisation
plot_dirac(k0=0, save="img/dirac.png")

#LOI UNIFORME DISCRÈTE (Diagramme en bâtons (bar))
def plot_uniforme_discrete(n=10, save="uniforme_discrete.png"):
    k = np.arange(n)
    pmf = np.ones(n) / n  # Probabilité égale 1/n

    plt.figure(figsize=(8,5))
    plt.bar(k, pmf, alpha=0.7, color='skyblue', edgecolor='navy', width=0.8)
    plt.title(f'Loi uniforme discrète (0 à {n-1})')
    plt.xlabel('k'); plt.ylabel('P(X=k)')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{IMG_DIR}/{save}', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

#LOI BINOMIALE (stem plot (tiges) (histogramme échantillons))
def plot_binomiale(n=20, p=0.5, size=1000, save="binomiale.png"):
    k = np.arange(n+1)
    pmf = stats.binom.pmf(k, n, p)
    samples = stats.binom.rvs(n, p, size=size)

    plt.figure(figsize=(10,6))
    plt.stem(k, pmf, basefmt=" ", linefmt='r-', markerfmt='ro', label='PMF théorique')
    plt.hist(samples, bins=range(n+2), density=True, alpha=0.6, color='orange', label='Échantillons')
    plt.title(f'Loi binomiale B({n},{p})')
    plt.xlabel('k'); plt.ylabel('P(X=k)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f'{IMG_DIR}/{save}', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

#LOI DE POISSON (Stem plot (tiges)  (histogramme))
def plot_poisson(mu=5, size=1000, save="poisson.png"):
    k_max = int(mu + 4*np.sqrt(mu))
    k = np.arange(k_max+1)
    pmf = stats.poisson.pmf(k, mu)
    samples = stats.poisson.rvs(mu, size=size)

    plt.figure(figsize=(10,6))
    plt.stem(k, pmf, basefmt=" ", linefmt='g-', markerfmt='go', label='PMF théorique')
    plt.hist(samples, bins=range(k_max+2), density=True, alpha=0.6, color='lightgreen', label='Échantillons')
    plt.title(f'Loi de Poisson λ={mu}')
    plt.xlabel('k'); plt.ylabel('P(X=k)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f'{IMG_DIR}/{save}', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

#LOI ZIPF-MANDELBROT (Diagramme en bâtons décroissant (queue lourde))
def plot_zipf_mandelbrot(n_max=50, alpha=1.5, s=0.5, save="zipf_mandelbrot.png"):
    k = np.arange(1, n_max+1)
    pmf = k ** (-alpha + s)
    pmf = pmf / pmf.sum()  # Normalisation

    plt.figure(figsize=(10,6))
    plt.bar(k, pmf, alpha=0.7, color='purple', edgecolor='darkviolet', width=0.8)
    plt.title(f'Loi Zipf-Mandelbrot (α={alpha}, s={s})')
    plt.xlabel('k (rang)'); plt.ylabel('P(X=k)')
    plt.yscale('log')  # Échelle log pour voir la queue lourde
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{IMG_DIR}/{save}', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
#Affichage
plot_uniforme_discrete(n=10, save="uniforme_discrete.png")
plot_binomiale(n=20, p=0.5, save="binomiale.png")
plot_poisson(mu=5, save="poisson.png")
plot_zipf_mandelbrot(n_max=50, save="zipf_mandelbrot.png")
#Lois continues

# 1. LOI DE POISSON CONTINUE ((approximation Gamma) (Histogramme + PDF))
def plot_poisson_continue(mu=5, size=1000, save="poisson_continue.png"):
    samples = stats.gamma.rvs(a=mu+1/3, scale=1, size=size)
    x = np.linspace(0, 20, 1000)
    pdf = stats.gamma.pdf(x, a=mu+1/3, scale=1)

    plt.figure(figsize=(10,6))
    plt.hist(samples, bins=50, density=True, alpha=0.6, color='lightcoral', label='Échantillons')
    plt.plot(x, pdf, 'r-', lw=3, label=f'Gamma(μ+1/3={mu+1/3})')
    plt.title(f'Loi de Poisson continue (approximation Γ)')
    plt.xlabel('x'); plt.ylabel('Densité f(x)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f'{IMG_DIR}/{save}', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

# 2. LOI NORMALE (Histogramme + PDF + intervalles confiance)
def plot_normale(mu=0, sigma=1, size=1000, save="normale.png"):
    samples = np.random.normal(mu, sigma, size)
    x = np.linspace(mu-4*sigma, mu+4*sigma, 1000)
    pdf = stats.norm.pdf(x, mu, sigma)

    plt.figure(figsize=(10,6))
    plt.hist(samples, bins=50, density=True, alpha=0.6, color='skyblue', label='Échantillons')
    plt.plot(x, pdf, 'b-', lw=3, label=f'N({mu},{sigma})')
    plt.axvline(mu, color='red', linestyle='--', label='μ')
    plt.axvline(mu+sigma, color='orange', linestyle='--', alpha=0.7, label='μ+σ')
    plt.title('Loi normale (Gaussienne)')
    plt.xlabel('x'); plt.ylabel('Densité f(x)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f'{IMG_DIR}/{save}', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

# 3. LOI LOG-NORMALE (Histogramme + PDF (échelle log optionnelle))
def plot_lognormale(mu=0, sigma=0.5, size=1000, save="lognormale.png"):
    samples = stats.lognorm.rvs(s=sigma, scale=np.exp(mu), size=size)
    x = np.linspace(0.1, 5, 1000)
    pdf = stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu))

    plt.figure(figsize=(10,6))
    plt.hist(samples, bins=50, density=True, alpha=0.6, color='gold', label='Échantillons')
    plt.plot(x, pdf, 'orange', lw=3, label=f'LogN(μ={mu},σ={sigma})')
    plt.title('Loi log-normale')
    plt.xlabel('x'); plt.ylabel('Densité f(x)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f'{IMG_DIR}/{save}', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

# 4. LOI UNIFORME (Histogramme plat + PDF rectangulaire)
def plot_uniforme(a=0, b=10, size=1000, save="uniforme_continue.png"):
    samples = np.random.uniform(a, b, size)
    x = np.linspace(a-1, b+1, 1000)
    pdf = stats.uniform.pdf(x, a, b-a)

    plt.figure(figsize=(10,6))
    plt.hist(samples, bins=50, density=True, alpha=0.6, color='lightgreen', label='Échantillons')
    plt.plot(x, pdf, 'g-', lw=3, label=f'U[{a},{b}]')
    plt.fill_between(x, pdf, alpha=0.2, color='green')
    plt.title('Loi uniforme continue')
    plt.xlabel('x'); plt.ylabel('Densité f(x)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f'{IMG_DIR}/{save}', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

# 5. LOI DU χ² (Histogramme asymétrique + PDF)
def plot_chi2(df=5, size=1000, save="chi2.png"):
    samples = stats.chi2.rvs(df, size=size)
    x = np.linspace(0, 20, 1000)
    pdf = stats.chi2.pdf(x, df)

    plt.figure(figsize=(10,6))
    plt.hist(samples, bins=50, density=True, alpha=0.6, color='violet', label='Échantillons')
    plt.plot(x, pdf, 'purple', lw=3, label=f'χ²(df={df})')
    plt.title('Loi du χ²')
    plt.xlabel('x'); plt.ylabel('Densité f(x)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f'{IMG_DIR}/{save}', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

# 6. LOI DE PARETO (Histogramme queue lourde + PDF (échelle log-log))
def plot_pareto(b=1.5, alpha=2.5, size=1000, save="pareto.png"):
    samples = stats.pareto.rvs(b=b, scale=1, size=size)
    x = np.linspace(1, 10, 1000)
    pdf = stats.pareto.pdf(x, b=b, scale=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))


    # Graphique log-log (caractéristique queue lourde)
    ax2.loglog(x, pdf, 'darkred', lw=3, label=f'Pareto(b={b},α={alpha})')
    ax2.scatter(samples[:100], np.ones(100)*0.01, alpha=0.5, s=10, color='brown')
    ax2.set_title('Loi de Pareto (échelle log-log)')
    ax2.set_xlabel('x (log)'); ax2.set_ylabel('f(x) (log)')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/{save}', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

#Affichage
if __name__ == "__main__":
    plot_poisson_continue()
    plot_normale()
    plot_lognormale()
    plot_uniforme()
    plot_chi2()
    plot_pareto()

#Etape 2
#LOI DE DIRAC
def stats_dirac(k0=0):
    print(f" LOI DE DIRAC (k₀={k0})")
    print(f"   Moyenne théorique : {float(k0):.4f}")
    print(f"   Écart-type théorique : {0.0000:.4f}\n")

#LOI UNIFORME DISCRÈTE
def stats_uniforme_discrete(n=10):
    distrib = stats.randint(0, n)
    print(f" LOI UNIFORME DISCRÈTE [0,{n-1}]")
    print(f"   Moyenne théorique : {distrib.mean():.4f}")
    print(f"   Écart-type théorique : {distrib.std():.4f}\n")

#LOI BINOMIALE
def stats_binomiale(n=20, p=0.5):
    distrib = stats.binom(n, p)
    print(f" LOI BINOMIALE B({n},{p})")
    print(f"   Moyenne théorique : {distrib.mean():.4f}")
    print(f"   Écart-type théorique : {distrib.std():.4f}\n")

#LOI DE POISSON (DISCRÈTE)
def stats_poisson(mu=5):
    distrib = stats.poisson(mu)
    print(f" LOI DE POISSON (λ={mu})")
    print(f"   Moyenne théorique : {distrib.mean():.4f}")
    print(f"   Écart-type théorique : {distrib.std():.4f}\n")

#LOI ZIPF-MANDELBROT
def stats_zipf_mandelbrot(n_max=50, alpha=1.5):
    k = np.arange(1, n_max+1)
    pmf = k ** (-alpha)
    pmf = pmf / pmf.sum()
    mean_zipf = np.sum(k * pmf)
    var_zipf = np.sum((k**2) * pmf) - mean_zipf**2
    std_zipf = np.sqrt(var_zipf)
    print(f" LOI ZIPF-MANDELBROT (α={alpha}, n_max={n_max})")
    print(f"   Moyenne théorique : {mean_zipf:.4f}")
    print(f"   Écart-type théorique : {std_zipf:.4f}\n")


#LOI DE POISSON CONTINUE
def stats_poisson_continue(mu=5):
    distrib = stats.gamma(a=mu+1/3, scale=1)
    print(f" LOI DE POISSON CONTINUE Γ(μ+1/3={mu+1/3:.2f})")
    print(f"   Moyenne théorique : {distrib.mean():.4f}")
    print(f"   Écart-type théorique : {distrib.std():.4f}\n")

#LOI NORMALE
def stats_normale(mu=0, sigma=1):
    distrib = stats.norm(mu, sigma)
    print(f" LOI NORMALE N({mu},{sigma})")
    print(f"   Moyenne théorique : {distrib.mean():.4f}")
    print(f"   Écart-type théorique : {distrib.std():.4f}\n")

#LOI LOG-NORMALE
def stats_lognormale(mu=0, sigma=0.5):
    distrib = stats.lognorm(s=sigma, scale=np.exp(mu))
    print(f" LOI LOG-NORMALE (μ={mu},σ={sigma})")
    print(f"   Moyenne théorique : {distrib.mean():.4f}")
    print(f"   Écart-type théorique : {distrib.std():.4f}\n")

#LOI UNIFORME CONTINUE
def stats_uniforme_continue(a=0, b=1):
    distrib = stats.uniform(a, b-a)
    print(f" LOI UNIFORME CONTINUE [{a},{b}]")
    print(f"   Moyenne théorique : {distrib.mean():.4f}")
    print(f"   Écart-type théorique : {distrib.std():.4f}\n")

#LOI DU χ²
def stats_chi2(df=5):
    distrib = stats.chi2(df)
    print(f" LOI DU χ² (df={df})")
    print(f"   Moyenne théorique : {distrib.mean():.4f}")
    print(f"   Écart-type théorique : {distrib.std():.4f}\n")

#LOI DE PARETO
def stats_pareto(b=1.5, alpha=2.5):
    distrib = stats.pareto(b, scale=1)
    print(f"  LOI DE PARETO (b={b},α={alpha})")
    print(f"   Moyenne théorique : {distrib.mean():.4f}")
    print(f"   Écart-type théorique : {distrib.std():.4f}\n")

if __name__ == "__main__":
    print("=== MOYENNE ET ÉCART-TYPE THÉORIQUES ===\n")

    # Discrètes (5)
    stats_dirac()
    stats_uniforme_discrete()
    stats_binomiale()
    stats_poisson()
    stats_zipf_mandelbrot()

    # Continues (6)
    stats_poisson_continue()
    stats_normale()
    stats_lognormale()
    stats_uniforme_continue()
    stats_chi2()
    stats_pareto()

    print()





