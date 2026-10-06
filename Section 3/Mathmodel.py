import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import stats


#Importing data and whatnot

path = os.path.join(os.path.dirname(__file__), '..' , 'rotterdam_dataset_06.xlsx')
rd = pd.read_excel(path)

types = ['Container', 'Bulk', 'Tanker']

def estimate_lambda(vessel_type, rd = rd):
    arrival = rd[rd['Vessel_Type'] == vessel_type]['Arrival_Date'].sort_values()
    arrival_time = ( arrival - arrival.iloc[0]).dt.total_seconds() / (3600 * 24) 
    inter_arrival_times = np.diff(arrival_time)
    return 1 / np.mean(inter_arrival_times)

#print the estimate lambda for vessels cuz i need to check them
#print(f'container lambda: {estimate_lambda('Container')}')
#print(f'bulk lambda: {estimate_lambda('Bulk')}')
#print(f'tanker lambda: {estimate_lambda('Tanker')}')

def estimate_lognormal(vessel_type, rd = rd):
    tons = rd[rd['Vessel_Type'] == vessel_type]['Cargo_tons'].to_numpy()
    m1 = np.mean(tons)
    m2 = np.mean(tons**2)

    sigma2 = np.log(m2 / m1**2)
    mu = np.log(m1) - sigma2 / 2
    return mu, np.sqrt(sigma2)

lam_C = estimate_lambda('Container')
lam_B = estimate_lambda('Bulk')
lam_T = estimate_lambda('Tanker')

mu_C, sigma_C = estimate_lognormal('Container')
mu_B, sigma_B = estimate_lognormal('Bulk')
mu_T, sigma_T = estimate_lognormal('Tanker')

def simPP(lam, t): 
    expDist = stats.expon(scale = 1/lam)
    time = expDist.rvs()
    nT = 0
    while time < t: 
        nT += 1 
        time += expDist.rvs()
    return nT

def simCargo(nT, mu, sigma):
    cargoDist = stats.lognorm(s = sigma, scale = np.exp(mu))
    cargo = cargoDist.rvs(nT)
    return np.sum(cargo)

def simPort(T):
    N_C = simPP(lam_C, T)
    N_B = simPP(lam_B, T)
    N_T = simPP(lam_T, T)
    N = N_C + N_B + N_T

    S_C = simCargo(N_C, mu_C, sigma_C)
    S_B = simCargo(N_B, mu_B, sigma_B)
    S_T = simCargo(N_T, mu_T, sigma_T)
    S = S_C + S_B + S_T

    return [N_C, N_B, N_T, N, S_C, S_B, S_T, S]

np.random.seed(7567)
T = 365
n = 1000

results = np.array([simPort(T) for _ in range(n)])

labels = ['N_C', 'N_B', 'N_T', 'N', 'S_C', 'S_B', 'S_T', 'S']
means = np.mean(results, axis=0)
stds = np.std(results, axis=0, ddof=1)
q95 = np.quantile(results, 0.95, axis=0)

table = pd.DataFrame({'Mean': means, 'Std': stds, '95% Quantile': q95}, index=labels)
print(table)

capacity = 650000
nDays = 20000
daily = np.array([simPort(1)[7] for _ in range(nDays)])

exceed = daily > capacity
p = np.mean(exceed)

s2 = np.var(exceed)
z = 1.96
halfWidth = z * np.sqrt(s2 / nDays)
interval  = (p - halfWidth, p + halfWidth)
print(p, interval)
print('excpected days over capacity / year: ', p * 365)

# Compare simulation with observed data
observed = rd.groupby(rd['Arrival_Date'].dt.date)['Cargo_tons'].sum().to_numpy()
p_obs = np.mean(observed > capacity)

plt.figure()
bins = np.arange(0, 1300, 25)   #in thousand tons
plt.hist(daily / 1000, bins = bins, density = True, alpha = 0.6, label = 'Simulated')
plt.hist(observed / 1000, bins = bins, density = True, histtype = 'step', color = 'black', linewidth = 1.5, label = 'Observed (2022-2024)')
plt.axvspan(capacity / 1000, bins[-1], color = 'red', alpha = 0.1, label = f'Over capacity: {p:.1%} sim, {p_obs:.1%} obs')
plt.axvline(capacity / 1000, color = 'red', linestyle = '--', label = 'Capacity (650k tons)')
plt.xlabel('Total cargo per day (thousand tons)')
plt.ylabel('Density')
plt.legend(loc = 'upper left')
plt.savefig(os.path.join(os.path.dirname(__file__), 'daily_throughput_hist.png'), bbox_inches = 'tight')
plt.show()

#exceedance curve: for every x, fraction of days above x
x = np.linspace(0, np.quantile(daily, 0.999), 500)
exceed_sim = np.array([np.mean(daily > xi) for xi in x])
exceed_obs = np.array([np.mean(observed > xi) for xi in x])
q95 = np.quantile(daily, 0.95)
q99 = np.quantile(daily, 0.99)

plt.figure()
plt.plot(x / 1000, exceed_sim, label = 'Simulated')
plt.plot(x / 1000, exceed_obs, color = 'black', linestyle = ':', label = 'Observed (2022-2024)')
plt.axvline(capacity / 1000, color = 'red', linestyle = '--', label = 'Capacity (650k tons)')
plt.axvline(q95 / 1000, color = 'grey', linestyle = '--', label = f'95% quantile ({q95/1000:.0f}k)')
plt.axvline(q99 / 1000, color = 'black', linestyle = '--', label = f'99% quantile ({q99/1000:.0f}k)')
plt.yscale('log')   #log scale so the tail is visible
plt.xlabel('Total cargo per day (thousand tons)')
plt.ylabel('P(daily throughput > x)')
plt.legend()
plt.savefig(os.path.join(os.path.dirname(__file__), 'daily_exceedance.png'), bbox_inches = 'tight')
plt.show()
