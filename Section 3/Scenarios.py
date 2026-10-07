import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Mathmodel import rd, lam_C, lam_B, lam_T, mu_C, mu_B, mu_T, sigma_C, sigma_B, sigma_T, simPP, simCargo

capacity = 650000
newCapacity = 800000

def simDay(lamC, lamB, lamT, muC, muB, muT):
    S_C = simCargo(simPP(lamC, 1), muC, sigma_C)
    S_B = simCargo(simPP(lamB, 1), muB, sigma_B)
    S_T = simCargo(simPP(lamT, 1), muT, sigma_T)
    return S_C + S_B + S_T

def simYear(lamC_day, lamB_day, lamT_day, muC, muB, muT):
    daily = np.zeros(365)
    for d in range(365):
        daily[d] = simDay(lamC_day[d], lamB_day[d], lamT_day[d], muC, muB, muT)
    return daily

def runScenario(lamC_day, lamB_day, lamT_day, muC, muB, muT, nYears):
    return np.array([simYear(lamC_day, lamB_day, lamT_day, muC, muB, muT) for _ in range(nYears)])

def summary(name, years, capacity = capacity):
    daysOver = np.sum(years > capacity, axis=1)
    m = np.mean(daysOver)
    s2 = np.var(daysOver, ddof=1)
    halfWidth = 1.96 * np.sqrt(s2 / len(daysOver))
    return [name, np.mean(years), np.std(years,ddof=1), np.quantile(years,0.95), np.quantile(years, 0.99), np.mean(years > capacity), m, m - halfWidth, m + halfWidth]

np.random.seed(7567)
nYears = 30

constant = np.ones(365)
basic = runScenario(constant * lam_C, constant * lam_B, constant * lam_T, mu_C, mu_B, mu_T, nYears)

#scenario 1: seasonal arrivals, lambda / month
def estimate_lambda_month(vessel_type, month, rd = rd): 
    inMonth = rd[rd['Arrival_Date'].dt.month == month]
    nDays = inMonth['Arrival_Date'].dt.date.nunique()
    nVessels = np.sum(inMonth['Vessel_Type'] == vessel_type)
    return nVessels / nDays

lamC_month = np.array([estimate_lambda_month('Container', m) for m in range(1, 13)])
lamB_month = np.array([estimate_lambda_month('Bulk', m) for m in range(1, 13)])
lamT_month = np.array([estimate_lambda_month('Tanker', m) for m in range(1, 13)])

months = pd.date_range('2025-01-01', periods = 365).month.to_numpy()
seasonal = runScenario(lamC_month[months - 1], lamB_month[months - 1], lamT_month[months - 1], mu_C, mu_B, mu_T, nYears)

#scenario 2: trade growth over 10 years, current vs new terminal
g = 0.02 # arbitrary growth rate
futureYears = np.arange(1, 11)
nYearsGrowth = 10
over650 = []
over800 = []
for y in futureYears:
    f = (1 + g) ** y
    grown = runScenario(constant * f * lam_C, constant * f * lam_B, constant * f * lam_T, mu_C, mu_B, mu_T, nYearsGrowth)
    over650.append(np.mean(np.sum(grown > capacity, axis=1)))
    over800.append(np.mean(np.sum(grown > newCapacity, axis=1)))
    print(f'Year {y}: {np.mean(np.sum(grown > capacity, axis=1))} days over 650k, {np.mean(np.sum(grown > newCapacity, axis=1))} days over 800k')

#scenario 3, larger vessles so more cargo per vessel, but less vessels (same ton arrival)

c = 1.15
larger = runScenario(constant * lam_C / c, constant * lam_B / c, constant * lam_T / c, mu_C + np.log(c), mu_B + np.log(c), mu_T + np.log(c), nYears)

#table with scenario summary
rows = [summary('Basic', basic), summary('Seasonal', seasonal), summary('Larger vessels', larger), summary('Basic, 800k capacity', basic, newCapacity)]
table = pd.DataFrame(rows, columns = ['Scenario', 'Mean', 'Std', '95% q', '99% q', 'P(day > cap)', 'Days over / year', 'CI low', 'CI high'])
print(table.to_string(index=False))

# plot 1 basic v seasonal
observed = rd.groupby(rd['Arrival_Date'].dt.date)['Cargo_tons'].sum().to_numpy()
x = np.linspace(0, 1100000, 500)
exceed_basic = np.array([np.mean(basic > x_i) for x_i in x])
exceed_seasonal = np.array([np.mean(seasonal > x_i) for x_i in x])
exceed_observed = np.array([np.mean(observed > x_i) for x_i in x])

plt.figure()
plt.plot(x / 1000, exceed_basic, label = 'Basic')
plt.plot(x / 1000, exceed_seasonal, label = 'Seasonal')
plt.plot(x / 1000, exceed_observed, label = 'Observed', color = 'black', linestyle = 'dashed')
plt.axvline(capacity / 1000, color = 'red', linestyle = '--', label = 'Capacity (650k tons)')
plt.yscale('log')   #log scale so the tail is visible
plt.xlabel('Total cargo per day (thousand tons)')
plt.ylabel('P(daily throughput > x)')
plt.legend()
plt.savefig(os.path.join(os.path.dirname(__file__), 'scenario_exceedance.png'), bbox_inches = 'tight')
plt.show()

#plot 2 going over cap per month
pMonth = [np.mean(seasonal[:, months == m] > capacity) for m in range(1, 13)]

plt.figure()
plt.bar(range(1, 13), pMonth, label = 'Seasonal')
plt.axhline(np.mean(basic > capacity), color = 'red', linestyle = '--', label = 'Basic')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.xlabel('Month')
plt.ylabel('P(daily throughput > 650k tons)')
plt.legend()
plt.savefig(os.path.join(os.path.dirname(__file__), 'scenario_monthly.png'), bbox_inches = 'tight')
plt.show()

#plot 3 growth over decade, 650k vs 800k capacity
plt.figure()
plt.plot(futureYears, over650, 'o-', color ='red', label = 'Current capacity')
plt.plot(futureYears, over800, 'o-', color ='blue', label = 'New capacity')
plt.xlabel(f'Years from now ({g:.0%} growth per year)')
plt.ylabel('Expected days over capacity per year')
plt.legend()
plt.savefig(os.path.join(os.path.dirname(__file__), 'scenario_growth.png'), bbox_inches = 'tight')
plt.show()

