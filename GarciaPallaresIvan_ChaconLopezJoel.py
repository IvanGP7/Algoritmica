import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import time

# Variables globals
GROUND_TRUTH = 0.325 # Valor calculat analíticament de la integral
MU = 0.85
SIGMA = 0.05

# Pressupostos de FE
B_VALUES = [500, 1500, 5000, 15000, 50000]
# Repeticions independents 
R_VALUES = [50, 100]

def grid_FB(n_points, mu, sigma):
    # Dividim [0,1] en n_points intervals iguals
    dx = (1 - 0) / n_points

    # Generem els punts mitjos: dx/2, dx/2 + dx,...
    x_mid = np.linspace(dx/2, 1 - dx/2, n_points)
    
    # Avaluem f(x) en aquests punts
    y_vals = 0.2 + np.exp(-((x_mid - mu)**2) / (2 * sigma**2))
    
    # Estimació de la integral per la regla del rectangle
    # Base = 1/n_points, Altura = y_vals
    estimation = np.sum(y_vals) * dx
    return estimation, x_mid, y_vals


def mc_simple(B, R, mu, sigma):
    # Variables per guardar resultats
    estimates = []
    times = []
    
    for r in range(R):
        start_time = time.time()
        
        # Generem B mostres uniformes a l'interval [0, 1]
        x_rand = np.random.uniform(0, 1, B)
        
        # Evaluem f(x) en aquestes mostres
        y_vals = 0.2 + np.exp(-((x_rand - mu)**2) / (2 * sigma**2))
        
        # Estimador MC Simple -> Mitjana de f(x) sobre les mostres
        est = np.mean(y_vals)
        
        end_time = time.time()
        
        estimates.append(est)
        times.append(end_time - start_time)
        
    return np.array(estimates), np.array(times)

def mc_is(B, R, mu, sigma, alpha):
    estimates = []
    times = []
    
    # Definim la normal truncada
    a, b = (0 - mu) / sigma, (1 - mu) / sigma
    trunc_norm = stats.truncnorm(a, b, loc=mu, scale=sigma)
    
    for r in range(R):
        start_time = time.time()
        
        # Generem B mostres de la mescla
        n_tn = np.random.binomial(B, alpha)
        n_unif = B - n_tn            
        
        samples_tn = trunc_norm.rvs(size=n_tn)
        samples_unif = np.random.uniform(0, 1, size=n_unif)
        
        x_samples = np.concatenate([samples_tn, samples_unif])
        
        # Calculem la pdf de la mescla (funció de densitat de probabilitat)
        pdf_vals = (1 - alpha) * 1.0 + alpha * trunc_norm.pdf(x_samples)
        
        # Calculam els pesos
        weights = 1.0 / pdf_vals
        
        # Evaluem f(x) en les mostres
        f_vals = 0.2 + np.exp(-((x_samples - mu)**2) / (2 * sigma**2))
        
        # Estimador MC IS
        est = np.mean(f_vals * weights)
        
        end_time = time.time()
        estimates.append(est)
        times.append(end_time - start_time)
        
    return np.array(estimates), np.array(times)



def grid_execution():
    # Variables per guardar resultats
    grid_estimations = []
    grid_errors = []
    grid_times = []
    
    # Punts de mostra per dibuixar la funció després
    sample_points_x = None
    sample_points_y = None

    print("--- TAULA GRID ---")
    for i, n in enumerate(B_VALUES): # limitem AF a B_VALUES
        time_start = time.time()
        
        est, x_vals, y_vals = grid_FB(n, MU, SIGMA) 
        
        time_end = time.time()
        elapsed = time_end - time_start
        
        # Calculem error absolut
        error = abs(est - I_REAL)
        if (n == 50000):
            error_n50k = error
        
        # Guardem les dades
        grid_estimations.append(est)
        grid_errors.append(error)
        grid_times.append(elapsed)
        
        # Guardem el grid de punts per N=500, la representació és més clara
        if (n == 500):
            sample_points_x = x_vals
            sample_points_y = y_vals
            
        print(f"N={n}: Est={est:.6f}, Error={error:.4e}, Time={elapsed:.6f}s")

    print("\n--- UNA PROVA EXPERIMENTAL AMB N = 50 i N = 500000 ---") # Per veure que realment a partir d'una N minima, ja dona igual si la augmentem mes que l'error no millora gaire
    for test_n in [50, 500000]:
        time_start = time.time()
        
        est, _, _ = grid_FB(test_n, MU, SIGMA) 
        
        time_end = time.time()
        elapsed = time_end - time_start
        
        error = abs(est - I_REAL)
        if (test_n == 500000):
            error_n500k = error
        
        print(f"N={test_n}: Est={est:.6f}, Error={error:.4e}, Time={elapsed:.6f}s")

    print(f"\nComparativa errors N=50000 vs N=500000: {error_n50k:.4e} vs {error_n500k:.4e}, pràcticament iguals\ni molt més costós (500k) a nivell d'execució.") # Gairebé iguals, com s'esperava
    print("\nEn canvi, amb N=50 l'error és molt més gran, com s'esperava també, per tant, eren necessaries més iteracions.")
    print("\nEn aquest cas (grid), no serveix de res utilitzar els diferents valors de R ja que és determinista,\nindependentment de les vegades que s'executi amb la mateixa B, el resultat serà el mateix.\nPer això cada B només l'executem una vegada en grid.\n")
    
    return grid_estimations, grid_errors, grid_times, sample_points_x, sample_points_y



def montecarlo_simple_execution():
    print("\n--- TAULA MC SIMPLE ---")
    
    mc_simple_results=[]
    for b in B_VALUES:
        for r in R_VALUES:         
            estimates, times = mc_simple(b, r, MU, SIGMA)
            
            # Calculem mètriques estadístiques: mitjana, bias, rmse, temps mig
            mean_estimate = np.mean(estimates)
            bias = mean_estimate - I_REAL
            rmse = np.sqrt(np.mean((estimates - I_REAL)**2))
            avg_time = np.mean(times)
            
            # Guardem les dades per la taula
            mc_simple_results.append({
                'B': b,
                'R': r,
                'Mean_Est': mean_estimate,
                'Bias': bias,
                'RMSE': rmse,
                'Avg_Time': avg_time
            })
            
            # Ho mostrem per pantalla
            print(f"MC Simple (B={b}, R={r}):")
            print(f"\tMean Estimation = {mean_estimate:.6f}")
            print(f"\tBias            = {bias:.6e}")
            print(f"\tRMSE            = {rmse:.6e}")
            print(f"\tAvg Time per Run= {avg_time:.6f} s\n")

    df_mc_simple = pd.DataFrame(mc_simple_results)
    return df_mc_simple


def montecarlo_sample_execution():
    print("\n--- TAULA MC IS, alpha = Y ---")
    # Variables per guardar resultats
    mc_is_results = [] 
    ALPHAS = [0.3, 0.7]     # Valors d'alpha definits al pdf

    for alpha in ALPHAS:
        print(f"--- Alpha = {alpha} ---")
        for b in B_VALUES:
            for r in R_VALUES:
                # Executem MC IS
                estimates, times = mc_is(b, r, MU, SIGMA, alpha)
                
                # Calculem mètriques estadístiques
                mean_estimate = np.mean(estimates)
                bias = mean_estimate - I_REAL
                rmse = np.sqrt(np.mean((estimates - I_REAL)**2))
                avg_time = np.mean(times)
                
                # Guardem les dades per la taula
                mc_is_results.append({
                    'Alpha': alpha,
                    'B': b,
                    'R': r,
                    'Mean_Est': mean_estimate,
                    'Bias': bias,
                    'RMSE': rmse,
                    'Avg_Time': avg_time
                })
                
                # Ho mostrem per pantalla
                print(f"MC IS (Alpha={alpha}, B={b}, R={r}):")
                print(f"\tMean Estimation = {mean_estimate:.6f}")
                print(f"\tBias            = {bias:.6e}")
                print(f"\tRMSE            = {rmse:.6e}")
                print(f"\tAvg Time per Run= {avg_time:.6f} s\n")

    df_mc_is = pd.DataFrame(mc_is_results)
    return df_mc_is


if __name__ == "__main__":
    
    I_REAL = GROUND_TRUTH # Valor calculat analíticament de la integral 
    print(f"Ground Truth (I): {I_REAL}\n")

    grid_estimations, grid_errors, grid_times, sample_points_x, sample_points_y = grid_execution()
    mc_simple_results = montecarlo_simple_execution()
    df_mc_is = montecarlo_sample_execution()

    df_mc_simple = pd.DataFrame(mc_simple_results)
    
    # Utilitzarem R=100 per als gràfics comparatius, ja que les estimacions són més estables
    R_PLOT = 100 
    
    # Filtrem les dades per R=100
    df_simple_plot = df_mc_simple[df_mc_simple['R'] == R_PLOT]
    df_is_plot = df_mc_is[df_mc_is['R'] == R_PLOT]

    # --- GRÀFIC 1: BRUTE FORCE FUNCTION + GRID ---
    plt.figure(figsize=(10, 6))
    
    # Dibuixem f(x)
    x_smooth = np.linspace(0, 1, 1000)
    y_smooth = 0.2 + np.exp(-((x_smooth - MU)**2) / (2 * SIGMA**2))
    plt.plot(x_smooth, y_smooth, label='f(x) Real', color='blue')
    
    # Dibuixem els punts del grid i la funció a aquests punts
    plt.plot(sample_points_x, sample_points_y, 'r.', markersize=3, label='Grid Points', alpha=0.6)
    plt.scatter(sample_points_x, np.zeros_like(sample_points_x), color='red', s=1, alpha=0.5, label='Midpoints on X')
    
    plt.title("BRUTE FORCE: FUNCTION + GRID")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- GRÀFIC 2: BRUTE FORCE ERROR (Log-Log) ---
    plt.figure(figsize=(10, 6))
    plt.loglog(B_VALUES, grid_errors, 'o-', label='Grid Error')
    plt.title("BRUTE FORCE: ERROR")
    plt.xlabel("Function Evaluations (N)")
    plt.ylabel("Absolute Error |I - I_est|")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend()
    plt.show()

    # ================================================
    # GRÀFICS MONTE CARLO (CONVERGÈNCIA I HISTOGRAMES)
    # ================================================
    
    # Configuració per a les gràfiques 
    B_DEMO = 15000          # B fix per a les estimacions
    B_MAX = max(B_VALUES)   # B màxim per a les comparatives
    np.random.seed(123)     # Afegim un sol seed per reproducibilitat

    # --- GRÀFIC 3: MC SIMPLE: CONVERGÈNCIA ---
    # Generem una trajectòria de mostres per veure la convergència
    x_rand = np.random.uniform(0, 1, B_MAX)
    y_vals = 0.2 + np.exp(-((x_rand - MU)**2) / (2 * SIGMA**2))

    # Estimació acumulada
    convergence_simple = np.cumsum(y_vals) / np.arange(1, B_MAX + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, B_MAX + 1), convergence_simple, label='MC Simple Trajectory')
    plt.axhline(I_REAL, color='r', linestyle='--', label='Ground Truth')
    plt.title(f"MC SIMPLE: CONVERGENCE")
    plt.xlabel("B-values (samples)")
    plt.ylabel("Cumulative Estimation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- GRÀFIC 4: MC SIMPLE: HISTOGRAMA ---
    # Executem MC Simple amb B_DEMO i R_PLOT per obtenir múltiples estimacions
    estimates_hist, _ = mc_simple(B_DEMO, R_PLOT, MU, SIGMA)
    
    plt.figure(figsize=(10, 6))
    plt.hist(estimates_hist, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(I_REAL, color='red', linestyle='dashed', linewidth=2, label='Ground Truth')
    plt.title("MC SIMPLE: HISTOGRAM")
    plt.xlabel("Estimated Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    # --- GRÀFIC 5: MC IMPORTANCE SAMPLING: CONVERGÈNCIA ---
    plt.figure(figsize=(10, 6))
    plt.axhline(I_REAL, color='r', linestyle='--', label='Ground Truth')
    np.random.seed(123)  # Resetem el seed per coherència
    
    # Generem la normal truncada 
    a_trunc, b_trunc = (0 - MU) / SIGMA, (1 - MU) / SIGMA
    trunc_norm = stats.truncnorm(a_trunc, b_trunc, loc=MU, scale=SIGMA)
    
    for alpha in [0.3, 0.7]:
        # Mostreig de la mescla
        is_from_tn = np.random.binomial(1, alpha, B_MAX)
        
        # Generem les mostres
        samples = np.zeros(B_MAX)

        # On és 1 -> trunc norm, on és 0 -> uniform
        count_tn = np.sum(is_from_tn)
        count_unif = B_MAX - count_tn
        
        if count_tn > 0:
            samples[is_from_tn == 1] = trunc_norm.rvs(size=count_tn)
        if count_unif > 0:
            samples[is_from_tn == 0] = np.random.uniform(0, 1, size=count_unif)
            
        # Calcul dels pesos
        pdf_vals = (1 - alpha) * 1.0 + alpha * trunc_norm.pdf(samples)
        weights = 1.0 / pdf_vals
        
        # Evaluació de f(x) * pesos
        f_vals = 0.2 + np.exp(-((samples - MU)**2) / (2 * SIGMA**2))
        weighted_vals = f_vals * weights
        
        # Acumulació per a la convergència
        convergence_is = np.cumsum(weighted_vals) / np.arange(1, B_MAX + 1)
        plt.plot(range(1, B_MAX + 1), convergence_is, label=f'Alpha={alpha}')

    plt.title(f"MC IS: CONVERGENCE. B = X, alpha = Y")
    plt.xlabel("n (samples)")
    plt.ylabel("Cumulative Estimation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- GRÀFIC 6: MC IMPORTANCE SAMPLING: HISTOGRAMA ---
    plt.figure(figsize=(10, 6))
    for alpha in [0.3, 0.7]:
        estimates_is_hist, _ = mc_is(B_DEMO, R_PLOT, MU, SIGMA, alpha)
        plt.hist(estimates_is_hist, bins=20, alpha=0.5, label=f'Alpha={alpha}')

    plt.axvline(I_REAL, color='red', linestyle='dashed', linewidth=2, label='Ground Truth')
    plt.title("MC IS: HISTOGRAM. B = X, alpha = Y")
    plt.xlabel("Estimated Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    # ==========================================
    # GRÀFICS COMPARATIUS (RMSE)
    # ==========================================
    
    # [RMSE vs AF (Log-Log)
    plt.figure(figsize=(10, 6))
    
    # Grid Uniform
    plt.loglog(B_VALUES, grid_errors, 'o--', label='Grid Uniform', color='gray')
    
    # MC Simple
    plt.loglog(df_simple_plot['B'], df_simple_plot['RMSE'], 'o-', label='MC Simple')
    
    # MC IS (Alpha 0.3)
    data_is_03 = df_is_plot[df_is_plot['Alpha'] == 0.3]
    plt.loglog(data_is_03['B'], data_is_03['RMSE'], 's-', label='MC IS (alpha=0.3)')
    
    # MC IS (Alpha 0.7)
    data_is_07 = df_is_plot[df_is_plot['Alpha'] == 0.7]
    plt.loglog(data_is_07['B'], data_is_07['RMSE'], '^-', label='MC IS (alpha=0.7)')
    
    plt.title("COMPARISON: RMSE vs Function Evaluations (Log-Log)")
    plt.xlabel("Function Evaluations (B)")
    plt.ylabel("RMSE")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend()
    plt.show()
    
    # RMSE vs temps (Log-Log)
    plt.figure(figsize=(10, 6))
    
    plt.loglog(grid_times, grid_errors, 'o--', label='Grid Uniform', color='gray')
    plt.loglog(df_simple_plot['Avg_Time'], df_simple_plot['RMSE'], 'o-', label='MC Simple')
    plt.loglog(data_is_03['Avg_Time'], data_is_03['RMSE'], 's-', label='MC IS (alpha=0.3)')
    plt.loglog(data_is_07['Avg_Time'], data_is_07['RMSE'], '^-', label='MC IS (alpha=0.7)')
    
    plt.title("COMPARISON: RMSE vs Execution Time (Log-Log)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("RMSE")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend()
    plt.show()