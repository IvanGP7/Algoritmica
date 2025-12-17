import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import quad
import time

#Variables globales
RESULT_I = 0.325
MU = 0.85
SIGMA = 0.05
# Presupuestos de evaluaciones (B)
B_VALUES = [500, 1500, 5000, 15000, 50000]
# Repeticiones (R)
R_VALUES = [50, 100]

# Función para obtener el valor real de la integral
def get_ground_truth(mu, sigma):

    def func(x):
        return 0.2 + np.exp(-((x - mu)**2) / (2 * sigma**2))
    
    val, error = quad(func, 0, 1)
    return val

def solve_grid(n_points, mu, sigma):
    # Puntos medios en el intervalo [0, 1]
    dx = 1.0 / n_points
    # Generar puntos: dx/2, dx/2 + dx, ...
    x_mid = np.linspace(dx/2, 1 - dx/2, n_points)
    
    # Evaluar función
    y_vals = 0.2 + np.exp(-((x_mid - mu)**2) / (2 * sigma**2))
    
    # La integral es la suma de áreas de rectángulos (base * altura)
    # Base = 1/n_points, Altura = y_vals
    estimation = np.sum(y_vals) * (1.0 / n_points)
    return estimation, x_mid, y_vals


def solve_mc_simple(B, R, mu, sigma):
    estimates = []
    times = []
    
    for r in range(R):
        start_time = time.time()
        
        # 1. Generar B muestras uniformes U ~ [0, 1]
        x_rand = np.random.uniform(0, 1, B)
        
        # 2. Evaluar f(x)
        y_vals = 0.2 + np.exp(-((x_rand - mu)**2) / (2 * sigma**2))
        
        # 3. Estimador MC = promedio de las evaluaciones
        est = np.mean(y_vals)
        
        end_time = time.time()
        
        estimates.append(est)
        times.append(end_time - start_time)
        
    return np.array(estimates), np.array(times)

def solve_mc_is(B, R, mu, sigma, alpha):
    estimates = []
    times = []
    
    # Definir la Normal Truncada en [0, 1]
    a, b = (0 - mu) / sigma, (1 - mu) / sigma
    trunc_norm = stats.truncnorm(a, b, loc=mu, scale=sigma)
    
    for r in range(R):
        start_time = time.time()
        
        # 1. Muestreo de la mezcla (Mixture Sampling)
        # Decidimos cuántas muestras vienen de TN y cuántas de Unif
        n_tn = np.random.binomial(B, alpha) # Número de muestras de la normal
        n_unif = B - n_tn                   # El resto de la uniforme
        
        samples_tn = trunc_norm.rvs(size=n_tn)
        samples_unif = np.random.uniform(0, 1, size=n_unif)
        
        x_samples = np.concatenate([samples_tn, samples_unif])
        
        # 2. Calcular la densidad q(x) para los pesos
        # q(x) = (1-alpha)*1 + alpha * pdf_trunc_norm(x)
        pdf_vals = (1 - alpha) * 1.0 + alpha * trunc_norm.pdf(x_samples)
        
        # 3. Calcular pesos w(x) = 1 / q(x) 
        weights = 1.0 / pdf_vals
        
        # 4. Evaluar f(x) original
        f_vals = 0.2 + np.exp(-((x_samples - mu)**2) / (2 * sigma**2))
        
        # 5. Estimador IS [cite: 37]
        # Suma(f(x)*w(x)) / n
        est = np.mean(f_vals * weights)
        
        end_time = time.time()
        estimates.append(est)
        times.append(end_time - start_time)
        
    return np.array(estimates), np.array(times)



def grid_execution():
    # Listas para guardar los resultados de todas las iteraciones
    grid_estimations = []
    grid_errors = []
    grid_times = []
    
    # Variable para guardar los puntos de UNA sola iteración (para el primer gráfico)
    # Usaremos una N pequeña (la primera de B_VALUES) para que el gráfico se vea bien
    sample_points_x = None
    sample_points_y = None

    print("--- BRUTE FORCE (GRID) ---")
    for i, n in enumerate(B_VALUES):
        time_start = time.time()
        
        # Asumiendo que solve_grid devuelve: estimación, array_x, array_y
        est, x_vals, y_vals = solve_grid(n, MU, SIGMA) 
        
        time_end = time.time()
        elapsed = time_end - time_start
        
        # Calcular error absoluto
        error = abs(est - I_REAL)
        
        # Guardar datos
        grid_estimations.append(est)
        grid_errors.append(error)
        grid_times.append(elapsed)
        
        # Guardamos los puntos de la primera iteración (N=500) para dibujar luego
        if i == 0:
            sample_points_x = x_vals
            sample_points_y = y_vals
            
        print(f"N={n}: Est={est:.6f}, Error={error:.4e}, Time={elapsed:.6f}s")
    return grid_estimations, grid_errors, grid_times, sample_points_x, sample_points_y



def montecarlo_simple_execution():
    print("\n--- MONTE CARLO SIMPLE ---")
    
    mc_simple_results=[]
    for b in B_VALUES:
        for r in R_VALUES:         

            # 1. Ejecutar el método
            estimates, times = solve_mc_simple(b, r, MU, SIGMA)
            
            # 2. Calcular métricas estadísticas [cite: 24, 25, 27]
            mean_estimate = np.mean(estimates)
            bias = mean_estimate - I_REAL
            rmse = np.sqrt(np.mean((estimates - I_REAL)**2))
            avg_time = np.mean(times)
            
            # 3. Guardar para la tabla (Pandas)
            mc_simple_results.append({
                'B': b,
                'R': r,
                'Mean_Est': mean_estimate,
                'Bias': bias,
                'RMSE': rmse,
                'Avg_Time': avg_time
            })
            
            # 4. Imprimir por pantalla (Estilo "Punto 1")
            print(f"MC Simple (B={b}, R={r}):")
            print(f"\tMean Estimation = {mean_estimate:.6f}")
            print(f"\tBias            = {bias:.6e}")
            print(f"\tRMSE            = {rmse:.6e}")
            print(f"\tAvg Time per Run= {avg_time:.6f} s\n")

    # Para ver la tabla final (opcional ahora, útil para el entregable)
    df_mc_simple = pd.DataFrame(mc_simple_results)
    return df_mc_simple


def montecarlo_sample_execution():
    
    print("\n--- MONTE CARLO IMPORTANCE SAMPLING (IS) ---")
    mc_is_results = [] # Lista para guardar datos para la Tabla posterior
    ALPHAS = [0.3, 0.7] # Valores de alpha especificados en el PDF 

    for alpha in ALPHAS:
        print(f"--- Alpha = {alpha} ---")
        for b in B_VALUES:
            for r in R_VALUES:
                
                # 1. Ejecutar el método
                # solve_mc_is debe devolver: (array_de_estimaciones, array_de_tiempos)
                # Recuerda pasar 'alpha' a tu función
                estimates, times = solve_mc_is(b, r, MU, SIGMA, alpha)
                
                # 2. Calcular métricas estadísticas [cite: 43, 44, 46]
                mean_estimate = np.mean(estimates)
                bias = mean_estimate - I_REAL
                rmse = np.sqrt(np.mean((estimates - I_REAL)**2))
                avg_time = np.mean(times)
                
                # 3. Guardar para la tabla
                mc_is_results.append({
                    'Alpha': alpha,
                    'B': b,
                    'R': r,
                    'Mean_Est': mean_estimate,
                    'Bias': bias,
                    'RMSE': rmse,
                    'Avg_Time': avg_time
                })
                
                # 4. Imprimir por pantalla
                print(f"MC IS (Alpha={alpha}, B={b}, R={r}):")
                print(f"\tMean Estimation = {mean_estimate:.6f}")
                print(f"\tBias            = {bias:.6e}")
                print(f"\tRMSE            = {rmse:.6e}")
                print(f"\tAvg Time per Run= {avg_time:.6f} s\n")

    # Para ver la tabla final
    df_mc_is = pd.DataFrame(mc_is_results)
    return df_mc_is


if __name__ == "__main__":
    
    I_REAL = get_ground_truth(MU, SIGMA)
    print(f"Ground Truth (I): {I_REAL}\n")

    grid_estimations, grid_errors, grid_times, sample_points_x, sample_points_y = grid_execution()
    mc_simple_results = montecarlo_simple_execution()
    df_mc_is = montecarlo_sample_execution()

    # Asumiendo que mc_simple_results es una lista de diccionarios
    df_mc_simple = pd.DataFrame(mc_simple_results)
    
    # df_mc_is ya viene como DataFrame según tu código
    # Filtrar por los R=50 o R=100 (usaremos R=100 para ser más robustos en las gráficas comparativas)
    R_PLOT = 100 
    
    # Filtrar datos para los gráficos comparativos
    df_simple_plot = df_mc_simple[df_mc_simple['R'] == R_PLOT]
    df_is_plot = df_mc_is[df_mc_is['R'] == R_PLOT]

    # --- GRÁFICO 1: BRUTE FORCE FUNCTION + GRID ---
    plt.figure(figsize=(10, 6))
    
    # Dibujar f(x) suave
    x_smooth = np.linspace(0, 1, 1000)
    y_smooth = 0.2 + np.exp(-((x_smooth - MU)**2) / (2 * SIGMA**2))
    plt.plot(x_smooth, y_smooth, label='f(x) Real', color='blue')
    
    # Dibujar los puntos del Grid (usamos los de la ejecución con N=500 que guardaste)
    # sample_points_x e y vienen de tu función grid_execution
    plt.plot(sample_points_x, sample_points_y, 'r.', markersize=3, label='Grid Points', alpha=0.6)
    plt.scatter(sample_points_x, np.zeros_like(sample_points_x), color='red', s=1, alpha=0.5, label='Midpoints on X')
    
    plt.title("BRUTE FORCE: FUNCTION + GRID")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- GRÁFICO 2: BRUTE FORCE ERROR (Log-Log) ---
    plt.figure(figsize=(10, 6))
    plt.loglog(B_VALUES, grid_errors, 'o-', label='Grid Error')
    plt.title("BRUTE FORCE: ERROR")
    plt.xlabel("Function Evaluations (N)")
    plt.ylabel("Absolute Error |I - I_est|")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend()
    plt.show()

    # ... (Después de tus gráficos de Brute Force) ...

    # ==========================================
    # GRÁFICOS MONTE CARLO (CONVERGENCIA E HISTOGRAMAS)
    # ==========================================
    
    # Configuración para las demos
    B_DEMO = 15000  # Un buen número para ver la convergencia
    
    # --- 3. MC SIMPLE: CONVERGENCIA (Trayectoria de una sola ejecución) ---
    # Generamos los datos paso a paso manualmente
    x_rand = np.random.uniform(0, 1, B_DEMO)
    y_vals = 0.2 + np.exp(-((x_rand - MU)**2) / (2 * SIGMA**2))
    # Estimación acumulada: (y1, (y1+y2)/2, ...)
    convergence_simple = np.cumsum(y_vals) / np.arange(1, B_DEMO + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, B_DEMO + 1), convergence_simple, label='MC Simple Trajectory')
    plt.axhline(I_REAL, color='r', linestyle='--', label='Ground Truth')
    plt.title(f"MC SIMPLE: CONVERGENCE (B={B_DEMO})")
    plt.xlabel("n (samples)")
    plt.ylabel("Cumulative Estimation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- 4. MC SIMPLE: HISTOGRAMA ---
    # Ejecutamos 100 veces para obtener 100 estimaciones y ver su distribución
    # Reutilizamos tu función solve_mc_simple
    estimates_hist, _ = solve_mc_simple(15000, 100, MU, SIGMA)
    
    plt.figure(figsize=(10, 6))
    plt.hist(estimates_hist, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(I_REAL, color='red', linestyle='dashed', linewidth=2, label='Ground Truth')
    plt.title("MC SIMPLE: HISTOGRAM (B=15000, R=100)")
    plt.xlabel("Estimated Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    # --- 5. MC IMPORTANCE SAMPLING: CONVERGENCIA ---
    plt.figure(figsize=(10, 6))
    plt.axhline(I_REAL, color='r', linestyle='--', label='Ground Truth')
    
    # Generamos una trayectoria para cada Alpha (0.3 y 0.7)
    from scipy import stats # Necesario para la normal truncada
    a_trunc, b_trunc = (0 - MU) / SIGMA, (1 - MU) / SIGMA
    trunc_norm = stats.truncnorm(a_trunc, b_trunc, loc=MU, scale=SIGMA)
    
    for alpha in [0.3, 0.7]:
        # Muestreo mezcla (Mixture) paso a paso
        # 1. Decidir origen de cada muestra (Binomial)
        is_from_tn = np.random.binomial(1, alpha, B_DEMO)
        
        # 2. Generar valores
        samples = np.zeros(B_DEMO)
        # Donde is_from_tn es 1, usamos TN, donde es 0, usamos Uniforme
        count_tn = np.sum(is_from_tn)
        count_unif = B_DEMO - count_tn
        
        if count_tn > 0:
            samples[is_from_tn == 1] = trunc_norm.rvs(size=count_tn)
        if count_unif > 0:
            samples[is_from_tn == 0] = np.random.uniform(0, 1, size=count_unif)
            
        # 3. Calcular Pesos
        pdf_vals = (1 - alpha) * 1.0 + alpha * trunc_norm.pdf(samples)
        weights = 1.0 / pdf_vals
        
        # 4. Evaluar f(x) * w(x)
        f_vals = 0.2 + np.exp(-((samples - MU)**2) / (2 * SIGMA**2))
        weighted_vals = f_vals * weights
        
        # 5. Acumular
        convergence_is = np.cumsum(weighted_vals) / np.arange(1, B_DEMO + 1)
        plt.plot(range(1, B_DEMO + 1), convergence_is, label=f'Alpha={alpha}')

    plt.title(f"MC IMPORTANCE SAMPLING: CONVERGENCE (B={B_DEMO})")
    plt.xlabel("n (samples)")
    plt.ylabel("Cumulative Estimation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # ==========================================
    # GRÁFICOS COMPARATIVOS (RMSE)
    # ==========================================
    
    # [cite_start]--- 6. RMSE vs AF (Log-Log) [cite: 50, 51] ---
    plt.figure(figsize=(10, 6))
    
    # Grid (Fuerza Bruta)
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
    
    # [cite_start]--- 7. RMSE vs TIEMPO (Log-Log) [cite: 51] ---
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