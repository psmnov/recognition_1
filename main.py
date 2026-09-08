import struct
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def mle_mean(X):
    n = X.shape[0]
    total = np.zeros(X.shape[1])
    for i in range(n):
        total = total + X[i]
    return total / n

def mle_cov(X, mu):
    n = X.shape[0]
    d = X.shape[1]
    total = np.zeros((d, d))
    for i in range(n):
        diff = (X[i] - mu).reshape(d, 1)
        total = total + diff @ diff.T
    return total / n

def g_se(x):
    diff = x - mu_se
    return -0.5 * diff @ sigma_se_inv @ diff - 0.5 * log_det_se + math.log(0.5)

def g_ve(x):
    diff = x - mu_ve
    return -0.5 * diff @ sigma_ve_inv @ diff - 0.5 * log_det_ve + math.log(0.5)

def log_normal_pdf(xi, m, s):
    # ln f(x) = -ln(sigma) - 0.5*ln(2*pi) - (x-mu)^2 / (2*sigma^2)
    return -math.log(s) - 0.5 * math.log(2 * math.pi) - ((xi - m) ** 2) / (2 * s ** 2)

def log_likelihood(x_data, m, s):
    total = 0.0
    for xi in x_data:
        total += log_normal_pdf(xi, m, s)
    return total

if __name__ == '__main__':
    with open('IRIS.DAT', 'rb') as f:
        raw_bytes = f.read()

    # ЗАДАНИЕ 1
    n_floats = len(raw_bytes) // 4  # получится 600 (150 * 4 = 50*12)
    print(n_floats)

    flat_values = struct.unpack('<' + 'f' * n_floats, raw_bytes) #кортеж из чисел типа float размером 600, чем младше байт - тем младше адрес

    data = np.array(flat_values, dtype=np.float32).reshape(150, 4) #превращаем кортеж в массивы массивов по каждому цветку
    se = data[0:50]
    ve = data[50:100]
    vi= data[100:150]
    print("Se, первые 3 объекта:\n", se[:3])
    print("Ve, первые 3 объекта:\n", ve[:3])
    print("Vi, первые 3 объекта:\n", vi[:3])

    ve_x3_1_25 = ve[0:25, 2] #вырезал третью координату из ve
    mu = sum(ve_x3_1_25) / len(ve_x3_1_25)
    print("mu: ", mu) #получается 4.3120003
    sigma = math.sqrt(sum((xi - mu)**2 for xi in ve_x3_1_25)/len(ve_x3_1_25))
    print("sigma: ", sigma) #sigma:  0.4348057141461965

    #ЗАДАНИЕ 2

    se_x2_reversed = se[:, 1][::-1]  #столбец x2, развёрнутый
    se_x3_normal = se[:, 2] #столбец x3, без изменений

    ve_x2_reversed = ve[:, 1][::-1]
    ve_x3_normal = ve[:, 2]

    # таблица из двух колонок и 50 строк для каждой пары
    Se = np.column_stack((se_x2_reversed, se_x3_normal))
    Ve = np.column_stack((ve_x2_reversed, ve_x3_normal))

    #функции подсчета вектора среднего и ковариационной матрицы описаны выше
    mu_se = mle_mean(Se) #среднее
    sigma_se = mle_cov(Se, mu_se) #ковариационная матрица

    mu_ve = mle_mean(Ve) #среднее
    sigma_ve = mle_cov(Ve, mu_ve) #ковариационная матрица

    print("mu_Se: ", mu_se)
    print("mu_Ve: ", mu_ve)
    print("Sigma_Se:\n", sigma_se)
    print("Sigma_Ve:\n", sigma_ve)

    #подготовка компонентов формулы дискриминантной функции
    sigma_se_inv = np.linalg.inv(sigma_se) #ищем обратные матрицы
    sigma_ve_inv = np.linalg.inv(sigma_ve)

    log_det_se = np.log(np.linalg.det(sigma_se)) #определители
    log_det_ve = np.log(np.linalg.det(sigma_ve))

    #Построение графика с границей решения

    all_x2 = np.concatenate((Se[:, 0], Ve[:, 0]))
    all_x3 = np.concatenate((Se[:, 1], Ve[:, 1]))

    x2_grid = np.linspace(all_x2.min() - 0.5, all_x2.max() + 0.5, 300)
    x3_grid = np.linspace(all_x3.min() - 0.5, all_x3.max() + 0.5, 300)

    X2, X3 = np.meshgrid(x2_grid, x3_grid)


    diff_grid = np.zeros_like(X2)
    for i in range(X2.shape[0]):
        for j in range(X2.shape[1]):
            point = np.array([X2[i, j], X3[i, j]])
            diff_grid[i, j] = g_se(point) - g_ve(point)

    fig, ax = plt.subplots(figsize=(9, 7))

    #все точки
    ax.scatter(Se[:, 0], Se[:, 1], color='red', label='Se', edgecolor='black')
    ax.scatter(Ve[:, 0], Ve[:, 1], color='blue', label='Ve', edgecolor='black')

    ax.scatter(*mu_se, color='red', marker='X', s=200, edgecolor='black') #вывод средней точки
    ax.scatter(*mu_ve, color='blue', marker='X', s=200, edgecolor='black') #вывод средней точки
    #вывод на график решающего правила
    ax.contour(X2, X3, diff_grid, levels=[0], colors='black', linewidths=2)
    #названия осей
    ax.set_xlabel('x2 (обратный порядок)')
    ax.set_ylabel('x3 (обычный порядок)')
    ax.legend()
    plt.savefig('bayes_decision_se_ve.png', dpi=150)
    print("График сохранён")

    #задание 3
    #Часть 1: сетка по mu при фиксированной sigma (берём sigma из Задания 1)
    mu_grid = np.arange(mu - 2.0, mu + 2.0 + 0.05, 0.1)
    mu_grid = np.round(mu_grid, 1)  # убираем погрешность float при накоплении шага 0.1

    log_lik_over_mu = []
    for mu_candidate in mu_grid:
        log_lik_over_mu.append(log_likelihood(ve_x3_1_25, mu_candidate, sigma))

    best_index_mu = int(np.argmax(log_lik_over_mu))
    mu_prime = mu_grid[best_index_mu]

    print("mu' (максимум на сетке):", mu_prime)
    print("mu (аналитический MLE, Задание 1):", mu)

    plt.figure(figsize=(9, 5))
    plt.plot(mu_grid, log_lik_over_mu, marker='o', markersize=3)
    plt.axvline(mu_prime, color='red', linestyle='--', label=f"mu' = {mu_prime:.1f}")
    plt.axvline(mu, color='green', linestyle=':', label=f"mu (MLE) = {mu:.4f}")
    plt.xlabel('mu')
    plt.ylabel('ln P(X | mu)')
    plt.title('Аналог функции правдоподобия по mu')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('likelihood_mu.png', dpi=150)
    plt.close()

    # Часть 2: сетка по sigma при mu = mu_prime (найденном в Части 1)
    sigma_grid = np.arange(0.1, sigma * 3.0, 0.1)
    sigma_grid = np.round(sigma_grid, 1)

    log_lik_over_sigma = []
    for sigma_candidate in sigma_grid:
        log_lik_over_sigma.append(log_likelihood(ve_x3_1_25, mu_prime, sigma_candidate))

    best_index_sigma = int(np.argmax(log_lik_over_sigma))
    sigma_prime = sigma_grid[best_index_sigma]

    print("sigma' (максимум на сетке):", sigma_prime)
    print("sigma (аналитический MLE, Задание 1):", sigma)

    plt.figure(figsize=(9, 5))
    plt.plot(sigma_grid, log_lik_over_sigma, marker='o', markersize=3)
    plt.axvline(sigma_prime, color='red', linestyle='--', label=f"sigma' = {sigma_prime:.1f}")
    plt.axvline(sigma, color='green', linestyle=':', label=f"sigma (MLE) = {sigma:.4f}")
    plt.xlabel('sigma')
    plt.ylabel('ln P(X | sigma)')
    plt.title("Аналог функции правдоподобия по sigma (mu = mu')")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('likelihood_sigma.png', dpi=150)
    plt.close()

    print()
    print("Сравнение: (mu', sigma') =", (mu_prime, sigma_prime), " vs (mu, sigma) =", (mu, sigma))



