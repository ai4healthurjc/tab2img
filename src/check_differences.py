from scipy.stats import wilcoxon
HenG= [0.81,0.85,0.88,0.88,0.90]
individual= [0.83,0.83,0.83,0.89,0.89]
# Realizar la prueba de Wilcoxon
statistic, p_value = wilcoxon(individual, HenG)

print(f"Wilcoxon statistic: {statistic}")
print(f"P-value: {p_value}")

