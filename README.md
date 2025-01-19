# Rapport sur l’assignation des fréquences par PSO

## Introduction

Dans les réseaux cellulaires, l’assignation efficace des fréquences est un problème critique. Les fréquences doivent être distribuées aux cellules de manière à minimiser les interferences tout en respectant des contraintes de séparation minimales basées sur une matrice de distances (également appelée matrice C1). Ce problème est complexe à résoudre par des méthodes exactes, surtout dans les réseaux de grande taille.

Pour aborder ce problème, nous avons utilisé l’**optimisation par essaim de particules** (PSO), un algorithme inspiré par le comportement collectif observé dans la nature, tel que les vols d'oiseaux ou les bancs de poissons. Cet algorithme cherche à optimiser une solution en itérant sur un groupe de particules, où chaque particule représente une solution candidate.

---

## Définition du Problème

Le problème consiste à assigner des fréquences à 4 cellules (énumérées dans la matrice D1) tout en minimisant le nombre de violations des contraintes de séparation. La matrice C1 fournit les séparations minimales exigées entre les fréquences des cellules :

**Matrice des contraintes (C1)** :
```python
C1 = np.array([
    [5, 4, 0, 0],
    [4, 5, 0, 1],
    [0, 0, 5, 2],
    [0, 1, 2, 5]
])
```

**Demande des cellules (D1)** :
```python
D1 = np.array([1, 1, 1, 3])
```

Les contraintes sont telles que deux cellules \(i\) et \(j\) doivent avoir des fréquences différant d’au moins C1[i][j], sinon une violation est enregistrée.

---

## Algorithme : Optimisation par Essaim de Particules (PSO)

### 1. Initialisation
Chaque particule est une solution aléatoire de taille égale à la somme des demandes dans D1. Les fréquences attribuées sont des entiers entre 1 et 11.

**Implémentation :**
```python
num_particles = 30  # Nombre de particules
particles = np.random.randint(1, 12, size=(num_particles, sum(D1)))  # Solutions aléatoires
velocities = np.zeros((num_particles, sum(D1)))  # Vitesse initiale des particules
```
Les meilleures positions personnelles (**pbest**) et la meilleure position globale (**gbest**) sont initialisées :
```python
pbest_positions = particles.copy()
pbest_scores = np.array([fitness(p) for p in particles])
gbest_position = pbest_positions[np.argmin(pbest_scores)]
gbest_score = np.min(pbest_scores)
```

### 2. Fonction d’évaluation
La fonction de fitness mesure le nombre de violations des contraintes. Voici le code de cette fonction :

**Implémentation :**
```python
def fitness(position):
    violations = 0
    n = len(D1)
    m = 11  # Nombre maximal de fréquences disponibles

    # Diviser la position en sous-groupes basés sur les demandes D1
    cell_frequencies = []
    idx = 0
    for demand in D1:
        cell_frequencies.append(position[idx:idx + demand])
        idx += demand

    # Calcul des violations
    for i in range(n):
        for a in cell_frequencies[i]:
            for j in range(n):
                if i != j:
                    for b in cell_frequencies[j]:
                        if abs(a - b) < C1[i, j]:
                            violations += 1
    return violations
```

### 3. Mise à jour des particules
Les particules sont mises à jour à chaque itération en modifiant leurs vitesses et leurs positions. Les vitesses sont calculées à l’aide de l’équation suivante :

**Implémentation :**
```python
for t in range(max_iter):
    for i in range(num_particles):
        r1, r2 = np.random.rand(), np.random.rand()
        velocities[i] = (w * velocities[i] +
                         c1 * r1 * (pbest_positions[i] - particles[i]) +
                         c2 * r2 * (gbest_position - particles[i]))

        # Mise à jour des positions
        particles[i] = particles[i] + velocities[i]
        particles[i] = np.clip(particles[i], 1, 11)

        # Calculer le nouveau fitness
        current_fitness = fitness(particles[i])

        # Mettre à jour pbest et gbest
        if current_fitness < pbest_scores[i]:
            pbest_scores[i] = current_fitness
            pbest_positions[i] = particles[i]

        if current_fitness < gbest_score:
            gbest_score = current_fitness
            gbest_position = particles[i]
```

### 4. Critère d’arrêt
L’algorithme s’arrête après un nombre fixe d’itérations (égales à 100 dans notre cas).

---

## Résultats

L’algorithme renvoie la meilleure solution trouvée (position de \(gbest\)) et son score (nombre minimal de violations). Voici un exemple de résultat :

**Solution optimale :**
```python
[6, 10, 5, 3, 8, 11]
```

**Score minimal :**
```python
0  # Aucune violation
```

**Assignation des fréquences :**
```python
idx = 0
for cell_idx, demand in enumerate(D1):
    frequencies = gbest_position[idx:idx + demand]
    print(f"Cellule {cell_idx + 1} : {frequencies}")
    idx += demand
```

---

## Conclusion

L’optimisation par essaim de particules (PSO) est une approche efficace pour résoudre des problèmes complexes comme l’assignation de fréquences dans les réseaux cellulaires. L’algorithme exploite le comportement collaboratif des particules pour minimiser les violations des contraintes en convergeant vers une solution optimale. Les paramètres de l’algorithme (poids d’inertie, coefficients d’accélération, etc.) influencent directement la qualité et la vitesse de convergence.

Cette approche peut être généralisée à d’autres problèmes d’optimisation combinatoire, rendant PSO une méthode puissante dans divers domaines.

