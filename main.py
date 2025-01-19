import numpy as np

C1 = np.array([
    [5, 4, 0, 0],
    [4, 5, 0, 1],
    [0, 0, 5, 2],
    [0, 1, 2, 5]
])

D1 = np.array([1, 1, 1, 3])  

num_particles = 30  
max_iter = 100     
w = 0.5             
c1 = 1.5            
c2 = 1.5            


def fitness(position):

    violations = 0
    n = len(D1)  
    m = 11      

    cell_frequencies = []
    idx = 0
    for demand in D1:
        cell_frequencies.append(position[idx:idx + demand])  
        idx += demand

    for i in range(n):  
        for a in cell_frequencies[i]: 
            for j in range(n):  
                if i != j:  
                    for b in cell_frequencies[j]:  
                        if abs(a - b) < C1[i, j]:  
                            violations += 1
    return violations  

particles = np.random.randint(1, 12, size=(num_particles, sum(D1)))  
velocities = np.zeros((num_particles, sum(D1)))                      
pbest_positions = particles.copy()                                   
pbest_scores = np.array([fitness(p) for p in particles])             
gbest_position = pbest_positions[np.argmin(pbest_scores)]            
gbest_score = np.min(pbest_scores)                                   


for t in range(max_iter):
    for i in range(num_particles):

        r1, r2 = np.random.rand(), np.random.rand()
        velocities[i] = (w * velocities[i] +
                         c1 * r1 * (pbest_positions[i] - particles[i]) +
                         c2 * r2 * (gbest_position - particles[i]))
        


        particles[i] = particles[i] + velocities[i]
        particles[i] = np.clip(particles[i], 1, 11)  

        current_fitness = fitness(particles[i])


        if current_fitness < pbest_scores[i]:
            pbest_scores[i] = current_fitness
            pbest_positions[i] = particles[i]
        

        if current_fitness < gbest_score:
            gbest_score = current_fitness
            gbest_position = particles[i]

    

print("Best Solution (Frequency Assignment):", gbest_position)
print("Best Fitness (Minimum Violations):", gbest_score)


idx = 0
for cell_idx, demand in enumerate(D1):
    frequencies = gbest_position[idx:idx + demand]
    print(f"Cell {cell_idx + 1} Frequencies: {frequencies}")
    idx += demand