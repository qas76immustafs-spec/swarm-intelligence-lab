# AI Labs 1–3: Search Spaces, Ant Colony Optimization, Particle Swarm Optimization

**Author:** Qasim Mustafa

This repository contains the code for three lab tasks. Each lab is one Python script that was converted from a Google Colab notebook. The written answers and comparison notes from each notebook are kept as comments at the bottom of each script.

| File | Topic | Algorithm |
|------|-------|-----------|
| `lab1.py` | Search spaces and benchmark functions | Random Search |
| `lab2.py` | Choosing the shortest of four nest→food paths | Ant Colony Optimization (ACO) |
| `lab3.py` | Minimising a 1-D parabola | Particle Swarm Optimization (PSO) |

---

## Requirements

- Python 3.8 or newer
- `numpy` (needed by Lab 1 only)
- `matplotlib`

Install them with:

```bash
pip install numpy matplotlib
```

## How to run

```bash
python lab1.py
python lab2.py
python lab3.py
```

Each script prints its results to the terminal and opens its plots in windows one after another. Close a plot window to continue to the next one.

To run a script without showing any windows (for example on a server), use:

```bash
MPLBACKEND=Agg python lab2.py
```

---

## Lab 1: Search Spaces and Random Search (`lab1.py`)

### Concept

A **search space** is the set of every possible answer an algorithm is allowed to try. When the goal is to find the lowest point in a landscape, the search space is every (x, y) location in that landscape. A **benchmark function** is a standard, well-known landscape that researchers use to test and compare optimization algorithms fairly.

### Benchmark functions

All three functions have their global minimum at f(0, 0) = 0.

| Function | Formula | Range | Landscape |
|----------|---------|-------|-----------|
| Sphere | x² + y² | [-5, 5] | A single smooth bowl with one minimum |
| Rastrigin | 20 + (x² − 10cos 2πx) + (y² − 10cos 2πy) | [-5.12, 5.12] | A regular grid of many local minima |
| Ackley | −20e^(−0.2√(0.5(x²+y²))) − e^(0.5(cos 2πx + cos 2πy)) + e + 20 | [-5, 5] | Mostly flat outer region with a sharp central hole and small ripples |

The script draws a contour plot of each landscape.

### Random Search

`random_search(func, bounds, iterations=1000)` samples 1,000 random (x, y) points inside the bounds and keeps the point with the lowest score. It has no direction and no memory beyond the best point found so far.

### Results from the notebook run

| Function | Best point | Best score |
|----------|------------|------------|
| Sphere | (0.2078, −0.0910) | 0.0515 |
| Rastrigin | (−0.0337, 0.8885) | 3.3709 |
| Ackley | (−0.0016, −0.1932) | 1.2941 |

Random Search is not seeded, so these numbers change on every run.

### Key findings

- **Best result: Sphere.** It has one smooth minimum, so random points easily land close to it.
- **Hardest: Rastrigin.** Its many local valleys make the exact global minimum hard to hit by chance.
- **More iterations** (10,000 instead of 1,000) should improve every result. The gain matters least for Sphere, which is already close to 0.
- **Main weakness:** Random Search has no strategy. It wastes many attempts and can miss the true minimum in complicated landscapes.

---

## Lab 2: Ant Colony Optimization (`lab2.py`)

### Problem

An ant colony must choose among four separate routes from the nest to food:

| Path | Length |
|------|--------|
| Path A | 10 |
| Path B | 15 |
| **Path C** | **8 (shortest)** |
| Path D | 12 |

### How the algorithm works

1. Every path starts with the same pheromone level (1.0), so no path is favoured at the start.
2. In each iteration, 10 ants each choose a path with probability proportional to that path's share of the total pheromone.
3. An ant deposits **Q / length** pheromone on the path it chose. Shorter paths get a bigger deposit per ant. This is the only place the algorithm "sees" path quality.
4. At the end of each iteration, every path loses a fraction of its pheromone (evaporation): `pheromone × (1 − evaporation_rate)`.
5. After 50 iterations, the path with the most pheromone is the one the colony converged on.

### Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `random.seed` | 67 | Makes the random choices repeatable (roll number) |
| `evaporation_rate` | 0.5 | Fraction of pheromone removed each iteration (the colony "forgetting") |
| `Q` | 100 | Deposit strength, i.e. how loud one ant's "vote" is |
| `iterations` | 50 | Number of walk-then-evaporate rounds |
| `ants_per_iteration` | 10 | Number of ants that walk in each round |

### Results

**Main run (evaporation 0.5, Q = 100):** the colony converged on **Path A**, which holds all of the pheromone.

**Experiment 1: evaporation rate (Q = 100)**

| Evaporation | Winner | Winner's share |
|-------------|--------|----------------|
| 0.1 | Path A | 95.1% |
| 0.5 | Path A | 100.0% |
| 0.9 | Path A | 100.0% |

**Experiment 2: deposit constant Q (evaporation 0.5)**

| Q | Winner | Winner's share |
|---|--------|----------------|
| 10 | **Path C** (the correct shortest path) | 100.0% |
| 100 | Path A | 100.0% |
| 500 | Path A | 100.0% |

### Key findings

- **Evaporation** mainly changes how high the pheromone levels climb and how fast they settle. It did not change the winner with this seed.
  - Low evaporation means the colony remembers longer and reinforces whatever it found first.
  - High evaporation means it forgets quickly and trusts only the most recent ants.
- **Q** had the biggest effect on the winner.
  - With Q = 10, deposits are small compared to the starting pheromone. The colony stays undecided for longer, samples all four paths, and finds the true shortest path (C).
  - With Q = 100 or 500, the first few ants lock the colony onto Path A within one or two iterations.
- **Overall:** this is the exploration-vs-exploitation trade-off.
  - Large Q with low evaporation is fast, confident, and easily wrong.
  - Small Q with moderate evaporation is slower but more likely to find the best path.

---

## Lab 3: Particle Swarm Optimization (`lab3.py`)

### Problem

Minimise the function

```
f(x) = (x − 7)² + 4
```

on the range [0, 14]. The true minimum is at **x = 7, f = 4**. The swarm never uses this fact. It can only evaluate f(x) and compare the values it gets back.

### How the algorithm works

1. Five particles start at random positions in [0, 14], each with velocity 0.
2. Each particle remembers its own best position (`pbest`). The swarm tracks the best position any particle has found (`gbest`).
3. In each iteration, every particle updates its velocity and position:

```
v = w·v + c1·r1·(pbest − x) + c2·r2·(gbest − x)
x = x + v
```

   - `w·v` is momentum: keep moving in the same direction.
   - `c1·r1·(pbest − x)` is the cognitive pull back towards the particle's own best spot.
   - `c2·r2·(gbest − x)` is the social pull towards the swarm's best spot.
   - `r1` and `r2` are fresh random numbers in [0, 1). They keep some exploration in the swarm.
4. After every particle has moved, `pbest` and `gbest` are updated if better positions were found.

### Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `random.seed` | 101 (main run), 67 (experiments) | Makes the swarm repeatable |
| `w` | 0.6 | Inertia: how much of the previous velocity carries over |
| `c1` | 1.5 | Cognitive pull towards the particle's own best |
| `c2` | 1.5 | Social pull towards the swarm's best |
| `iterations` | 40 | Number of swarm updates |
| particles | 5 | Swarm size |

### Results

**Main run (seed 101):**

- Starting positions: [8.1361, 2.7266, 13.5135, 12.9357, 6.5399]
- Initial gbest: 6.5399 (f = 4.2117)
- Final gbest: **x = 6.99988**, f = 4.0000000145
- Error in x: 0.00012

**Experiment 1: inertia weight w (c1 = c2 = 1.5)**

| w | Final gbest | Iterations to reach \|x−7\| < 0.01 | Final swarm spread |
|---|-------------|-----------------------------------|--------------------|
| 0.2 | 7.000000 | 9 | 0.0000 |
| 0.6 | 6.999985 | **7** | 0.0053 |
| 0.9 | 7.002622 | 30 | 38.7043 |

**Experiment 2: pull strengths c1 = c2 (w = 0.6)**

| c1 = c2 | Final gbest | Iterations to reach \|x−7\| < 0.01 | Final swarm spread |
|---------|-------------|-----------------------------------|--------------------|
| 0.5 | 7.000020 | 13 | 0.0011 |
| 1.5 | 6.999985 | **7** | 0.0053 |
| 3.0 | 6.733620 | never | 99,945.93 (diverged) |

### Key findings

- **Inertia w** controls how much a particle explores versus how quickly it settles.
  - w = 0.2: the swarm collapses immediately. It is accurate, but no exploration is left.
  - w = 0.6: fastest convergence (7 iterations).
  - w = 0.9: particles keep overshooting the target and take 30 iterations to converge.
- **Pull strengths c1 and c2** control how hard particles are pulled towards the best positions.
  - 0.5: gentle and stable, but slower (13 iterations).
  - 3.0: the swarm blows up. Each correction overshoots further than the last, velocities grow every step, and the particles fly off to extreme coordinates.
- **Standard values** (w ≈ 0.6–0.7, c1 = c2 ≈ 1.5) sit inside the stability region, where the velocity update converges instead of diverging.
- **Overall:** this is the same exploration-vs-exploitation trade-off as evaporation and Q in the ACO lab.
  - Too small, and the swarm converges quickly but blindly.
  - Too large, and it never settles.

---

## Known issues and notes

- **Lab 1:** the written answers quote scores of 0.0403 (Sphere) and 0.4606 (Rastrigin). These don't match the printed outputs (0.0515 and 3.3709), because Random Search is unseeded and every run gives different numbers. Add `random.seed(...)` before the searches to make the results repeatable.
- **Lab 2:** the `run_aco` function signature was cut off in the source PDF. It was completed as `seed=79`.
  - With this seed, the evaporation 0.1 run gives a 95.1% share. The notebook showed 97.7%, so it probably used a different seed there.
  - The main plot title says "seed 21", but the code uses seed 79.
- **Lab 3:** the main run uses seed 79, but the particle-trajectory plot title says "seed 79". The `run_pso` experiments default to seed 79.
  - All printed Lab 3 results match the original notebook exactly.
