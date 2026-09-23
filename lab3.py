# Lab 3 - Particle Swarm Optimization (PSO)

# ===============================================================
# 1. Setup (Part 1 of the task)
# ===============================================================

# PSO needs random numbers in two places: scattering the initial swarm, and
# the r1/r2 factors that make each particle's pull slightly unpredictable.
import random

# Only used at the end, to plot how the global best improves over time.
import matplotlib.pyplot as plt

# Seeding locks the random number generator to a fixed starting state, so the
# whole notebook produces the SAME output every time I run it. I use the numeric
# part of MY roll number (101) so my starting positions, my convergence
# curve and my final answer are personally mine - a classmate with a different
# seed gets a different swarm even with identical code.
random.seed(101)  # <-- my roll number


# The objective ("fitness") function the swarm is trying to minimise. It is a
# simple upward parabola shifted right by 7 and lifted up by 4, so its lowest
# point is at x = 7 with value 4. The swarm never uses that fact - it can only
# call f(x) and compare the numbers it gets back.
def f(x):
    return (x - 7) ** 2 + 4


# Scatter 5 particles uniformly at random across the whole search range [0, 14].
# Starting them spread out is important: if they all began in one corner the
# swarm would have no diversity and could converge on whatever is nearby
# instead of properly searching the space.
positions = [random.uniform(0, 14) for _ in range(5)]

# Every particle starts at rest. Velocity here is not physics - it is just the
# step the particle will take next iteration. Starting at 0 means the first
# move is driven purely by the pbest/gbest attraction terms, not by momentum.
velocities = [0 for _ in range(5)]

# Each particle's personal best position so far. At iteration 0 the only place
# a particle has ever visited is where it started, so pbest = start.
# positions[:] makes a COPY of the list. If I wrote pbest_pos = positions then
# both names would point at the SAME list object, so every time I updated a
# particle's position I would silently overwrite its memory too - the algorithm
# would lose its personal-best term completely and stop working.
pbest_pos = positions[:]

# The swarm's global best: the single best position any particle has found.
# min(..., key=f) evaluates f on every personal best and returns the POSITION
# with the smallest f value (not the f value itself), because we are minimising.
gbest_pos = min(pbest_pos, key=f)

# The three PSO coefficients:
#   w  (inertia)   - how much of the previous velocity is carried over.
#                    High w = the particle keeps drifting = more exploration.
#   c1 (cognitive) - how strongly a particle is pulled back towards its OWN
#                    best spot. This is individual memory / self-confidence.
#   c2 (social)    - how strongly it is pulled towards the SWARM's best spot.
#                    This is herd behaviour / trust in the group.
# The balance between them is the whole exploration-vs-exploitation trade-off.
w, c1, c2 = 0.6, 1.5, 1.5

# How many times the whole swarm updates. 40 is plenty for a 1-D parabola.
iterations = 40

print("My starting positions:", [round(p, 4) for p in positions])
print("Initial global best  :", round(gbest_pos, 4), "  f =", round(f(gbest_pos), 4))

# Record the global best after each iteration so we can plot convergence later.
gbest_history = []
# Record every particle's position each iteration to show the swarm collapsing.
position_history = []


# ===============================================================
# 2. The main PSO loop (Part 2 of the task)
# ===============================================================
for it in range(iterations):
    for i in range(5):
        # Two fresh random numbers in [0, 1) for THIS particle, THIS iteration.
        # They randomise how hard the personal and social pulls act each step,
        # so particles do not all fly along identical straight lines - this
        # randomness is what keeps a little exploration alive in the swarm.
        r1, r2 = random.random(), random.random()

        # THE VELOCITY UPDATE - the core equation of PSO. Three terms added:
        #   w * velocities[i]                       -> momentum: keep going the
        #                                              way I was already going
        #   c1 * r1 * (pbest_pos[i] - positions[i]) -> cognitive: steer back
        #                                              towards MY own best spot
        #   c2 * r2 * (gbest_pos - positions[i])    -> social: steer towards the
        #                                              BEST spot anyone found
        # Each bracket is a direction-and-distance: if my best is to the right
        # of me the term is positive and pushes me right, and the further away
        # it is the harder the push.
        velocities[i] = (w * velocities[i]
                         + c1 * r1 * (pbest_pos[i] - positions[i])
                         + c2 * r2 * (gbest_pos - positions[i]))

        # Actually move the particle. Position is updated by adding the velocity,
        # exactly like x_new = x_old + step. Without this line the particles
        # would compute where they want to go and then never go there.
        positions[i] = positions[i] + velocities[i]

        # Did this particle just land somewhere better than anywhere it has
        # been before? "Better" means a SMALLER f, because this is minimisation.
        if f(positions[i]) < f(pbest_pos[i]):
            # Yes - overwrite its personal memory with the new position. This is
            # the only memory an individual particle has.
            pbest_pos[i] = positions[i]

    # After all 5 particles have moved, recompute the swarm's global best from
    # the personal bests. gbest can only ever improve or stay the same, since
    # every pbest it is chosen from only ever improves.
    gbest_pos = min(pbest_pos, key=f)
    gbest_history.append(gbest_pos)
    position_history.append(positions[:])  # copy again, for the same reason as before

print("Swarm converged near x =", gbest_pos)
print("f(gbest) =", f(gbest_pos))
print("True minimum is x = 7, f = 4  -> error in x:", abs(gbest_pos - 7))


# ===============================================================
# 3. Convergence plots
# ===============================================================
# Left plot: how the global best position approaches the true minimum.
# Right plot: how the f-value of the global best drops towards 4.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

ax1.plot(gbest_history, marker="o", markersize=3, linewidth=2)
# A reference line at the true answer, so we can see how close the swarm got.
ax1.axhline(7, color="red", linestyle="--", label="true minimum x = 7")
ax1.set_xlabel("Iteration"); ax1.set_ylabel("Global best position (x)")
ax1.set_title("gbest position converging to x = 7"); ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot([f(g) for g in gbest_history], marker="o", markersize=3, linewidth=2, color="darkgreen")
ax2.axhline(4, color="red", linestyle="--", label="true minimum f = 4")
ax2.set_xlabel("Iteration"); ax2.set_ylabel("f(gbest)")
ax2.set_title("Best fitness found so far"); ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout(); plt.show()

# Show the swarm physically collapsing: each line is one particle's trajectory.
plt.figure(figsize=(9, 5))
for i in range(5):
    plt.plot([step[i] for step in position_history], linewidth=1.8, label=f"Particle {i+1}")
plt.axhline(7, color="red", linestyle="--", label="true minimum")
plt.xlabel("Iteration"); plt.ylabel("Position (x)")
plt.title("All 5 particles converging on the same point - seed 67")
plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.show()


# ===============================================================
# 4. Experiment 1 - changing the inertia weight w
# ===============================================================
# Wrap the whole simulation in a function so experiments do not need copy-paste.
# Re-seeding inside means every run starts from the SAME swarm, so any
# difference we observe is caused by the parameter we changed, not by luck.
def run_pso(w=0.6, c1=1.5, c2=1.5, iterations=40, seed=67, n=5):
    random.seed(seed)
    positions = [random.uniform(0, 14) for _ in range(n)]
    velocities = [0 for _ in range(n)]
    pbest_pos = positions[:]
    gbest_pos = min(pbest_pos, key=f)
    history = []

    for it in range(iterations):
        for i in range(n):
            r1, r2 = random.random(), random.random()
            velocities[i] = (w * velocities[i]
                             + c1 * r1 * (pbest_pos[i] - positions[i])
                             + c2 * r2 * (gbest_pos - positions[i]))
            positions[i] = positions[i] + velocities[i]
            if f(positions[i]) < f(pbest_pos[i]):
                pbest_pos[i] = positions[i]
        gbest_pos = min(pbest_pos, key=f)
        history.append(gbest_pos)

    # First iteration at which the swarm got within 0.01 of the true minimum -
    # a simple numeric measure of "how fast did it converge".
    speed = next((i + 1 for i, g in enumerate(history) if abs(g - 7) < 0.01), None)
    # How far apart the particles still are at the end. Near 0 = fully collapsed,
    # large = still flying around (or diverged).
    spread = max(positions) - min(positions)
    return gbest_pos, history, speed, spread


# Low inertia (0.2), the default (0.6) and high inertia (0.9), with c1/c2 fixed.
for ww in [0.2, 0.6, 0.9]:
    g, h, speed, spread = run_pso(w=ww)
    print(f"w = {ww}:  gbest = {g:.6f}   f = {f(g):.8f}   "
          f"iterations to reach |x-7| < 0.01 = {speed}   final swarm spread = {spread:.4f}")


# ===============================================================
# 5. Experiment 2 - changing c1 and c2 (the pull strengths)
# ===============================================================
# Weak pulls (0.5), default (1.5) and very strong pulls (3.0), with w fixed.
for cc in [0.5, 1.5, 3.0]:
    g, h, speed, spread = run_pso(c1=cc, c2=cc)
    print(f"c1 = c2 = {cc}:  gbest = {g:.6f}   f = {f(g):.6f}   "
          f"iterations to reach |x-7| < 0.01 = {speed}   final swarm spread = {spread:.4f}")

# Visual comparison of all six settings on one figure.
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for ax, ww in zip(axes[0], [0.2, 0.6, 0.9]):
    g, h, speed, spread = run_pso(w=ww)
    ax.plot(h, linewidth=2)
    ax.axhline(7, color="red", linestyle="--")
    ax.set_title(f"w = {ww}   (converged in {speed} iters)")
    ax.set_xlabel("Iteration"); ax.grid(alpha=0.3)
axes[0][0].set_ylabel("gbest position")

for ax, cc in zip(axes[1], [0.5, 1.5, 3.0]):
    g, h, speed, spread = run_pso(c1=cc, c2=cc)
    ax.plot(h, linewidth=2, color="darkorange")
    ax.axhline(7, color="red", linestyle="--")
    ax.set_title(f"c1 = c2 = {cc}   (converged in {speed} iters)")
    ax.set_xlabel("Iteration"); ax.grid(alpha=0.3)
axes[1][0].set_ylabel("gbest position")

plt.tight_layout(); plt.show()


# ===============================================================
# 6. Comparison notes (the 2-3 sentences asked for in the slides)
# ===============================================================
# Inertia weight w. Inertia decides how much of a particle's old motion
# survives into the next step, so it directly controls exploration versus
# settling down. At w = 0.2 the particles almost forget their previous velocity
# and just fall straight onto the current best - the swarm was fully collapsed
# (final spread 0.0000) and hit the minimum in 9 iterations, accurate but with
# no exploration left at all. At w = 0.6 it was fastest on my seed
# (7 iterations) with a tiny bit of residual movement, and at w = 0.9 the
# particles kept overshooting and swinging past the target: it took
# 30 iterations to get within 0.01 and the swarm was still spread over a range
# of about 38 units at the end. So higher inertia = more exploration and more
# oscillation but slower, noisier convergence; lower inertia = quick, decisive,
# but the swarm loses its ability to search once it has committed.
#
# Pull strengths c1 and c2. These set how hard a particle is yanked towards its
# own best (c1) and the swarm's best (c2). At c1 = c2 = 0.5 the pulls were
# gentle, so the swarm drifted in more cautiously and took 13 iterations
# instead of 7 - slower, but smooth and stable. At c1 = c2 = 3 the swarm blew
# up: the correction term is now so large that each particle shoots far past
# the target, then gets an even larger correction back the other way, and the
# velocities amplify themselves every step. It never reached within 0.01 of
# the minimum at all, ended on a worse answer (x ~ 6.734, f ~ 4.071), and the
# final particle spread was about 99,946 - the particles had flown off to
# absurd coordinates. This is exactly why standard PSO uses c1 = c2 ~ 1.5 with
# w ~ 0.6-0.7: those values sit inside the stability region where the velocity
# update converges instead of diverging.
#
# Overall. w, c1 and c2 together control the same exploration-vs-exploitation
# trade-off we saw with evaporation and Q in the ACO lab. Too small and the
# swarm converges quickly but blindly; too large and it never settles at all.
# Only the middle band gives both a proper search and a converged answer.
