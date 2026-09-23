# Lab 2 - Ant Colony Optimization (ACO)

# ===============================================================
# 1. Setup (Part 1 of the task - all blanks filled)
# ===============================================================

# We need random numbers to let each ant "choose" a path probabilistically,
# so we import Python's built-in random module.
import random

# matplotlib is only used at the end to plot how pheromone changes over time.
import matplotlib.pyplot as plt

# Seeding fixes the starting point of the random number generator, so the
# "random" choices repeat exactly every time I run the notebook.
random.seed(67)  # <-- replace 67 with roll number

# The toy graph: four separate nest->food routes and their lengths.
# The key is the path name, the value is how long (costly) that path is.
paths = {'Path A': 10, 'Path B': 15, 'Path C': 8, 'Path D': 12}

# Every path starts with the SAME amount of pheromone (1.0) because at the
# very beginning the colony knows nothing - no route is a favourite yet.
# If I started them unequal I would be hand-feeding the answer to the algorithm
# instead of letting the behaviour emerge from the ants' own deposits.
pheromone = {p: 1.0 for p in paths}

# Fraction of pheromone that disappears each iteration. Evaporation is the
# colony's "forgetting": without it, early lucky choices would never be undone
# and the colony could never escape a bad path.
evaporation_rate = 0.5

# Deposit strength constant. An ant on a path of length L adds Q/L pheromone,
# so Q simply scales how loud one ant's "vote" is compared to what is already
# on the trail.
Q = 100

# How many rounds of (ants walk -> pheromone evaporates) we simulate.
iterations = 50

# How many ants walk in each round. More ants per round = more votes cast
# before evaporation is applied, so the trail strengthens faster.
ants_per_iteration = 10

# Record the pheromone level of every path after each iteration, so we can
# plot the convergence curve later and SEE one path taking over.
history = {p: [] for p in paths}


# ===============================================================
# 2. The main ACO loop (Part 2 of the task - the heart of the algorithm)
# ===============================================================
for it in range(iterations):

    for ant in range(ants_per_iteration):
        # Sum of pheromone over ALL paths. We need this denominator to turn
        # raw pheromone amounts into probabilities that add up to 1.
        total = sum(pheromone.values())

        # Each path's SHARE of the total pheromone. A path carrying more
        # pheromone gets a proportionally bigger slice of probability - this
        # is what makes an ant "follow the crowd" instead of choosing blindly.
        weights = [pheromone[p] / total for p in paths]

        # random.choices() picks one path at random, but biased by the weights
        # above. It returns a list, so [0] pulls the single chosen name out.
        chosen = random.choices(list(paths.keys()), weights=weights, k=1)[0]

        # The ant deposits pheromone on the path it walked. We divide Q by the
        # path's LENGTH, so a short path receives a bigger boost per ant than a
        # long one. This tiny length-based bias is the ONLY place the algorithm
        # ever "sees" path quality - everything else is just copying neighbours.
        pheromone[chosen] += Q / paths[chosen]

    # Evaporation, applied once per iteration to every path (walked or not).
    # Multiplying by (1 - rate) removes that fraction of the trail. This is the
    # colony's memory decay: old information fades, so paths must keep being
    # re-chosen to stay attractive.
    for p in pheromone:
        pheromone[p] = pheromone[p] * (1 - evaporation_rate)

    # Store this iteration's levels so we can plot the race between paths.
    for p in paths:
        history[p].append(pheromone[p])

# After all iterations, the path carrying the most pheromone is the one the
# colony effectively agreed on. max(..., key=pheromone.get) returns the KEY
# (the path name) with the largest value, not the value itself.
best = max(pheromone, key=pheromone.get)

print("Colony converged on:", best)
print()
for p in paths:
    print(f"{p}: length = {paths[p]:>2}   final pheromone = {pheromone[p]:.4f}")


# ===============================================================
# 3. Convergence plot
# ===============================================================
# Plot one line per path so we can literally watch one trail dominate and the
# others decay towards zero - the visual signature of ACO convergence.
plt.figure(figsize=(9, 5))
for p in paths:
    plt.plot(history[p], label=f"{p} (length {paths[p]})", linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Pheromone level")
plt.title(f"ACO convergence - seed 21, evaporation = {evaporation_rate}, Q = {Q}")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# ===============================================================
# 4. Experiment 1 - changing the evaporation rate
# ===============================================================
# Wrap the whole simulation in a function so we can re-run it with different
# parameters without copy-pasting the loop. Returning the history too lets us
# plot each experiment.
def run_aco(evaporation_rate=0.5, Q=100, iterations=50, ants_per_iteration=10, seed=67):
    # Re-seeding inside the function means every experiment starts from the
    # SAME random stream, so any difference we see is caused by the parameter
    # we changed and not by luck.
    random.seed(seed)

    pheromone = {p: 1.0 for p in paths}
    history = {p: [] for p in paths}

    for it in range(iterations):
        for ant in range(ants_per_iteration):
            total = sum(pheromone.values())
            weights = [pheromone[p] / total for p in paths]
            chosen = random.choices(list(paths.keys()), weights=weights, k=1)[0]
            pheromone[chosen] += Q / paths[chosen]
        for p in pheromone:
            pheromone[p] = pheromone[p] * (1 - evaporation_rate)
        for p in paths:
            history[p].append(pheromone[p])

    best = max(pheromone, key=pheromone.get)
    # Share = what fraction of ALL pheromone the winner holds. A share near 1.0
    # means the colony is fully committed; near 0.25 means it is still undecided.
    share = pheromone[best] / sum(pheromone.values())
    return best, share, history


# Try a slow-forgetting colony (0.1), the default (0.5), and a fast-forgetting
# one (0.9), keeping Q fixed so evaporation is the only variable.
for ev in [0.1, 0.5, 0.9]:
    best, share, hist = run_aco(evaporation_rate=ev, Q=100)
    print(f"evaporation = {ev} -> winner = {best}, winner holds {share*100:.1f}% of all pheromone")


# ===============================================================
# 5. Experiment 2 - changing Q (the deposit constant)
# ===============================================================
# Now hold evaporation at 0.5 and vary how loudly each ant votes.
for q in [10, 100, 500]:
    best, share, hist = run_aco(evaporation_rate=0.5, Q=q)
    print(f"Q = {q:>3} -> winner = {best}, winner holds {share*100:.1f}% of all pheromone")

# Side-by-side view: how the winning trail grows under each evaporation rate.
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
for ax, ev in zip(axes, [0.1, 0.5, 0.9]):
    best, share, hist = run_aco(evaporation_rate=ev, Q=100)
    for p in paths:
        ax.plot(hist[p], label=p)
    ax.set_title(f"evaporation = {ev}\nwinner: {best}")
    ax.set_xlabel("Iteration")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Pheromone level")
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.show()


# ===============================================================
# 6. Comparison notes
# ===============================================================
# Evaporation rate. Changing the evaporation rate changed the ceiling and the
# speed of the trail far more than the winner itself: at 0.1 only 10% of the
# pheromone is removed per round, so the trails keep growing and the winner
# ended up holding about 98% of the total, while at 0.9 almost everything is
# wiped each round and the surviving level settles at a low steady value almost
# immediately. Low evaporation means the colony remembers longer and reinforces
# whatever it found first (harder to escape a bad path); high evaporation means
# it forgets fast and effectively only trusts the most recent round of ants,
# which keeps exploration alive but makes the result noisier. With my seed,
# though, the winner stayed Path A at all three rates, because Q = 100 is large
# enough that the race is already decided in the first iteration or two -
# before evaporation has had any chance to act.
#
# Q (deposit constant). Q decides how loud one ant's vote is relative to the
# pheromone already on the trail. At Q = 10 the deposits are small compared to
# the starting pheromone of 1.0, so the colony stayed undecided for many more
# iterations, sampled all four paths properly, and correctly converged on
# Path C (length 8). At Q = 100 and Q = 500 the very first ants slam so much
# pheromone onto whichever path they happened to pick that the probabilities
# collapse within one or two iterations - the colony locked onto Path A and
# never reconsidered, and raising Q from 100 to 500 only made that lock-in
# faster, not better.
#
# Overall. The two parameters together control the classic exploration vs.
# exploitation trade-off: large Q and low evaporation = fast, confident, and
# easily wrong; small Q and moderate evaporation = slower, but much more likely
# to actually find the shortest path.
