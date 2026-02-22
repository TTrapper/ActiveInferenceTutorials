# Lesson Progression

Each lesson builds on the previous one. The student should finish each lesson
with a working simulation and a clear intuition for why the next lesson is
needed.

The destination: agents with **no hardcoded behavior** that discover hunting,
fleeing, and foraging purely from preferences and free energy minimization.
Early lessons use hardcoded behavior as scaffolding to teach the underlying
concepts; later lessons remove that scaffolding entirely.

---

## L1: State Transitions (implemented)

**Concept:** A generative model is a set of rules that define how states
evolve over time.

**Setup:** A single prey agent moves on a 32×32 grid. It follows a fixed
stochastic policy: it is biased to move away from the center of the grid.

**What the student sees:** The prey's behavior changes depending on where it
is (e.g., it hugs the edges). The heatmap shows the true transition
probabilities from the current cell, which always point away from the center.

**Key takeaway:** State-dependent transition models are the building block of
everything that follows. The prey's policy *is* the generative model that the
predator will need to learn.

---

## L2: Learning Transitions (implemented)

**Concept:** An agent can learn another agent's generative model by observing
state transitions and counting.

**Setup:** A predator is introduced. The prey's behavior changes: it now moves
stochastically away from the *predator's* current position. The predator
uses a `BayesianWorldModel` where the state key is the prey's position only
(1,024 states).

**What the student sees:** The predator's belief heatmap converges over time,
but it is "blurry" because the predator is ignoring its own position—the
very thing the prey is reacting to. The predator learns an average of how
the prey moves at each location.

**Key takeaway:** Bayesian updating works, but if your model ignores key
variables (like the predator's own position), the learned model will be an
inaccurate "average" of reality.

---

## L3: State Space Explosion (implemented)

**Concept:** Adding dimensions to the state key allows for more accurate
models but causes combinatorial explosion in tabular models.

**Setup:** The predator's position is added to the state key for its model.
State space goes from 32² = 1,024 to 32⁴ = 1,048,576 possible states.

**What the student sees:** The predator's belief heatmap is much more precise
(it correctly predicts the prey will move away from its current spot), but
learning is visibly slower. Most state combinations are never visited, so
the model stays "cold" for much longer.

**Key takeaway:** Tabular models are precise but don't scale. We need a way
to generalize across similar states instead of memorizing each one
individually.

---

## L4: Function Approximation (implemented)

**Concept:** Neural networks can generalize across states, solving the
tabular explosion problem.

**Setup:** Replace the `BayesianWorldModel` lookup table with a small neural
network. Input: current state (prey position, predator position). Output:
predicted transition probabilities over 8 directions. Train on observed
transitions using cross-entropy loss. The predator still has hardcoded
"move toward believed prey location" behavior — only the *world model* is
a neural net, not the *action selection*.

**What the student sees:** The predator learns the prey's behavior in the
joint state space (L3's setup) much faster than the tabular model. States
that have never been visited still get reasonable predictions because nearby
states share structure.

**Key takeaway:** Function approximation generalizes across states. But the
predator still has hardcoded "chase" behavior and passively learns a world
model.

---

## L5: Environmental Complexity (Obstacles)

**Concept:** Hardcoded behavior fails when the environment isn't a simple
open grid. Obstacles provide a reason to move beyond simple "chase" logic.

**Setup:** Add walls/obstacles to the grid. The predator's hardcoded "move
directly toward prey" behavior now gets it stuck behind walls.

**What the student sees:** The predator is visibly incompetent. It "hugs"
the wall trying to reach the prey on the other side, unable to navigate
around the obstacle. The world model (from L4) starts learning that
transitions into wall cells have zero probability.

**Key takeaway:** Simple rules like "move toward target" aren't enough for
real worlds. We need a way to *plan* based on the learned world model.

---

## L6: Preferences Replace Hardcoded Behavior

**Concept:** Instead of programming *what to do*, we specify *what the agent
expects to observe* (preferences) and let behavior emerge from minimizing
free energy.

**Core idea:** An active inference agent selects actions that minimize
**expected free energy**:

    G(a) = ambiguity + risk

**Setup:** Remove the predator's hardcoded "move toward believed prey
location" action policy. Replace it with expected free energy minimization.
The predator has a preference prior: high probability of observing prey at
its own location.

**What the student sees:** The predator *discovers* hunting and navigation.
It now navigates *around* the walls introduced in L5 because the free energy
landscape accounts for the transition model (which says walls are impassable)
and the preference (which drives it toward the prey). Chasing and
pathfinding *emerge* from the same objective.

**Key takeaway:** Behavior doesn't need to be programmed. Given a world model
and preferences, free energy minimization produces intelligent navigation
and hunting.

---

## L7: Both Agents Use Active Inference

**Concept:** When both agents are active inference agents, complex behavior
emerges from simple preferences without any hardcoding.

**Setup:** The prey also becomes an active inference agent:
- **Prey's preferences:** "I expect to observe myself alive" and "I expect
  to observe food at my location."
- On death, the simulation resets but learned models carry over.

**What the student sees:**
- The prey *discovers* evasion and foraging.
- An arms race emerges: the prey adapts its evasion as the predator gets
  better at hunting.
- The predator handles the non-stationarity naturally.

**Key takeaway:** Complex multi-agent behavior — hunting, fleeing, foraging,
balancing competing goals — emerges from simple preferences and a single
optimization principle.

---

## L8: Scaling to Complex Dynamics

**Concept:** The framework scales to realistic complexity because adding
structure just changes the generative model, not the inference algorithm.

**Setup:** Add rich terrain (mud that slows movement), multiple prey/food
sources, and changing environments.

**What the student sees:**
- The agents learn to use the environment to their advantage (e.g. using cover).
- Emergent strategies like ambush or path planning to minimize risk.
- The same minimize-free-energy recipe produces increasingly sophisticated
  behavior as the environment gets richer.

**Key takeaway:** No hardcoded pathfinding, evasion tactics, or ambush
strategies are required. You just expand the generative model.

---

## Summary of Progression

| Lesson | Introduces | Removes | Key question answered |
|--------|-----------|---------|----------------------|
| L1 | State transitions | — | What is a generative model? |
| L2 | Bayesian learning | — | How do you learn someone else's model? |
| L3 | Joint state | — | What happens when state space grows? |
| L4 | Neural networks | Tabular models | How do you generalize across states? |
| L5 | Obstacles | — | Why is hardcoded behavior insufficient? |
| L6 | Free energy + preferences | Predator's hardcoded behavior | Can behavior emerge from preferences? |
| L7 | Both agents active | All hardcoded behavior | Can multi-agent behavior emerge? |
| L8 | Complex environments | Simplistic grid assumptions | Does this scale to real complexity? |

### The arc in one sentence

We start with hardcoded behavior and a lookup table, introduce obstacles to
break simple rules, and then replace all programmed behavior with a single
principle—minimizing free energy—where hunting, fleeing, and navigation
emerge naturally.
