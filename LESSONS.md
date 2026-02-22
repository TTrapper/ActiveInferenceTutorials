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

**Setup:** A single prey agent moves on a 32×32 grid. Each grid cell has its
own randomly generated movement policy (a probability distribution over 8
directions). No predator.

**What the student sees:** The prey's behavior changes depending on where it
is. The heatmap shows the true transition probabilities from the current cell.

**Key takeaway:** State-dependent transition models are the building block of
everything that follows. The prey's policy *is* the generative model that the
predator will need to learn.

---

## L2: Learning Transitions (implemented)

**Concept:** An agent can learn another agent's generative model by observing
state transitions and counting.

**Setup:** A predator is introduced. It uses a `BayesianWorldModel` — a
Dirichlet-categorical model that maintains a table of direction counts per
state. The state key is the prey's position only (1,024 possible states).
The predator still has hardcoded "move toward believed prey location"
behavior.

**What the student sees:** The predator's belief heatmap converges toward the
prey's true policy over time. The model error heatmap shrinks as the predator
gathers more observations.

**Key takeaway:** Bayesian updating works — given enough observations, the
predator's model matches reality. But the state key only encodes the prey's
position, so the model assumes the prey behaves the same regardless of where
the predator is.

---

## L3: State Space Explosion (implemented)

**Concept:** Adding dimensions to the state key causes combinatorial explosion
in tabular models.

**Setup:** The predator's position is added to the state key for both the
prey's behavior and the predator's model. State space goes from 32² = 1,024
to 32⁴ = 1,048,576 possible states.

**What the student sees:** Learning is visibly slower. The model error stays
high for much longer because most state combinations are never visited. The
predator struggles to catch the prey.

**Key takeaway:** Tabular models don't scale. We need a way to generalize
across similar states instead of memorizing each one individually.

---

## L4: Function Approximation

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

**Implementation notes:**
- Use a simple MLP (2–3 hidden layers) running in the browser via a
  lightweight library (e.g. TensorFlow.js or a minimal from-scratch
  implementation for pedagogical clarity).
- Training happens online: each observed transition is a training example.
- Compare learning curves between L3 (tabular) and L4 (neural) side by side.

**Key takeaway:** Function approximation generalizes across states. But the
predator still has hardcoded "chase" behavior and passively learns a world
model. It has no concept of *seeking information* or *deciding what to do
based on what it knows*.

**Scaffolding still present:** Predator's action policy is hardcoded.

---

## L5: Preferences Replace Hardcoded Behavior

**Concept:** Instead of programming *what to do*, we specify *what the agent
expects to observe* (preferences) and let behavior emerge from minimizing
free energy.

**Core idea:** An active inference agent has:
1. A **generative model** — learned beliefs about how the world works
   (transition model from L4)
2. **Preference priors** — a probability distribution over observations the
   agent "expects" to experience (e.g. "I expect to observe prey at my
   location")
3. **Free energy** — a single scalar that measures the gap between the
   agent's model/preferences and reality

The agent selects actions that minimize **expected free energy**:

    G(a) = ambiguity + risk

- **Ambiguity** = expected uncertainty about observations under action `a`
  → drives *exploration* (go where you'll learn the most)
- **Risk** = divergence between predicted observations and preferred
  observations → drives *exploitation* (go where you'll get what you want)

**Setup:** Remove the predator's hardcoded "move toward believed prey
location" action policy. Replace it with expected free energy minimization:
- The predator's neural net world model (from L4) provides transition
  predictions.
- The predator has a preference prior: high probability of observing prey at
  its own location.
- Action selection: evaluate G for each possible move, pick the one with
  lowest G.

**What the student sees:** The predator *discovers* hunting behavior. Early
on, it explores (ambiguity dominates — the world model is uncertain). As it
learns the prey's patterns, it shifts to exploitation (risk dominates — move
toward the prey). No one told it to chase — chasing *emerges* from
preferences + free energy.

**Key takeaway:** Behavior doesn't need to be programmed. Given a world model
and preferences, free energy minimization produces intelligent behavior.
Exploration and exploitation are not separate mechanisms — they fall out of
the same objective.

**Scaffolding removed:** Predator's action policy. Hardcoded behavior is gone.

---

## L6: Both Agents Use Active Inference

**Concept:** When both agents are active inference agents, complex behavior
emerges from simple preferences without any hardcoding.

**Setup:** The prey also becomes an active inference agent:
- **Prey's generative model:** Learns world transitions (including predator
  movement).
- **Prey's preferences:** "I expect to observe myself alive" and "I expect
  to observe food at my location." No explicit "run from predator" rule.
- **Prey's actions:** Minimize expected free energy given its own model and
  preferences.

The predator keeps its L5 setup. Both agents learn and act simultaneously.

**What the student sees:**
- The prey *discovers* evasion — it learns that the predator moves toward
  it, and that being caught violates its "stay alive" preference, so it
  moves away. Nobody programmed "flee."
- The prey *discovers* foraging — food satisfies its preference prior, so
  it navigates toward food sources. Nobody programmed "seek food."
- An arms race emerges: the prey adapts its evasion as the predator gets
  better at hunting, and the predator adapts as the prey gets better at
  fleeing. Both are continuously learning non-stationary opponents.
- The predator handles the non-stationarity naturally: when its model
  accuracy drops (the prey changed strategy), ambiguity rises, driving
  re-exploration.

**Implementation notes:**
- Add an energy/hunger mechanic: the prey must eat food to survive. The prey
  has a preference for observing high energy (staying fed) and observing
  itself alive.
- Death occurs when the predator catches the prey OR when the prey's energy
  reaches zero. Both pressure the prey to balance foraging and evasion.
- The predator has a preference for observing prey at its location (eating).
- On death, the simulation resets but learned models carry over, so behavior
  improves across episodes.

**Key takeaway:** Complex multi-agent behavior — hunting, fleeing, foraging,
balancing competing goals — emerges from simple preferences and a single
optimization principle. No behavior is hardcoded.

---

## L7: Rich Environments

**Concept:** Active inference handles environmental complexity gracefully
because adding structure just changes the generative model, not the inference
algorithm. This is where hardcoded pathfinding would be impossible, but
active inference agents navigate naturally.

**Setup:** Add walls/obstacles, multiple food sources that respawn, terrain
that affects movement speed, and possibly multiple prey.

**What the student sees:**
- The prey learns to use walls as cover — it discovers that the predator
  can't reach it through walls, and positions itself accordingly. No
  pathfinding algorithm was written.
- The prey learns to plan routes to food that minimize predator exposure.
  This is emergent path planning from free energy minimization.
- The predator learns to ambush — it discovers that certain wall
  configurations funnel prey into dead ends, and it exploits this. Nobody
  programmed ambush behavior.
- Adding a new obstacle doesn't require new code — the agents' generative
  models learn the new transition dynamics and behavior adapts.

**Implementation notes:**
- Walls are cells that block movement (transitions into wall cells are
  impossible). The generative model must learn which transitions are valid.
- Multiple food sources with random respawn keep the prey moving and create
  interesting predator strategies.
- The world model's input representation needs to encode local environment
  features (nearby walls, food) — this is where the neural net's ability to
  generalize from local structure really pays off.

**Key takeaway:** The framework scales to realistic complexity. No hardcoded
pathfinding, no hardcoded evasion tactics, no hardcoded ambush strategies.
The same minimize-free-energy recipe produces increasingly sophisticated
behavior as the environment gets richer. You just expand the generative
model.

---

## Summary of Progression

| Lesson | Introduces | Removes | Key question answered |
|--------|-----------|---------|----------------------|
| L1 | State transitions | — | What is a generative model? |
| L2 | Bayesian learning | — | How do you learn someone else's model? |
| L3 | Joint state | — | What happens when state space grows? |
| L4 | Neural networks | Tabular models | How do you generalize across states? |
| L5 | Free energy + preferences | Predator's hardcoded behavior | Can behavior emerge from preferences alone? |
| L6 | Active inference for both agents | All hardcoded behavior | Can complex multi-agent behavior emerge? |
| L7 | Rich environments | Simplistic grid assumptions | Does this scale to real complexity? |

### The arc in one sentence

We start with hardcoded behavior and a lookup table, progressively strip away
every assumption and hardcoded rule, and end with agents that discover
hunting, fleeing, foraging, and path planning purely from preferences and free
energy minimization.
