# Yet Another Network Analysis Toolkit.
At the moment, this bad boy containts a few network communication metrics and models at its `core` and a game-theoretical network generation framework at `generative_game_theoric`. I'll just add more stuff as I go.

## Installation
Guess what, `pip install yanat` is all you need.

## Example
See the `examples` folder for more examples.

### Computing the Optimal Influence (OI) state of a given network: Step by step.
So, not to kill the vibe here but `yanat` was firstly designed to make computing (OI) states of networks a breeze. It then grew to include other communication models. Then it grew a bit more to include some network analysis tools. I don't know how far it keeps growing but the trickiest thing is to compute OI. The rest is really just calling one function and getting the result. 
Anyway, let's say you have a network called `connectivity` as a `np.ndarray` adjacency matrix. For the sake of these examples, we assume they are undirected and positively weighted. To make sure the dynamics of the network are well-behaved (stable and happy, unlike Nietzsche), we can normalize the adjacency matrix using the spectral normalization function.

```python
from yanat import core, utils as ut
connectivity = ut.spectral_normalization(1,connectivity)
```

Then you need a model of local dynamics, describing how regions of the network interact with their neighbors. For now, we implemented a simple model that assumes the local dynamics are linear, i.e., the state of a node at time `t+1` is a linear combination of the states of its neighbors at time `t`. This model is implemented in the `yanat.core` module as `simulate_dynamical_system`. You can make it nonlinear by passing (so far) either ``tanh`` or ``relu`` as the `function` argument. Everything is `numba`-compiled so it's pretty fast. For other parameters, you can specify them like: 

```python
NOISE_STRENGTH:float = 1.0 # SD of the Gaussian input noise to each node.
DELTA:float = 0.01 # Time step of the simulation.
TAU:float = 0.01 # Time constant of the local dynamics.
G:float = 0.9 # Coupling strength.
DURATION:float = 10. # Duration of the simulation.

model_params:dict = {"dt": DELTA, "timeconstant": TAU, "coupling": G, "duration": DURATION}

N_NODES:int = connectivity.shape[0] # Number of nodes in the network.
SEED:int = 11 # Random seed for reproducibility.

rng = np.random.default_rng(seed=SEED) # Random generator.
input_noise = rng.normal(0, NOISE_STRENGTH, (N_NODES, int(DURATION / DELTA)))

# Simulate the dynamics of the network. Should look very noisy of course. As long as you don't see NaNs or massive values, you're good.
dynamics = core.simulate_dynamical_system(adjacency_matrix=connectivity, input_matrix=input_noise, function=core.identity, **model_params)
```

Now we have the ingredients to see exactly how much nodes influence on each other. BUT we need to have a **game** (I will explain things better one day I swear). We already implemented a `default_game` in `yanat.core` that receives your `connectivity` and `simulate_dynamical_system` and simulates the **lesioned state** of the network. Everything else comes from another library we wrote called **MSApy** so you can check it out if you want to know more about the process. You can now put everything together and compute the OI state of the network.

```python

game_params:dict = {"adjacency_matrix": connectivity, "input_noise": input_noise, "model_params": model_params}

oi = core.optimal_influence(n_elements=N_NODES, game=default_game, game_kwargs=game_params)
```

This might take literally forever if you don't think it through. The reason is that for every target node, the algorithm lesions potentially hundreds of thousands of source node combinations. This means, running the same simulations for all these constellations of nodes. To put things into perspective, for a network of 220 nodes, it took about half a billion simulations to compute OI. So my advice is to use networks with `N_NODES < 100` locally (still, not laptops maybe) but above that, you should consider using a cluster. Unless you can wait for a few weeks or more leaving things running.

### Computing OI state of a given network using default parameters.
If you don't really care about all those parameters and models and all, you can get going with just two lines:

```python
default_parameters = ut.optimal_influence_default_values(connectivity)
oi = core.optimal_influence(connectivity.shape[0], game_kwargs=default_parameters)
```
BOOM! That's all for now.

## Want to contribute?
We need more `numba`jitted models of dynamics. If you are good at these things, consider contributing:

- Hopf model.
- Kuramoto model.
- FitzHugh-Nagumo model.
- Wilson-Cowan model.
- Delayed linear/nonlinear models.

We also need more network communication/analysis tools. If you are good at these things, consider contributing:

- Navigation efficiency.
- Diffusion efficiency.
- Search information.

- Shortest path algorithms.
- Path retrieval algorithms.

And lastly, we need everything to be as fast as possible. If you are good at these things, consider contributing:

- Parallelization of the algorithms.
- Optimization of the algorithms.
- GPU acceleration of the algorithms.

## How to cite:
If you use `yanat` in your research, you can cite the repository until our paper is out. Here's an example of how to do it:

```
@software{YANAT,
  author = {Kayson Fakhar and Shrey Dixit},
  title = {YANAT: Yet Another Network Analysis Toolkit.},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/kuffmode/YANAT}},
}
```
