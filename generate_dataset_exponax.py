"""Generate one reference trajectory for the 1D viscous Burgers equation.

The configuration and Exponax workflow follow the 1D Burgers emulator example:
https://fkoehler.site/exponax/examples/learning_burgers_autoregressive_neural_operator/

The resulting ``burgers_solution_data.npz`` is used only to evaluate the PINN.
It is not part of the physics-informed training loss.
"""

from pathlib import Path

import exponax as ex
import jax
import jax.numpy as jnp
import numpy as np


# Physical and numerical configuration. Change these values, regenerate the
# dataset, and restart the notebook to study a different Burgers problem.
DOMAIN_EXTENT = 1.0
NUM_POINTS = 100
DT = 0.01
DIFFUSIVITY = 0.03
INITIAL_CONDITION_WAVENUMBER_CUTOFF = 5
TEMPORAL_HORIZON = 50
INITIAL_CONDITION_SEED = 0

OUTPUT_PATH = Path(__file__).with_name("burgers_solution_data.npz")


def generate_reference_solution() -> dict[str, np.ndarray]:
    """Return a reproducible Exponax trajectory and its metadata."""
    reference_stepper = ex.stepper.Burgers( num_spatial_dims=1, domain_extent=DOMAIN_EXTENT, num_points=NUM_POINTS, dt=DT, diffusivity=DIFFUSIVITY )
    ic_generator = ex.ic.RandomTruncatedFourierSeries( num_spatial_dims=1, cutoff=INITIAL_CONDITION_WAVENUMBER_CUTOFF )

    # Shape: (num_samples=1, channels=1, num_points)
    initial_condition_set = ex.build_ic_set( ic_generator, num_points=NUM_POINTS, num_samples=1,  key=jax.random.PRNGKey(INITIAL_CONDITION_SEED) )
    initial_condition = initial_condition_set[0]

    # Shape before removing the singleton channel: (Nt + 1, 1, Nx)
    trajectory = ex.rollout( reference_stepper, TEMPORAL_HORIZON, include_init=True )(initial_condition)

    grid = ex.make_grid(1, DOMAIN_EXTENT, NUM_POINTS)[0]
    time = jnp.arange(TEMPORAL_HORIZON + 1) * DT

    return {
        "x": np.asarray(grid),
        "t": np.asarray(time),
        "u": np.asarray(trajectory[:, 0, :]),
        "u0": np.asarray(initial_condition[0]),
        "domain_extent": np.asarray(DOMAIN_EXTENT),
        "num_points": np.asarray(NUM_POINTS),
        "dt": np.asarray(DT),
        "diffusivity": np.asarray(DIFFUSIVITY),
        "initial_condition_wavenumber_cutoff": np.asarray(INITIAL_CONDITION_WAVENUMBER_CUTOFF),
        "temporal_horizon": np.asarray(TEMPORAL_HORIZON),
        "initial_condition_seed": np.asarray(INITIAL_CONDITION_SEED),
        "solver": np.asarray("exponax.stepper.Burgers"),
    }

 
def main() -> None:
    data = generate_reference_solution()
    np.savez_compressed(OUTPUT_PATH, **data)

    print(f"Saved reference data to: {OUTPUT_PATH}")
    print(f"x shape: {data['x'].shape}")
    print(f"t shape: {data['t'].shape}")
    print(f"u shape: {data['u'].shape}")
    print(f"time interval: [{data['t'][0]:.2f}, {data['t'][-1]:.2f}]")


if __name__ == "__main__":
    main()
