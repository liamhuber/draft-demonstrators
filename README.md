# Three ontologically-annotated atomistic workflows

Here we present three physically-meaningful demonstration workflows. They are built on the foundation of the [`pyiron_workflow_atomistics` node library](https://github.com/pyiron/pyiron_workflow_atomistics), and use the [PMDco](https://w3id.org/pmd/co) for ontological annotation of inputs and outputs where available.

## Workflows

### Elastic tensor

The [`ex1-elastic`](ex1-elastic.ipynb) notebook computes the full elastic tensor of an elemental crystal. A bulk unit cell is relaxed, then strained along each of its independent normal and shear components; a linear fit of the resulting stresses gives the 6×6 stiffness tensor, from which the Voigt-Reuss-Hill bulk modulus is derived. The demo uses gold with an effective medium theory (EMT) potential.

### Solute-grain boundary segregation

The [`ex2-grain_boundary`](ex2-grain_boundary.ipynb) notebook measures how strongly different solute atoms prefer grain boundary sites over the bulk. For each symmetrically distinct site at a relaxed grain boundary, it substitutes a solute, relaxes, and compares against the most favourable bulk site to get a segregation energy. These energies are then related to each site's excess Voronoi volume. The demo scans Cu, Ag, Au and Ni in a Σ5 aluminium grain boundary using EMT.

### Unary phase diagram

The [`ex3-phase_stability`](ex3-phase_stability.ipynb) notebook maps which crystal structure of an element is stable at which temperature and pressure. It computes quasiharmonic Gibbs free energies for competing phases across pressure and temperature sweeps, then locates the pressures at which their free energies cross to trace phase boundaries. The demo follows Pb through its FCC → HCP → BCC transitions with an embedded-atom method (EAM) potential.

## Installation

...