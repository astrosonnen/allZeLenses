import daft


def simple_Mstar_Mh_PGM():

    # Instantiate a PGM.
    pgm = daft.PGM([3.3, 3.0], origin=[0.0, 0.0], grid_unit=2.6, node_unit=1.3, observed_style="inner")

    # Hyper-parameters.
    pgm.add_node(daft.Node("hyperMh", r"$\mu^{\rm h}$", 0.3, 2.0))
    pgm.add_node(daft.Node("hyperMstar", r"$\mu^{\rm *}$", 0.3, 1.0))

    # Latent variables:
    pgm.add_node(daft.Node("Mh", r"$M^{\rm h}_k$", 1.1, 2.0))
    pgm.add_node(daft.Node("Mstar", r"$M^{\rm *}_k$", 1.1, 1.0))

    # Data:
    pgm.add_node(daft.Node("observedMh", r"$M^{\rm h}_{k,\rm obs}$", 1.9, 2.0, observed=True))
    pgm.add_node(daft.Node("observedMstar", r"$M^{\rm *}_{k,\rm obs}$", 1.9, 1.0, observed=True))

    # Constant observational uncertainty:
    pgm.add_node(daft.Node("obssigmaMh", r"$\sigma^{\rm h}_{\rm obs}$", 2.7, 2.0, fixed=True, offset=(0.0,4.0)))
    pgm.add_node(daft.Node("obssigmaMstar", r"$\sigma^{\rm *}_{\rm obs}$", 2.7, 1.0, fixed=True, offset=(0.0,4.0)))

    # Add in the edges.
    pgm.add_edge("hyperMh", "Mh")
    pgm.add_edge("hyperMstar", "Mstar")
    pgm.add_edge("Mh", "Mstar")
    pgm.add_edge("Mh", "observedMh")
    pgm.add_edge("Mstar", "observedMstar")
    pgm.add_edge("obssigmaMh", "observedMh")
    pgm.add_edge("obssigmaMstar", "observedMstar")

    # And a plate for the galaxies
    pgm.add_plate(daft.Plate([0.7, 0.5, 1.6, 2.1], label=r"galaxies $k$", shift=-0.1))

    # Render and save.
    pgm.render()
    pgm.figure.savefig("Simple_Mstar_Mh_PGM.png", dpi=300)

    return
