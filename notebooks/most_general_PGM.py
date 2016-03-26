import daft


def hierarchical_PGM():

    # Instantiate a PGM.
    pgm = daft.PGM([3.3, 3.0], origin=[0.0, 0.0], grid_unit=2.6, node_unit=1.3, observed_style="inner")

    # Hyper-parameters.
    pgm.add_node(daft.Node("hyper", r"$\mu$", 0.3, 2.0))
    pgm.add_node(daft.Node("hubble", r"$h$", 2.7, 2.0))

    # Latent variables:
    pgm.add_node(daft.Node("a", r"$a_k$", 1.1, 2.0))

    # Data:
    pgm.add_node(daft.Node("observeddata", r"$d_{k,\rm obs}$", 1.1, 1.0, observed=True))
    pgm.add_node(daft.Node("observeddeltat", r"$\Delta t_{k,\rm obs}$", 1.9, 1.0, observed=True))

    # Add in the edges.
    pgm.add_edge("hyper", "a")
    pgm.add_edge("a", "observeddata")
    pgm.add_edge("a", "observeddeltat")
    pgm.add_edge("hubble", "observeddeltat")

    # And a plate for the galaxies
    pgm.add_plate(daft.Plate([0.7, 0.5, 1.6, 2.1], label=r"galaxies $k$", shift=-0.1))

    # Render and save.
    pgm.render()
    pgm.figure.savefig("most_general_hierarchical_PGM.png", dpi=300)

    return

def individual_PGM():

    # Instantiate a PGM.
    pgm = daft.PGM([3.3, 3.0], origin=[0.0, 0.0], grid_unit=2.6, node_unit=1.3, observed_style="inner")

    # Latent variables:
    pgm.add_node(daft.Node("a", r"$a_k$", 1.1, 2.0))

    # Data:
    pgm.add_node(daft.Node("observeddata", r"$d_{k,\rm obs}$", 1.1, 1.0, observed=True))

    # Add in the edges.
    pgm.add_edge("a", "observeddata")

    # Render and save.
    pgm.render()
    pgm.figure.savefig("most_general_individual_PGM.png", dpi=300)

    return
