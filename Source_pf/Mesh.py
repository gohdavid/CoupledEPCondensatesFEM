import gmsh
import dolfinx
import dolfinx.io
import numpy as np
from dolfinx.io import gmshio

from enum import IntEnum
from dataclasses import dataclass

# create some markers
class DomainType(IntEnum):
    Solvent = 2
    
# create some markers
class BoundaryType(IntEnum):
    Solvent = 12

# define parameters with dimensions
@dataclass
class DomainParameters:
    L       : float = 7.0    # system size in units of [sqrt(D / k_2)]
    dim     : int   = 3
    
"""
This is a helper function to create a disk mesh of size L, with a small disk of radius R inside.
"""
class Mesh:
    
    def __init__(self, MPI_COMM, parameters):
                 #R, L, dimension = 2, aspect = 1):

        assert parameters.dim <= 3 and parameters.dim >= 2,     "Only dimensions 2 or 3 are sensible!"

        # Store dimension
        self.dimension = parameters.dim
        
        # Set some variables that depend on the dimensionality of the mesh
        mesh_dim     = 2
        mesh_elem    = "Triangle"

        # Initialize gmsh
        gmsh.initialize()

        # Silent running
        gmsh.option.set_number("General.Verbosity", 0)
        gmsh.option.set_number('General.Terminal', 0)
        
        # Add points that delimit background and droplet
        solvent_pl      = gmsh.model.occ.add_point(-parameters.L, 0, 0)
        solvent_pr      = gmsh.model.occ.add_point(parameters.L, 0, 0)
        solvent_pt      = gmsh.model.occ.add_point(0, parameters.L, 0)
        solvent_pc      = gmsh.model.occ.add_point(0, 0, 0)

        # Connect points to baselines
        solvent_b       = gmsh.model.occ.add_line(solvent_pr, solvent_pl)
        solvent_c1      = gmsh.model.occ.add_circle_arc(solvent_pl, solvent_pc, solvent_pt)
        solvent_c2      = gmsh.model.occ.add_circle_arc(solvent_pt, solvent_pc, solvent_pr)

        # Define closed curves for solvent, and fill them
        solvent_curve = gmsh.model.occ.add_curve_loop([solvent_b, solvent_c1, solvent_c2])
        solvent_fill  = gmsh.model.occ.add_plane_surface([solvent_curve])

        # build the model
        gmsh.model.occ.synchronize()
        
        # add physical domains
        gmsh.model.add_physical_group(mesh_dim, [solvent_fill], tag=DomainType.Solvent)

        # add physical boundaries
        gmsh.model.add_physical_group(mesh_dim-1, [solvent_c1, solvent_c2], tag=BoundaryType.Solvent)

        # Generate mesh
        gmsh.option.set_number("Mesh.Algorithm", 5)
        gmsh.option.set_number("Mesh.CharacteristicLengthMin", 2.0e-2)
        gmsh.option.set_number("Mesh.CharacteristicLengthMax", 5.0e-2)
        gmsh.model.mesh.generate(mesh_dim-1)
        gmsh.model.mesh.generate(mesh_dim)

        #Mesh.CharacteristicLenghtFactor
        
        # Get mesh geometry
        self.mesh, self.cell_markers, self.facet_markers = gmshio.model_to_mesh(gmsh.model, MPI_COMM, 0, gdim=mesh_dim)

        # Close gmsh and clean up
        gmsh.finalize()