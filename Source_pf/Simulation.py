import ufl
import tqdm
import dolfinx
import tempfile
import numpy as np
import scipy.integrate
from petsc4py import PETSc

from types import SimpleNamespace
from dataclasses import dataclass

# create cache directory
#self.cachedir = tempfile.TemporaryDirectory()
#"cache_dir": self.cachedir.name

"""
DOLFINx compilation parameters.
"""
_jit_params = {"cffi_extra_compile_args": ["-O3"], "cffi_libraries": ["m"]}

"""
Simulation parameters.
"""
# define parameters with dimensions
@dataclass
class Parameters:
    R       : float = 1.0    # droplet radius in units of [sqrt(D / k_2)]
    x0      : float = 0.0    # initial droplet location in units of [sqrt(D / k_2)]
    width   : float = 1.0e-1 # width of the interface in units of [sqrt(D / k_2)]
    c_out   : float = 0.1    # concentration in the dilute phase in units of [c_+]
    χ       : float = -0.05  # enzyme-substrate interaction in units of [r]
    M       : float = 1.0    # enzyme mobility in units of [D / (r c_+)]
    dt      : float = 1.0e-3 # in units of [1/k_2]

"""
"""
def get_binodal(parameters, dim = 2, laplace_term=True):
    # concentration difference between the two minima of the Cahn-Hilliard free energy density
    Δc    = 1.0 - parameters.c_out
    # concentration at the local maximum of the Cahn-Hilliard free energy density
    c̃     = parameters.c_out + 0.5*Δc
    # get surface tension
    τ     = (1.0/6.0) * Δc**2 * parameters.width
    p_L   = (dim-1)*τ/parameters.R if laplace_term else 0

    # chemical potential in the homogeneous approximation
    def μ(c):
        return -(c-c̃) + 4*(c-c̃)**3/Δc**2

    def condition_1(c_vals):
        c_max, c_min = c_vals
        return μ(c_max) - μ(c_min)
    def condition_2(c_vals):
        c_max, c_min = c_vals
        integral, _ = scipy.integrate.quad(μ, c_min, c_max)
        return integral - (c_max * μ(c_max) - c_min * μ(c_min)) + p_L

    optimization = scipy.optimize.minimize(lambda c_vals: condition_1(c_vals)**2 + condition_2(c_vals)**2, [1, parameters.c_out])
    
    potential = np.mean([μ(c) for c in optimization.x])
    return list(optimization.x), potential

"""
Define a helper class that prepares the system for PETSc.
"""
class NonlinearProblem:
    def __init__(self, F, u, bcs=[]):
        # Create form and PETSc Vector for residual
        self._F_Form    = dolfinx.fem.form(F, jit_params = _jit_params)
        self.F_Vector   = dolfinx.fem.petsc.create_vector(self._F_Form)
        
        # Create form and PETSc Matrix for Jacobian
        self._J_Form    = dolfinx.fem.form(ufl.derivative(F, u, ufl.TrialFunction(u.function_space)), jit_params  = _jit_params)
        self.J_Matrix   = dolfinx.fem.petsc.create_matrix(self._J_Form)
        
        # Store pointers to boundary conditions and variables, so that we can access them
        self.bcs        = bcs
        self._variables = u        

    def F(self, snes, x, F):
        """Assemble residual vector."""
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self._variables.vector)
        self._variables.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        F.zeroEntries()
        dolfinx.fem.petsc.assemble_vector(F, self._F_Form)
        dolfinx.fem.petsc.apply_lifting(F, [self._J_Form], bcs=[self.bcs], x0=[x], scale=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(F, self.bcs, x, -1.0)

    def J(self, snes, x, J, P):
        """Assemble Jacobian matrix."""
        J.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(J, self._J_Form, bcs=self.bcs)
        J.assemble()        

"""
"""
class Simulation():
    def __init__(self, MPI_COMM, mesh, parameters, initialize_noise=0.05):
        
        self.mesh, self.cell_markers, self.facet_markers, self.dimension = mesh.mesh, mesh.cell_markers, mesh.facet_markers, mesh.dimension
        
        # Define function spaces
        P1 = ufl.FiniteElement("Lagrange", self.mesh.ufl_cell(), 1)
        W  = dolfinx.fem.FunctionSpace(self.mesh, ufl.MixedElement([P1 for _ in range(2)]), jit_params = _jit_params)

        # define concentration field which will drive condensate
        V = dolfinx.fem.FunctionSpace(self.mesh, P1, jit_params = _jit_params)
        self.m = dolfinx.fem.Function(V)
        self.m.x.array[:] = 0.0
        
        # Functions
        # current timestep
        self.now       = SimpleNamespace()
        self.now.fun   = dolfinx.fem.Function(W)
        self.now.c, self.now.μ = ufl.split(self.now.fun)

        # previous timestep
        self.pre       = SimpleNamespace()
        self.pre.fun   = dolfinx.fem.Function(W)
        self.pre.c, self.pre.μ = ufl.split(self.pre.fun)
        
        # Test function spaces
        test           = SimpleNamespace()
        test.fun       = ufl.TestFunction(W)
        test.c, test.μ = ufl.split(test.fun)

        """
        Interpolate initial conditions
        """
        self.now.fun.x.array[:] = 0.0        
        rng = np.random.default_rng(1000)
        
        def droplet_indicator(x):
            return np.linalg.norm(x.T - [parameters.x0, 0, 0], axis=1) < parameters.R

        enzyme, potential = get_binodal(parameters, dim = mesh.dimension, laplace_term=True)
            
        # initialize condensate
        V, self._c_idx = W.sub(0).collapse()
        tmp = dolfinx.fem.Function(V)
        tmp.interpolate(lambda x: np.where(droplet_indicator(x), enzyme[0], enzyme[1]))
        self.now.fun.vector[self._c_idx] = tmp.vector[:]

        # initialize chemical potential
        V, V_to_W = W.sub(1).collapse()
        tmp = dolfinx.fem.Function(V)
        tmp.interpolate(lambda x: np.repeat(potential, x.shape[1]))
        self.now.fun.vector[V_to_W] = tmp.vector[:]
        
        self.now.fun.x.scatter_forward()

        """
        Setup physical processes in non-dimensionalized form
        """
        # concentration difference between the two minima of the Cahn-Hilliard free energy density
        Δc    = 1.0 - parameters.c_out
        # concentration at the local maximum of the Cahn-Hilliard free energy density
        c̃     = parameters.c_out + 0.5*Δc
        
        # chemical potential of the Cahn-Hillard model
        dfdc  = -(self.now.c-c̃) + 4*(self.now.c-c̃)**3/Δc**2
        
        # setup mass currents
        self.now.fluxes    = SimpleNamespace()
        self.now.fluxes.c  = - parameters.M * ufl.grad(self.now.μ + parameters.χ * self.m)
        
        """
        Setup weak form of the equations in non-dimensionalized form
        """
        # time step, which will actually be made adaptive
        self.dt = dolfinx.fem.Constant(self.mesh, parameters.dt)
                
        # variational (weak) form of the system of equations
        # use backward euler method for time discretization
        self.weak_form     = SimpleNamespace()
        self.weak_form.c   = (self.now.c - self.pre.c)*test.c - self.dt * ufl.dot(self.now.fluxes.c, ufl.grad(test.c))
        self.weak_form.μ   = (self.now.μ - dfdc)*test.μ       - 0.5*parameters.width**2 * ufl.dot(ufl.grad(self.now.c), ufl.grad(test.μ))
        
        # 3d simulations have a modified variational problem and a different integration factor
        # due to the axisymmetric coordinate system that we use
        integration_factor = 1.0 if mesh.dimension != 3 else ufl.SpatialCoordinate(self.mesh)[1]
        
        # set up final variational form, which is a functional
        self.weak_form.functional = (self.weak_form.c + self.weak_form.μ) * integration_factor * ufl.dx
        
        # Get Jacobian
        self.weak_form.jacobian   = ufl.derivative(self.weak_form.functional, self.now.fun, ufl.TrialFunction(W))        
        
        # problem and solver statement
        self.problem = NonlinearProblem(self.weak_form.functional, self.now.fun)
        
        """
        Create SNES Solver, and associate the local MPI handle with it.
        """
        self.solver = PETSc.SNES().create(MPI_COMM)
        self.solver.setFunction(self.problem.F, self.problem.F_Vector)
        self.solver.setJacobian(self.problem.J, self.problem.J_Matrix)

        # use a direct solver
        self.solver.setTolerances(rtol=1.0e-8, atol=1.0e-9, max_it=10)
        self.solver.getKSP().setType("preonly")
        self.solver.getKSP().setTolerances(rtol=1.0e-8)
        self.solver.getKSP().setTolerances(atol=1.0e-9)
        self.solver.getKSP().getPC().setType("lu")
        self.solver.getKSP().getPC().setFactorSolverType("mumps")
        
        # finish initialization
        self.pre.fun.x.array[:] = self.now.fun.x.array[:]
        self.solver.solve(None, self.now.fun.vector)
        
        # set up measurements
        ϕ = ufl.conditional(ufl.ge(self.now.c, c̃), 1, 0)
        
        # set up measurement for droplet size
        self.measure  = SimpleNamespace()
        self.measure.volume = dolfinx.fem.form(
            ϕ * integration_factor * ufl.dx,
            jit_params = _jit_params
        )
        
        # set up measurement for droplet position
        self.measure.weighted_position = dolfinx.fem.form(
            ϕ * ufl.SpatialCoordinate(self.mesh)[0] * integration_factor * ufl.dx,
            jit_params = _jit_params
        )
        
        # set up measurement for total enzyme mass
        self.measure.enzyme_mass = dolfinx.fem.form(
            self.now.c * integration_factor * ufl.dx,
            jit_params = _jit_params
        )
        
        # set up measurement for chemical potential imbalance
        self.measure.chem_pot_imbalance = dolfinx.fem.form(
            ϕ * self.now.μ.dx(0) * integration_factor * ufl.dx,
            jit_params = _jit_params
        )

        self.measure.gradient_m = dolfinx.fem.form(
            ϕ * self.m.dx(0) * integration_factor * ufl.dx,
            jit_params = _jit_params
        )
        
        # set up start time and position
        self.now.time     = 0.0
        self.now.position = self.get_position()
       
    """
    """
    def get_total_mass(self):
        return dolfinx.fem.assemble_scalar(self.measure.enzyme_mass)
    
    """
    """
    def get_volume(self):
        return dolfinx.fem.assemble_scalar(self.measure.volume)
       
    """
    """
    def get_position(self):
        return dolfinx.fem.assemble_scalar(self.measure.weighted_position) / self.get_volume()
    
    """
    """
    def get_chemical_potential_imbalance(self):
        return dolfinx.fem.assemble_scalar(self.measure.chem_pot_imbalance)

    """
    """
    def get_gradient_m(self):
        return dolfinx.fem.assemble_scalar(self.measure.gradient_m)
    
    """
    """
    def step(self, max_timestep=np.inf):
        # populate previous timestep
        self.pre.fun.x.array[:] = self.now.fun.x.array[:]
        
        # helper function for adaptive timestepping
        # this basically ensures that the number of timesteps until convergence is relatively small
        def finished():
            self.solver.solve(None, self.now.fun.vector)
            
            Δ = (self.now.fun.x.array[self._c_idx] - self.pre.fun.x.array[self._c_idx]) / self.pre.fun.x.array[self._c_idx]            
            if np.abs(Δ).max() > 2.5e-1 or self.solver.getIterationNumber() > 6 or not self.solver.getConvergedReason() > 0:                
                self.now.fun.x.array[:] = self.pre.fun.x.array[:]
                self.dt.value /= 1.5
                return False
            elif (np.abs(Δ).max() < 1.0e-3 or self.solver.getIterationNumber() < 4) and self.dt.value * 1.25 <= max_timestep:
                self.now.time += self.dt.value
                self.dt.value *= 1.25
                return True
            else:
                self.now.time += self.dt.value
                return True
        
        while not finished():
            pass
        
    """
    """
    def run(self, duration, max_timestep=np.inf, callbacks = []):
        with tqdm.tqdm(total=duration) as progressbar:
            while (progressbar.n < progressbar.total):
                # propagate the simulation a step forward
                self.step(max_timestep=max_timestep)
                
                # execute callback functions
                for callback in callbacks:
                    callback()
                    
                # update progress bar
                progressbar.n = min(self.now.time, progressbar.total)
                progressbar.refresh()
