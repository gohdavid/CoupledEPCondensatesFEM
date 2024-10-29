import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.tri as mtri
import matplotlib.colors
import matplotlib.animation

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)

class VideoController:
    def __init__(self, simulation, dt=0.01):
        self.dt        = dt
        self.time_prev = 0
        self.time_next = self.dt
        self.simulation = simulation

        self.writer    = matplotlib.animation.FFMpegWriter(fps=30)
        
        # initialize figure
        gs_kw = dict(width_ratios=[1, 0.05], height_ratios=[1])
        self.fig, axd = plt.subplot_mosaic([['left', 'right']],
                                      gridspec_kw=gs_kw, figsize=(4.0, 3.0), dpi=600,
                                      layout="constrained")
        self.ax = axd["left"]
        self.ax.set_aspect(1)
        
        self._c_idx = simulation._c_idx

        self.triangles_x = mtri.Triangulation(
            np.hstack([self.simulation.mesh.geometry.x[:,0], self.simulation.mesh.geometry.x[:,0]]),
            np.hstack([self.simulation.mesh.geometry.x[:,1], -self.simulation.mesh.geometry.x[:,1]])
        )
        self.data_next = self.simulation.now.fun.vector[:]
        self.c_plot = self.ax.tricontourf(
            self.triangles_x, np.hstack([self.data_next[self._c_idx],self.data_next[self._c_idx]]), levels=np.linspace(0,1,100), cmap="viridis", extend="max", zorder=1)

        self.ax.set_xticks(ticks=[],labels=[])
        self.ax.set_yticks(ticks=[],labels=[])

        #mu_plot.set_clim(0,1)
        cbar = plt.colorbar(self.c_plot, cax=axd["right"])
        cbar.ax.set_yticks([0,0.5,1])
        cbar.set_label("enzyme concentration [$c_+$]")

        scalebar = AnchoredSizeBar(
            self.ax.transData, 1, "$R$", 'lower left', 
            pad=0.1, color='black', frameon=False, label_top=True,
            size_vertical=0.1)

        self.ax.add_artist(scalebar)
        self.label = self.ax.text(0.5,0.95,
                      "time: {:.2} $\\tau_0$".format(self.simulation.now.time), 
                      horizontalalignment='center', verticalalignment='top', transform=self.ax.transAxes, zorder=100)
        
    def update_plot(self):
        
        if self.simulation.now.time > self.time_next:
            self.time_prev = self.time_next
            self.time_next += self.dt
            
            self.data_prev = self.data_next
            self.data_next = self.simulation.now.fun.vector[:]
            data = (self.simulation.now.time - self.time_prev)/self.dt * self.data_prev + (self.time_next - self.simulation.now.time)/self.dt * self.data_next
        
            for coll in self.c_plot.collections:
                coll.remove()

            self.c_plot = self.ax.tricontourf(
                self.triangles_x, np.hstack([data[self._c_idx],data[self._c_idx]]), levels=np.linspace(0,1,100), cmap="viridis", extend="max", zorder=1)
            
            self.label.set_text("time: {:.2} $\\tau_0$".format(self.simulation.now.time))

            self.writer.grab_frame()