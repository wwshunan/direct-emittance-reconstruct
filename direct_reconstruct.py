import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import matplotlib
import os
from wrdis import *
from mpl_toolkits import mplot3d

lattice_name = 'MEBT_emittace.dat'
txt = '''FREQ 162.5  
DRIFT 134 40 0
superpose_map 0 0 0 0 0 0
MATCH_FAM_GRAD 6 0
FIELD_MAP 90 300 0 40 $1 0 0 0 quad1
superpose_map 192 0 0 0 0 0
MATCH_FAM_GRAD 6 0
FIELD_MAP 90 300 0 40 -$2 0 0 0 quad2
superpose_map 379 0 0 0 0 0
MATCH_FAM_GRAD 6 0
FIELD_MAP 90 300 0 40 $3 0 0 0 quad1
superpose_map 629 0 0 0 0 0
;MATCH_FAM_FIELD 6 0
;SET_SYNC_PHASE 
;FIELD_MAP 7700 240 -90 40 0 0 0 0 buncher
DRIFT 190 40 0
DRIFT 77 40 0

;MATCH_FAM_GRAD 6 0
QUAD 80 0 40 0 0 0 0 0 0

DRIFT 122.5 40 0

;slit
;DRIFT 375.5 40 0
end
'''

def part_regen(xs, xps, weights, xgrid, xpgrid, nparts_t, halo):
    #f = weights > 0.1
    #plt.plot(xs[f], xps[f], '.')
    #xs = xs[noise]
    #xps = xps[noise]
    #weights = weights[noise]
    h = np.histogram(weights)
    avg_x = np.average(xs, weights=weights)
    var_x = np.average((xs - avg_x)**2,  weights=weights)
    avg_xp = np.average(xps, weights=weights)
    var_xp = np.average((xps - avg_xp)**2,  weights=weights)
    covar = np.average(xs * xps, weights=weights)
    rot_angle = np.arctan(2 * covar / (var_x - var_xp)) / 2

    x_old = xs - avg_x
    xp_old = xps - avg_xp
    x_new = np.cos(rot_angle) * x_old + np.sin(rot_angle) * xp_old
    xp_new = -np.sin(rot_angle) * x_old + np.cos(rot_angle) * xp_old

    x_min = min(x_new)
    x_max = max(x_new)
    xp_min = min(xp_new)
    xp_max = max(xp_new)
    #r = (x_max - x_min) / (xp_max - xp_min)
    #xpgrid = int(xgrid / r)
    dx = (x_max - x_min) / xgrid
    dxp = (xp_max - xp_min) / xpgrid
    x_min = x_min - dx / 2
    x_max = x_max + dx / 2
    xp_min = xp_min - dxp / 2
    xp_max = xp_max + dxp / 2
    xgrid += 1
    xpgrid += 1

    particle_density = np.zeros((xgrid, xpgrid))
    x_cell = np.floor((x_new - x_min) / dx).astype(np.int)
    xp_cell = np.floor((xp_new - xp_min) / dxp).astype(np.int)
    coords = np.c_[x_cell, xp_cell]
    cell_indices, indxs = np.unique(coords, axis=0, return_inverse=True)
    weights = np.bincount(indxs, weights)
    particle_density[cell_indices[:, 0], cell_indices[:, 1]] = weights

    rst_phase_space = np.zeros((nparts_t, 2))
    dens_sum = np.sum(particle_density)
    particle_density = particle_density / dens_sum

    halo_nparts = 0
    thick_prop = 0.1
    if halo:
        nparts = 5000
        xs_halo, xps_halo = gen_halo(x_new, xp_new, thick_prop, nparts, weights)
        halo_nparts += len(xs_halo)
    else:
        nparts = 0
        xs_halo = []
        xps_halo = []
            
    particle_density = particle_density.reshape(xgrid * xpgrid)
    num_part = nparts_t - halo_nparts
    x_rand = np.zeros(num_part)
    y_rand = np.zeros(num_part)
    cell_indeces = np.random.choice(xgrid * xpgrid,
                                    num_part, p=particle_density)
    x_rand = (np.random.random(num_part) + 
            cell_indeces / xpgrid) * dx + x_min
    xp_rand = (np.random.random(num_part) +
            cell_indeces % xpgrid) * dxp + xp_min
    
    x_rand = np.concatenate((x_rand, xs_halo))
    xp_rand = np.concatenate((xp_rand, xps_halo))
    plt.show()
    rst_phase_space = np.c_[x_rand, xp_rand]
        
    x_new = np.cos(rot_angle) * rst_phase_space[:, 0] - np.sin(rot_angle) * rst_phase_space[:, 1]
    xp_new = np.sin(rot_angle) * rst_phase_space[:, 0] + np.cos(rot_angle) * rst_phase_space[:, 1]
    x_new = x_new + avg_x
    xp_new = xp_new + avg_xp
    #x_min, x_max = xs.min(), xs.max()
    #xp_min, xp_max = xps.min(), xps.max()
    #x_range = np.linspace(x_min, x_max, xgrid)
    #xp_range = np.linspace(xp_min, xp_max, xpgrid)
    #x_grid, xp_grid = np.meshgrid(x_range, xp_range)
    #density = interpolate.griddata((xs, xps), weights, (x_grid, xp_grid))
    #ax = plt.axes(projection='3d')
    ##ax.plot_surface(x_grid, y_grid, grid_weights, rstride=1, cstride=1,
    ##                cmap='viridis', edgecolor='none')
    #ax.contour3D(x_grid, xp_grid, density, 50, cmap='viridis')
    #plt.contourf(x_grid, y_grid, grid_weights, 15, cmap=matplotlib.cm.jet)
    plt.show()
    return x_new, xp_new

def create_uniform_distr(xWidth, yWidth, nparts):
    xs = np.random.random(nparts) * xWidth - xWidth / 2.
    ys = np.random.random(nparts) * yWidth - yWidth / 2.
    xs = xs - np.average(xs)
    ys = ys - np.average(ys)
    return xs, ys

def read_emittance(filename):
    data = np.loadtxt(filename, skiprows=2)
    x = data[:, 0]
    y = data[:, 1]
    weights = data[:, 2]
    x_center = np.average(x, weights=weights)
    y_center = np.average(y, weights=weights)
    x = x - x_center
    y = y - y_center
    return x, y, weights 

def worker(q1, q2, q3):
    txt_r = txt.replace('$1', str(q1))
    txt_r = txt_r.replace('$2', str(q2))
    txt_r = txt_r.replace('$3', str(q3))

    with open('MEBT_emittace.dat', 'w') as f:
        f.write(txt_r)

    cmd_str = './TraceWin MEBT.ini dst_file1=RFQ.dst'
    os.system(cmd_str)

def calc_weights(dx, dy, grid_density, x, y, rot_angle, x_min, x_max, y_min, y_max, xs, ys):
    x_new = np.cos(rot_angle) * x + np.sin(rot_angle) * y
    y_new = -np.sin(rot_angle) * x + np.cos(rot_angle) * y
    mask = (x_new > x_min) & (x_new < x_max) & (y_new > y_min) & (y_new < y_max)
    xs = xs[mask]
    ys = ys[mask]
    x = x_new[mask]
    y = y_new[mask]
    weights = np.zeros_like(x)
    x_int = ((x - x_min) / dx).astype(int)
    y_int = ((y - y_min) / dy).astype(int)
    x_fract = np.fmod(x - x_min, dx)
    y_fract = np.fmod(y - y_min, dy)
    w_lb= (1 - x_fract/dx) * (1 - y_fract/dy) * grid_density[x_int.astype(int) + 1, y_int.astype(int) + 1]
    w_rb= x_fract/dx * (1 - y_fract/dy) * grid_density[x_int.astype(int), y_int.astype(int) + 1]
    w_lu= (1 - x_fract/dx) * y_fract/dy * grid_density[x_int.astype(int) + 1, y_int.astype(int)]
    w_ru=  x_fract/dx * y_fract/dy * grid_density[x_int.astype(int), y_int.astype(int)]
    w = grid_density[x_int.astype(int), y_int.astype(int)]
    weights = w_lb + w_rb + w_lu + w_ru
    #f = w > 0.0001
    #plt.plot(x[f], y[f], '.')
    #plt.show()

    xgrid = xpgrid = 120
    x, xp = xs, ys
    dx = (x.max() - x.min()) / xgrid
    dxp = (xp.max() - xp.min()) / xpgrid
    x_range = np.linspace(min(x), max(x), xgrid)
    xp_range = np.linspace(min(xp), max(xp), xpgrid)
    dx = x_range[1] - x_range[0]
    dxp = xp_range[1] - xp_range[0]
    x_grid, xp_grid = np.meshgrid(x_range, xp_range)
    density = interpolate.griddata((x, xp), weights, (x_grid, xp_grid))
    #ax = plt.axes(projection='3d')
    #ax.plot_surface(x_grid, y_grid, grid_weights, rstride=1, cstride=1,
    #                cmap='viridis', edgecolor='none')
    #ax.contour3D(x_grid, xp_grid, density, 50, cmap='viridis')
    #plt.contourf(x_grid, y_grid, grid_weights, 15, cmap=matplotlib.cm.jet)
    #plt.show()
    return xs, ys, weights

def cast_measure_density_to_grid(xs, xps, weights, xgrid, xpgrid):
    avg_x = np.average(xs, weights=weights)
    var_x = np.average((xs - avg_x)**2,  weights=weights)
    avg_xp = np.average(xps, weights=weights)
    var_xp = np.average((xps - avg_xp)**2,  weights=weights)
    covar = np.average(xs * xps, weights=weights)
    rot_angle = np.arctan(2 * covar / (var_x - var_xp)) / 2

    x_old = xs - avg_x
    xp_old = xps - avg_xp
    x_new = np.cos(rot_angle) * x_old + np.sin(rot_angle) * xp_old
    xp_new = -np.sin(rot_angle) * x_old + np.cos(rot_angle) * xp_old

    x_min = min(x_new)
    x_max = max(x_new)
    xp_min = min(xp_new)
    xp_max = max(xp_new)
    dx = (x_max - x_min) / xgrid
    dxp = (xp_max - xp_min) / xpgrid
    x_min = x_min - dx / 2
    x_max = x_max + dx / 2
    xp_min = xp_min - dxp / 2
    xp_max = xp_max + dxp / 2
    xgrid += 2
    xpgrid += 2

    grid_weights = np.zeros((xgrid, xpgrid))
    left_bottom_grid_weights = np.zeros((xgrid, xpgrid))
    right_bottom_grid_weights = np.zeros((xgrid, xpgrid))
    left_up_grid_weights = np.zeros((xgrid, xpgrid))
    right_up_grid_weights = np.zeros((xgrid, xpgrid))
    x_int = ((x_new - x_min) / dx).astype(int)
    xp_int = ((xp_new - xp_min) / dxp).astype(int)
    x_fract = np.fmod(x_new - x_min, dx)
    xp_fract = np.fmod(xp_new - xp_min, dxp)


    left_bottom_grid_weights[x_int, xp_int] = (1 - x_fract/dx) * (1 - xp_fract/dxp) * weights
    right_bottom_grid_weights[x_int+1, xp_int] = x_fract/dx * (1 - xp_fract/dxp) * weights
    left_up_grid_weights[x_int, xp_int+1] = (1 - x_fract/dx) * xp_fract/dxp * weights
    right_up_grid_weights[x_int+1, xp_int+1] =  x_fract/dx * xp_fract/dxp * weights
    grid_weights = left_bottom_grid_weights + right_up_grid_weights + left_up_grid_weights + right_up_grid_weights
    x_range = np.arange(xgrid)
    y_range = np.arange(xpgrid)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    #ax = plt.axes(projection='3d')
    #ax.plot_surface(x_grid, y_grid, grid_weights, rstride=1, cstride=1,
    #                cmap='viridis', edgecolor='none')
    #ax.contour3D(x_grid, y_grid, grid_weights, 50, cmap='viridis')
    #plt.contourf(x_grid, y_grid, grid_weights, 15, cmap=matplotlib.cm.jet)
    #plt.show()

    return grid_weights, rot_angle, x_min, x_max, xp_min, xp_max, dx, dxp

def main(hor_fname, ver_fname, I, q1, q2, q3, xgrid, xpgrid):
    xWidth = 3
    xpWidth = 5e-2
    nparts = 100000 

    partran_dist = readDis('RFQ_base.dst') 
    xs, xps = create_uniform_distr(xWidth, xpWidth, nparts)
    ys, yps = create_uniform_distr(xWidth, xpWidth, nparts)
    partran_dist[:, 0] = xs
    partran_dist[:, 1] = xps
    partran_dist[:, 2] = ys
    partran_dist[:, 3] = yps

    generate_new_dis(partran_dist, I)
    worker(q1, q2, q3)

    xgrid = 200
    xpgrid = 200
    x, xp, w_x = read_emittance(hor_fname)
    #dx = (x.max() - x.min()) / xgrid
    #dxp = (xp.max() - xp.min()) / xpgrid
    #x_range = np.linspace(min(x), max(x), xgrid)
    #xp_range = np.linspace(min(xp), max(xp), xpgrid)
    #dx = x_range[1] - x_range[0]
    #dxp = xp_range[1] - xp_range[0]
    #x_grid, xp_grid = np.meshgrid(x_range, xp_range)
    density_interp_x, rot_angle_x, x_min, x_max, xp_min, xp_max, dx, dxp  =\
        cast_measure_density_to_grid(x, xp, w_x, xgrid, xpgrid)
    #plt.figure()
    #plt.contourf(x_plot, xp_plot, density_interp_x, 15, cmap=matplotlib.cm.jet)
    y, yp, w_y = read_emittance(ver_fname)
    ygrid, ypgrid = 200, 200
    #dy = (y.max() - y.min()) / ygrid
    #dyp = (yp.max() - yp.min()) / ypgrid
    #y_range = np.linspace(min(y), max(y), ygrid)
    #yp_range = np.linspace(min(yp), max(yp), ypgrid)
    #dy = y_range[1] - y_range[0]
    #dyp = yp_range[1] - yp_range[0]
    #y_grid, yp_grid = np.meshgrid(y_range, yp_range)
    density_interp_y, rot_angle_y, y_min, y_max, yp_min, yp_max, dy, dyp =\
        cast_measure_density_to_grid(y, yp, w_y, ygrid, ypgrid)

    distr_slit = exitDis('results/dtl1.plt')

    #f_x = Rbf(x, xp, w_x)
    #plt.figure()
    #density= f_x(x_plot.ravel(), xp_plot.ravel())
    #plt.contourf(x_plot, xp_plot, density.reshape(200, 200), 15, cmap=matplotlib.cm.jet)
    #plt.show()

    #x_filter = ((distr_slit[:, 0] * 10 < x.max()) & (distr_slit[:, 0] * 10 > x.min()))
    #xp_filter = ((distr_slit[:, 1] * 1000 < xp.max()) & (distr_slit[:, 1] * 1000 > xp.min()))
    #x_plane_filter = (x_filter & xp_filter)

    #f_y = Rbf(y, yp, w_y)
    #y_filter = ((distr_slit[:, 2] * 10 < y.max()) & (distr_slit[:, 2] * 10 > y.min()))
    #yp_filter = ((distr_slit[:, 3] * 1000 < yp.max()) & (distr_slit[:, 3] * 1000 > yp.min()))
    #y_plane_filter = y_filter & yp_filter

    #distr_filter = x_plane_filter & y_plane_filter
    #distr_new = distr_slit[distr_filter]

    #x_slit, xp_slit, y_slit, yp_slit = distr_slit[:, 0], distr_slit[:, 1], distr_slit[:, 2], distr_slit[:, 3]
    #x = np.cos(rot_angle) * x + np.sin(rot_angle) * y
    #y = -np.sin(rot_angle) * x + np.cos(rot_angle) * y
    #mask = (x > x_min) & (x < x_max) & (y > y_min) & (y < y_max)
    xs, xps, weights_x = calc_weights(dx, dxp, density_interp_x, distr_slit[:, 0] * 10, distr_slit[:, 1] * 1000, rot_angle_x,
                             x_min, x_max, xp_min, xp_max, xs, xps)
    #weights_x = f_x(distr_new[:, 0] * 10, distr_new[:, 1] * 1000)
    #weights_y = f_y(distr_new[:, 2] * 10, distr_new[:, 3] * 1000)
    ys, yps, weights_y = calc_weights(dy, dyp, density_interp_y, distr_slit[:, 2] * 10, distr_slit[:, 3] * 1000, rot_angle_y,
                             y_min, y_max, yp_min, yp_max, ys, yps)
    #xs, xps = xs[distr_filter], xps[distr_filter]
    #ys, yps = ys[distr_filter], yps[distr_filter]

    xs, xps = part_regen(xs, xps, weights_x, xgrid, xpgrid, nparts, halo=False)
    ys, yps = part_regen(ys, yps, weights_y, xgrid, xpgrid, nparts, halo=False)

    partran_dist[:, 0] = xs
    partran_dist[:, 1] = xps
    partran_dist[:, 2] = ys
    partran_dist[:, 3] = yps

    generate_new_dis(partran_dist, I)

if __name__ == '__main__':
    h_fname = 'hor-5mA'
    v_fname = 'ver-5mA'
    I = 5
    q1, q2, q3 = 91.5, 104, 88.5
    xgrid = xpgrid = 100
    main(h_fname, v_fname, I, q1, q2, q3, xgrid, xpgrid)



