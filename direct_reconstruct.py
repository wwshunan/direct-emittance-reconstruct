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

def surface_graph(xs, ys, weights):
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_grid = np.linspace(x_min, x_max, 200)
    y_grid = np.linspace(y_min, y_max, 200)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    weights_grid = interpolate.griddata((xs, ys), weights, (x_mesh, y_mesh))
    #ax = plt.axes(projection='3d')
    #ax.contour3D(x_mesh, y_mesh, weights_grid, 50, cmap='viridis')
    plt.contour(x_mesh, y_mesh, weights_grid, 15, cmap=matplotlib.cm.jet)
    #plt.show()

def part_regen(xs, xps, weights, xgrid, xpgrid, nparts_t, halo):
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
    weights = np.bincount(indxs, weights) / np.bincount(indxs)
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
    rst_phase_space = np.c_[x_rand, xp_rand]
        
    x_new = np.cos(rot_angle) * rst_phase_space[:, 0] - np.sin(rot_angle) * rst_phase_space[:, 1]
    xp_new = np.sin(rot_angle) * rst_phase_space[:, 0] + np.cos(rot_angle) * rst_phase_space[:, 1]
    x_new = x_new + avg_x
    xp_new = xp_new + avg_xp
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

def worker(q1, q2, q3, dx, dxp, dy, dyp):
    txt_r = txt.replace('$1', str(q1))
    txt_r = txt_r.replace('$2', str(q2))
    txt_r = txt_r.replace('$3', str(q3))

    with open('MEBT_emittace.dat', 'w') as f:
        f.write(txt_r)

    cmd_str = './TraceWin MEBT.ini dst_file1=RFQ.dst'
    cmd_str = '%s x1=%s xp1=%s' % (cmd_str, dx * 10, dxp * 1000)
    cmd_str = '%s y1=%s yp1=%s' % (cmd_str, dy * 10, dyp * 1000)
    os.system(cmd_str)

def calc_weights(x_distr_start, y_distr_start, x_distr_end, y_distr_end, x_measure, y_measure,
                 weights_measure, xgrid=200, ygrid=200):
    avg_x = np.average(x_measure, weights=weights_measure)
    var_x = np.average((x_measure - avg_x)**2,  weights=weights_measure)
    avg_y = np.average(y_measure, weights=weights_measure)
    var_y = np.average((y_measure - avg_y)**2,  weights=weights_measure)
    covar = np.average(x_measure * y_measure, weights=weights_measure)
    rot_angle = np.arctan(2 * covar / (var_x - var_y)) / 2

    x_old = x_measure - avg_x
    y_old = y_measure - avg_y
    x_new = np.cos(rot_angle) * x_old + np.sin(rot_angle) * y_old
    y_new = -np.sin(rot_angle) * x_old + np.cos(rot_angle) * y_old
    #x_new, y_new = x_old, y_old

    x_min = min(x_new)
    x_max = max(x_new)
    y_min = min(y_new)
    y_max = max(y_new)
    dx = (x_max - x_min) / xgrid
    dy = (y_max - y_min) / ygrid
    x_min = x_min - dx / 2
    x_max = x_max + dx / 2
    y_min = y_min - dy / 2
    y_max = y_max + dy / 2
    xgrid += 2
    ygrid += 2

    x_range = np.linspace(x_min, x_max, xgrid)
    y_range = np.linspace(y_min, y_max, ygrid)
    #dx = x_range[1] - x_range[0]
    #dxp = y_range[1] - y_range[0]
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    weights_grid = interpolate.griddata((x_new, y_new), weights_measure, (x_grid, y_grid), fill_value=0,
                                        method='linear')

    x_end_new = np.cos(rot_angle) * x_distr_end + np.sin(rot_angle) * y_distr_end
    y_end_new = -np.sin(rot_angle) * x_distr_end + np.cos(rot_angle) * y_distr_end
    #x_end_new, y_end_new = x_distr_end, y_distr_end

    mask = (x_end_new > x_min) & (x_end_new < x_max) & (y_end_new > y_min) & (y_end_new < y_max)
    x_end_mask = x_end_new[mask]
    y_end_mask = y_end_new[mask]
    x_start_mask = x_distr_start[mask]
    y_start_mask = y_distr_start[mask]
    #weights = np.zeros_like(x_end_mask)
    weights = interpolate.griddata((x_grid.ravel(), y_grid.ravel()), weights_grid.ravel(),
                                   (x_end_mask, y_end_mask), fill_value=0, method='linear')
    #surface_graph(x_end_mask, y_end_mask, weights)

    return x_start_mask, y_start_mask, weights

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
    nparts = 800000

    partran_dist = readDis('RFQ_base.dst') 
    xs, xps = create_uniform_distr(xWidth, xpWidth, nparts)
    ys, yps = create_uniform_distr(xWidth, xpWidth, nparts)
    partran_dist[:, 0] = xs
    partran_dist[:, 1] = xps
    partran_dist[:, 2] = ys
    partran_dist[:, 3] = yps

    generate_new_dis(partran_dist, I)

    for i in range(6):
        dx, dxp = np.average(xs), np.average(xps)
        dy, dyp = np.average(ys), np.average(yps)
        worker(q1, q2, q3, dx, dxp, dy, dyp)

        xgrid = 200
        xpgrid = 200
        x, xp, w_x = read_emittance(hor_fname)
        dx = (x.max() - x.min()) / xgrid
        dxp = (xp.max() - xp.min()) / xpgrid
        x_range = np.linspace(min(x), max(x), xgrid)
        xp_range = np.linspace(min(xp), max(xp), xpgrid)
        #dx = x_range[1] - x_range[0]
        #dxp = xp_range[1] - xp_range[0]
        x_grid, xp_grid = np.meshgrid(x_range, xp_range)
        weights_grid_x = interpolate.griddata((x, xp), w_x, (x_grid, xp_grid), fill_value=0)
        #plt.figure()
        #plt.contourf(x_plot, xp_plot, density_interp_x, 15, cmap=matplotlib.cm.jet)
        y, yp, w_y = read_emittance(ver_fname)
        ygrid, ypgrid = 200, 200
        dy = (y.max() - y.min()) / ygrid
        dyp = (yp.max() - yp.min()) / ypgrid
        y_range = np.linspace(min(y), max(y), ygrid)
        yp_range = np.linspace(min(yp), max(yp), ypgrid)
        #dy = y_range[1] - y_range[0]
        #dyp = yp_range[1] - yp_range[0]
        y_grid, yp_grid = np.meshgrid(y_range, yp_range)
        weights_grid_y = interpolate.griddata((y, yp), w_y, (y_grid, yp_grid), fill_value=0)

        distr_slit = exitDis('results/dtl1.plt')

        xs, xps, weights_x = calc_weights(xs, xps, distr_slit[:, 0] * 10, distr_slit[:, 1] * 1000,
                                          x, xp, w_x)

        #weights_x = f_x(distr_new[:, 0] * 10, distr_new[:, 1] * 1000)
        #weights_y = f_y(distr_new[:, 2] * 10, distr_new[:, 3] * 1000)
        ys, yps, weights_y = calc_weights(ys, yps, distr_slit[:, 2] * 10, distr_slit[:, 3] * 1000,
                                          y, yp, w_y)
        #surface_graph(ys, yps, weights_y)
        #plt.plot(ys, yps, 'r.')
        #f = weights_y > weights_y.max() * 0.005

        #plt.plot(ys[f], yps[f], 'b.')
        #xs, xps = xs[distr_filter], xps[distr_filter]
        #ys, yps = ys[distr_filter], yps[distr_filter]

        #plt.plot(ys, yps, 'r.')
        xgrid = xpgrid = 120
        xs, xps = part_regen(xs, xps, weights_x, xgrid, xpgrid, nparts, halo=False)
        ys, yps = part_regen(ys, yps, weights_y, xgrid, xpgrid, nparts, halo=False)
        #plt.plot(ys, yps, 'b.')
        #plt.show()


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



