import numpy as np
from matplotlib import pyplot as plt
from .smooth_contour import smooth_contour,smooth_contourf

def scatter_contour(x, y,
                    levels=10,
                    threshold=100,
                    log_counts=False,
                    histogram2d_args=None,
                    plot_args=None,
                    contour_args=None,
                    filled_contour=True,
                    smooth=None,
                    ax=None):
    """Scatter plot with contour over dense regions

    Parameters
    ----------
    x, y : arrays
        x and y data for the contour plot
    levels : integer or array (optional, default=10)
        number of contour levels, or array of contour levels
    threshold : float (default=100)
        number of points per 2D bin at which to begin drawing contours
    log_counts :boolean (optional)
        if True, contour levels are the base-10 logarithm of bin counts.
    histogram2d_args : dict
        keyword arguments passed to numpy.histogram2d
        see doc string of numpy.histogram2d for more information
    plot_args : dict
        keyword arguments passed to plt.plot.  By default it will use
        dict(marker='.', linestyle='none').
        see doc string of pylab.plot for more information
    contour_args : dict
        keyword arguments passed to plt.contourf or plt.contour
        see doc string of pylab.contourf for more information
    filled_contour : bool
        If True (default) use filled contours. Otherwise, use contour outlines.
    smooth : float
        If provided smooth the resulting conours using an interpolated b-spline.
        A value of 0 will pass through all the original contour points, larger 
        value will trade off accuracy for smoothness. This will cause the 
        smooth_contour function to be called, see doc string of smooth_contour
        for additional keywords that can be passed using contour_args.
    ax : pylab.Axes instance
        the axes on which to plot.  If not specified, the current
        axes will be used

    Returns
    -------
    points, contours :
       points is the return value of ax.plot()
       contours is the return value of ax.contour or ax.contourf
    """
    x = np.asarray(x)
    y = np.asarray(y)

    default_plot_args = dict(marker='.', linestyle='none')

    if plot_args is not None:
        default_plot_args.update(plot_args)
    plot_args = default_plot_args

    if histogram2d_args is None:
        histogram2d_args = {}

    if contour_args is None:
        contour_args = {}

    if ax is None:
        ax = plt.gca()

    H, xbins, ybins = np.histogram2d(x, y, **histogram2d_args)
    
    Nx = len(xbins)
    Ny = len(ybins)

    if log_counts:
        H = np.log10(1 + H)
        threshold = np.log10(1 + threshold)

    levels = np.asarray(levels)

    if levels.size == 1:
        levels = np.linspace(threshold, H.max(), levels)

    extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]

    i_min = np.argmin(levels)

    if smooth is not None:
        cont = smooth_contour
        contf = smooth_contourf
        contour_args['smooth'] = smooth
        contour_args['ax']=ax
    else:
        cont=ax.contour
        contf=ax.contourf
    
    # draw a zero-width line: this gives us the outer polygon to
    # reduce the number of points we draw
    # somewhat hackish... we could probably get the same info from
    # the full contour plot below.
    outline = cont(H.T, levels[i_min:i_min + 1],
                         linewidths=0, extent=extent)
    
    if filled_contour:
        contours = contf(H.T, levels, extent=extent, **contour_args)
    else:
        contours = cont(H.T, levels, extent=extent, **contour_args)
    
    # cmk: The original method fails if the lowest contour contains multiple paths.
    #      This method quickly checks for the points that are outside the 
    #      lowest contour, or right on the edge. Then the previous mothod
    #      is checked to clean up the points right on the edge. That way
    #      if lowest contour contains multiple paths it still looks good.
    points_outside = np.zeros_like(x,dtype=np.bool)
    wx,wy = np.where((H<threshold)&(H>0))
    for i,j in zip(wx,wy):
        idx = (x>=xbins[i])&(x<xbins[i+1])&(y>=ybins[j])&(y<ybins[j+1])
        points_outside = points_outside | idx

    #X = np.hstack([x[:, None], y[:, None]])
    X = np.vstack([x[points_outside],y[points_outside]]).T
    
    if len(outline.allsegs[0]) > 0:
        outer_poly = outline.allsegs[0][0]
        try:
            # this works in newer matplotlib versions
            from matplotlib.path import Path
            points_inside = Path(outer_poly).contains_points(X)
        except:
            # this works in older matplotlib versions
            import matplotlib.nxutils as nx
            points_inside = nx.points_inside_poly(X, outer_poly)

        Xplot = X[~points_inside]
    else:
        Xplot = X
    
    points = ax.plot(Xplot[:,0], Xplot[:,1], zorder=1, **plot_args)

    return points, contours
