from pylab import *
import matplotlib.pyplot as plt

def dark():

    params = {'axes.labelsize': 30,
          'axes.titlesize': 22,
          'font.size': 25,
          'legend.fontsize': 25,
          'xtick.labelsize': 40,
          'ytick.labelsize': 40,
          'text.usetex': True,
          'figure.subplot.left'    : 0.14,
          'figure.subplot.right'   : 0.94  ,
          'figure.subplot.bottom'  : 0.14  ,
          'figure.subplot.top'     : 0.96  ,
          'figure.subplot.wspace'  : 0.  ,
          'figure.subplot.hspace'  : 0.  ,
          #'lines.markersize' : 6,
          #'lines.markeredgewidth'  : 50,
          'lines.linewidth' : 2.5,
          'text.latex.unicode': True,
          'xtick.direction': 'in',
          'ytick.direction': 'in',
          'xtick.minor.size': 4,
          'xtick.minor.width': 2,
          'ytick.minor.size': 4,
          'ytick.minor.width': 2,
          'xtick.major.size': 7.5,
          'xtick.major.width': 2,
          'ytick.major.size': 7.5,
          'ytick.major.width': 2,
          'axes.facecolor'   : 'black',
          'grid.color'       : 'white',
          'xtick.color'      : 'white',
          'ytick.color'      : 'white',
          'figure.facecolor' : 'black',
          'figure.edgecolor' : 'white',
          'axes.edgecolor'   : 'white',
          'axes.labelcolor'  : 'white'    
          }

    rc('font', **{'family': 'sans-serif', 'sans-serif': ['sans-serif']})
    #plt.minorticks_on()
    ion() # Turn on interactive plotting
    rcParams.update(params)

    return "Now using the dark theme"

def default():

    params = {'axes.labelsize': 25,
          'axes.titlesize': 22,
          'axes.linewidth': 1.5,
          'font.size': 20,
          'legend.fontsize': 20,
          'xtick.labelsize': 25,
          'ytick.labelsize': 25,
          'xtick.major.pad': 8,
          'ytick.major.pad': 8,
          'text.usetex': True,
          'figure.subplot.left'    : 0.18,
          'figure.subplot.right'   : 0.94  ,
          'figure.subplot.bottom'  : 0.15  ,
          'figure.subplot.top'     : 0.96  ,
          'figure.subplot.wspace'  : 0.  ,
          'figure.subplot.hspace'  : 0.  ,
          #'lines.markersize' : 6,
          #'lines.markeredgewidth'  : 50,
          'lines.linewidth' : 2.5,
          'text.latex.unicode': True,
          'xtick.direction': 'in',
          'ytick.direction': 'in',
          'xtick.minor.size': 4,
          'xtick.minor.width': 2,
          'ytick.minor.size': 4,
          'ytick.minor.width': 2,
          'xtick.major.size': 7.5,
          'xtick.major.width': 2,
          'ytick.major.size': 7.5,
          'ytick.major.width': 2,
          'xtick.top': True,
          'ytick.right': True,
          'axes.facecolor'   : 'white',
          'grid.color'       : 'black',
          'xtick.color'      : 'black',
          'ytick.color'      : 'black',
          'figure.facecolor' : 'white',
          'figure.edgecolor' : 'black',
          'axes.edgecolor'   : 'black',
          'axes.labelcolor'  : 'black'    
          }


    #rc('font', family='serif')
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['sans-serif']})
    #plt.minorticks_on()
    #ion() # Turn on interactive plotting                                                               
    rcParams.update(params)

    return "Now using the default theme"

def buba():

    params = {'axes.labelsize': 24,#28,
          'axes.titlesize': 22,
          'axes.linewidth': 2.5,
          'font.size': 25,
          'legend.fontsize': 25,
          'xtick.labelsize': 22,#27,#38,
          'ytick.labelsize': 22,#27,#38,
          'xtick.major.pad': 8,
          'ytick.major.pad': 8,
          'text.usetex': True,
          'figure.subplot.left'    : 0.18,
          'figure.subplot.right'   : 0.94  ,
          'figure.subplot.bottom'  : 0.15  ,
          'figure.subplot.top'     : 0.96  ,
          'figure.subplot.wspace'  : 0.  ,
          'figure.subplot.hspace'  : 0.  ,
          #'lines.markersize' : 6,
          #'lines.markeredgewidth'  : 50,
          'lines.linewidth' : 2.5,
          'text.latex.unicode': True,
          'xtick.direction': 'in',
          'ytick.direction': 'in',
          'xtick.minor.size': 4,
          'xtick.minor.width': 2,
          'ytick.minor.size': 4,
          'ytick.minor.width': 2,
          'xtick.major.size': 7.5,
          'xtick.major.width': 2,
          'ytick.major.size': 7.5,
          'ytick.major.width': 2,
          'xtick.top': True,
          'ytick.right': True,
          'axes.facecolor'   : 'white',
          'grid.color'       : 'black',
          'xtick.color'      : 'black',
          'ytick.color'      : 'black',
          'figure.facecolor' : 'white',
          'figure.edgecolor' : 'black',
          'axes.edgecolor'   : 'black',
          'axes.labelcolor'  : 'black'    
          }


    #rc('font', family='serif')
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['sans-serif']})
    #plt.minorticks_on()
    #ion() # Turn on interactive plotting                                                               
    rcParams.update(params)

    return "Now using buba"
