#series of plotting tools

import pylab
import numpy as np
#from matplotlib import rc
#rc('text',usetex=True)

def rgb_alpha(color,alpha):
    if type(color[0]) == int:
        fcolor = []
        for col in color:
            fcolor.append(col/255.)
    else:
        fcolor = color
        
    R = alpha*fcolor[0] + 1. - alpha
    G = alpha*fcolor[1] + 1. - alpha
    B = alpha*fcolor[2] + 1. - alpha

    return (R,G,B)

def rgb_to_hex(color):
    digits = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
    hexx = '#'
    for col in color:
        col = int(col*255)
        first = digits[col/16]
        second = digits[col%16]
        hexx += first
        hexx += second
    return hexx

def probcontour(xarr,yarr,style='black',smooth=2,bins=100,weights=None,label=None):
    from scipy import ndimage
    import numpy
    H,xbins,ybins = pylab.histogram2d(xarr,yarr,bins=bins,weights=weights)

    H = ndimage.gaussian_filter(H,smooth)
    sortH = numpy.sort(H.flatten())
    cumH = sortH.cumsum()
# 1, 2, 3-sigma, for the old school:
    lvl68 = sortH[cumH>cumH.max()*0.32].min()
    lvl95 = sortH[cumH>cumH.max()*0.05].min()
    lvl99 = sortH[cumH>cumH.max()*0.003].min()
    if style == 'black':
        pylab.contour(H.T,[lvl68,lvl95,lvl99],colors=style,\
                          extent=(xbins[0],xbins[-1],ybins[0],ybins[-1]))

    else:
        # Shaded areas:
        if type(style) == str:
            pylab.contourf(H.T,[lvl99,lvl95],colors=style,alpha=0.2,\
                               extent=(xbins[0],xbins[-1],ybins[0],ybins[-1]))
            pylab.contourf(H.T,[lvl95,lvl68],colors=style,alpha=0.5,\
                               extent=(xbins[0],xbins[-1],ybins[0],ybins[-1]))
            pylab.contourf(H.T,[lvl68,1e8],colors=style,alpha=0.9,\
                               extent=(xbins[0],xbins[-1],ybins[0],ybins[-1]),label=label)
        else:
            pylab.contourf(H.T,[lvl99,lvl95],colors=rgb_to_hex(rgb_alpha(style,0.2)),\
                               extent=(xbins[0],xbins[-1],ybins[0],ybins[-1]))
            pylab.contourf(H.T,[lvl95,lvl68],colors=rgb_to_hex(rgb_alpha(style,0.5)),\
                               extent=(xbins[0],xbins[-1],ybins[0],ybins[-1]))
            pylab.contourf(H.T,[lvl68,1e8],colors=rgb_to_hex(rgb_alpha(style,0.9)),\
                               extent=(xbins[0],xbins[-1],ybins[0],ybins[-1]),label=label)




def cornerplot(stuff,color='b',botm=0.1,leftm = 0.1,height=0.8,width=0.85,nbins=20,fontsize=20,title=None,space=0.1,valcol='r',nticks=None,weights=None):
    N = len(stuff)
    
    fig = pylab.figure()
    pylab.axes([botm,leftm,height,width])
    #makes the histograms above
    pylab.subplots_adjust(left=leftm,right=leftm+width,bottom=botm,top=botm+height,hspace=space,wspace=space)
    for i in range(0,N):
        ax = pylab.subplot(N,N,(N+1)*i + 1)
        #pylab.subplots_adjust(left=leftm,right=leftm+width,bottom=botm,top=botm+height,hspace=space,wspace=space)
        hcolor = color
        if hcolor=='black':
            hcolor='white'
        if type(stuff[i]['data']) == type([]):
            if weights is not None:
                w1,w2 = weights
            else:
                w1 = None
                w2 = None
            pylab.hist(stuff[i]['data'][0],bins=nbins,color='white',weights=w1)
            pylab.hist(stuff[i]['data'][1],bins=nbins,color=hcolor,weights=w2)
        else:
            pylab.hist(stuff[i]['data'],bins=nbins,color=hcolor,weights=weights)
        #yticks = pylab.yticks()
        #pylab.yticks(yticks[0],())
        pylab.yticks(())

        if 'range' in stuff[i]:
            pylab.xlim(stuff[i]['range'][0],stuff[i]['range'][1])

        if 'ticks' in stuff[i]:
            xticks = stuff[i]['ticks']
        elif nticks is not None:
            pxticks = pylab.xticks()[0]
            if len(pxticks) >= nticks:
                step = len(pxticks)/nticks
                xticks = []
                for tickno in np.arange(nticks):
                    xticks.append(xticks[tickno*step])
        else:
            xticks = pylab.xticks()[0]

        if i==N-1:
            if 'label' in stuff[i]:
                pylab.xlabel(r'%s'%(stuff[i]['label']),fontsize=fontsize)
                #pylab.xlabel(r'$\alpha$',fontsize=fontsize)
                pylab.xticks(xticks)
        else:
            pylab.xticks(xticks,())


    for j in range(1,N): #loops over rows 
        for i in range(0,j): #loops over columns 
            ax = pylab.subplot(N,N,N*j+i+1) #for N=4: 5,
            pylab.subplots_adjust(left=leftm,right=leftm+width,bottom=botm,top=botm+height,hspace=space,wspace=space)
            if type(stuff[i]['data']) == type([]):
                if weights is not None:
                    w1,w2 = weights
                else:
                    w1 = None
                    w2 = None
                probcontour(stuff[i]['data'][0],stuff[j]['data'][0],style='black',weights=w1)
                probcontour(stuff[i]['data'][1],stuff[j]['data'][1],style=color,weights=w2)
 
            else:
                probcontour(stuff[i]['data'],stuff[j]['data'],style=color,weights=weights)
            if 'value' in stuff[j] and 'value' in stuff[i]:
                pylab.axvline(stuff[i]['value'],linestyle=':',color='k')
                pylab.axhline(stuff[j]['value'],linestyle=':',color='k')
                pylab.scatter(stuff[i]['value'],stuff[j]['value'],color=valcol)
            if 'range' in stuff[i]:
                pylab.xlim(stuff[i]['range'][0],stuff[i]['range'][1])
            if 'range' in stuff[j]:
                pylab.ylim(stuff[j]['range'][0],stuff[j]['range'][1])

            if 'ticks' in stuff[i]:
                xticks = stuff[i]['ticks']
            elif nticks is not None:
                pxticks = pylab.xticks()[0]
                if len(pxticks) >= nticks:
                    step = len(pxticks)/nticks
                    xticks = []
                    for tickno in np.arange(nticks):
                        xticks.append(xticks[tickno*step])
            else:
                xticks = pylab.xticks()[0]

            if 'ticks' in stuff[j]:
                yticks = stuff[j]['ticks']
            elif nticks is not None:
                pyticks = pylab.yticks()[0]
                if len(pyticks) >= nticks:
                    step = len(pyticks)/nticks
                    yticks = []
                    for tickno in np.arange(nticks):
                        yticks.append(yticks[tickno*step])
            else:
                yticks = pylab.yticks()[0]

            if i==0:
                if 'label' in stuff[j]:
                    pylab.ylabel(r'%s'%(stuff[j]['label']),fontsize=fontsize)
                pylab.yticks(yticks)
            else:
                pylab.yticks(yticks,())

            if j==N-1:
                if 'label' in stuff[i]:
                    pylab.xlabel(r'%s'%(stuff[i]['label']),fontsize=fontsize)
                pylab.xticks(xticks)
            else:
                pylab.xticks(xticks,())

    if title is not None:
        fig.text(0.5,0.975,title,horizontalalignment='center',verticalalignment='top')
        #pylab.subplots_adjust(left=leftm,right=leftm+width,bottom
    
