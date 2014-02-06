from matplotlib.contour import *
import matplotlib.pyplot as pyp
from scipy.interpolate import splprep,splev

class SmoothContour(QuadContourSet):
    """
    Wrapper for QuadContourSet to allow for contour smoothing using 
    a periodic b-spline of order k.
    """
    def __init__(self, ax, *args, **kwargs):
        """
        Optional keyword arguments:
        
          *smooth*:
            The user can use *smooth* to control the tradeoff between closeness and smoothness of fit.
            Larger *smooth* means more smoothing while smaller values of s indicate less smoothing.
            A *smooth* of 0 (default) causes the contour to pass through all the original points.
            
          *int_number*:
            The number of interpolated points on each smoothed contour. Default is 100
            
          *k*:
            Degree of the spline. Cubic splines are recommended. Even values of
            k should be avoided especially with a small s-value. 1 <= k <= 5.
            Default is 3
        """
        self.smooth=kwargs.pop('smooth',0)
        self.k=kwargs.pop('k',3)
        self.int_number=kwargs.pop('int_number',100)
        self.kinds_fill=np.ones(self.int_number)*2
        self.kinds_fill[0]=1
        self.unew=np.linspace(0,1,self.int_number)
        ContourSet.__init__(self, ax, *args, **kwargs)

    def _smooth_segs(self,seg,kind,avg=False):
        useg,per=unique_vec(seg)
        if len(useg)>self.k: #check if long enough to smooth
            tck,u=splprep(useg.T,s=self.smooth,k=self.k,per=per)
            seg=np.vstack(splev(self.unew,tck)).T
            if (self.smooth>0) and (avg):
                #when smooth>0 you get different results if you reverse the order of the points
                #do it both ways and take the average of the two solutions
                #and return that instead (needed for filled contours to line up correctly)
                tck2,u2=splprep(useg[::-1].T,s=self.smooth,k=self.k,per=per)
                seg2=np.vstack(splev(self.unew[::-1],tck2)).T
                seg=np.dstack([seg,seg2]).mean(axis=2)
            if kind is not None:
                kind=self.kinds_fill
        return seg,kind
        
    def _get_allsegs_and_allkinds(self):
        """
        Create and return allsegs and allkinds by calling underlying C code.
        """
        allsegs = []
        if self.filled:
            lowers, uppers = self._get_lowers_and_uppers()
            allkinds = []
            for level, level_upper in zip(lowers, uppers):
                nlist = self.Cntr.trace(level, level_upper,
                                        nchunk=self.nchunk)
                nseg = len(nlist) // 2
                segs = nlist[:nseg]
                kinds = nlist[nseg:]
                for i in range(len(segs)): #smooth each contour                    
                    if (kinds[i]==1).sum()>1: #check for breaks in the path (where kinds==1)
                        ndx,=np.where(kinds[i]==1) #find the breaks
                        ndx=np.append(ndx,len(kinds[i])) #add last element to ndx (so it can all go in one loop)
                        for j in range(1,len(ndx)): #split sample based on the breaks and smooth each one
                            current_seg=segs[i][ndx[j-1]:ndx[j]] #grab the current slice
                            current_kind=kinds[i][ndx[j-1]:ndx[j]]
                            s,k=self._smooth_segs(current_seg,current_kind,avg=True) #smooth it
                            if j==1: #inialize arrays to hold path
                                segs_tmp=s
                                kinds_tmp=k
                            else: #re-combine the paths
                                segs_tmp=np.vstack([segs_tmp,s])
                                kinds_tmp=np.append(kinds_tmp,k)
                        segs[i]=segs_tmp
                        kinds[i]=kinds_tmp
                    else:
                        segs[i],kinds[i]=self._smooth_segs(segs[i],kinds[i])
                allsegs.append(segs)
                allkinds.append(kinds)
        else:
            allkinds = None
            for level in self.levels:
                nlist = self.Cntr.trace(level)
                nseg = len(nlist) // 2
                segs = nlist[:nseg]
                kinds=None
                for i in range(len(segs)): #smooth each contour
                    segs[i],kinds=self._smooth_segs(segs[i],kinds,avg=True)
                allsegs.append(segs)
        return allsegs, allkinds

def unique_vec(A):
    ud={}
    per=0
    for idx,i in enumerate(A):
        k=tuple(i)
        ud.setdefault(k,0)
        ud[k]=idx
    gdx=np.sort(ud.values()) #index of unique values
    if gdx[0]!=0:
        gdx=np.append(0,gdx)
        per=1
    return A[gdx],per
    
def smooth_contour(*args,**kwargs):
    kwargs['filled'] = False
    ax=kwargs.pop('ax',pyp.gca())
    washold = ax.ishold()
    hold = kwargs.pop('hold', None)
    if hold is not None:
        ax.hold(hold)
    try:
        ret = SmoothContour(ax,*args,**kwargs)
        pyp.draw_if_interactive()
    finally:
        ax.hold(washold)
    if ret._A is not None: pyp.sci(ret)
    return ret
    
def smooth_contourf(*args,**kwargs):
    kwargs['filled'] = True
    ax=kwargs.pop('ax',pyp.gca())
    washold = ax.ishold()
    hold = kwargs.pop('hold', None)
    if hold is not None:
        ax.hold(hold)
    try:
        ret = SmoothContour(ax,*args,**kwargs)
        pyp.draw_if_interactive()
    finally:
        ax.hold(washold)
    if ret._A is not None: pyp.sci(ret)
    return ret 

if __name__=='__main__':
    from pylab import *
    np.random.seed(0)
    n=100000
    x=np.random.standard_normal(n)
    y=2+3*x+4*np.random.standard_normal(n)
    H,xx,yy=np.histogram2d(x,y,bins=20)
    figure(1)
    subplot(221)
    title('Original')
    contour(xx[:-1],yy[:-1],H)
    subplot(222)
    title('smooth=0')
    smooth_contourf(xx[:-1],yy[:-1],H,smooth=0)
    subplot(223)
    title('smooth=0.1')
    smooth_contourf(xx[:-1],yy[:-1],H,smooth=0.1)
    subplot(224)
    title('smooth=1')
    smooth_contourf(xx[:-1],yy[:-1],H,smooth=1)
    tight_layout()
    show()
