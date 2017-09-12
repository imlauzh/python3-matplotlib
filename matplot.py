plt.figure()
plt.plot(x,y1,linewidth=10)
plt.xlim((-3,2))
plt.ylim((-2,3))
plt.xlabel("I'm x")
plt.ylabel("I'm y")

new_ticks=np.linspace(-1,2,5)
print(new_ticks)
plt.xticks(new_ticks)
plt.yticks([-2,-1.8,-1,1.22,3],
    [r'$very\ bad$',r'$bad$',r'$normal$',r'$good$',r'$very\ \alpha$']
    )
##gca='get current axis'
ax=plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))

l1,=plt.plot(x,y2,label='up')
l2,=plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--',label='down')

plt.legend(handles=[l1,l2],labels=[],loc='best')

x0=0.25
y0=2*x0+1
plt.scatter(x0,y0,s=50,color='b')
plt.plot([x0,x0],[y0,0],'k--',lw=2)

####method 1
plt.annotate(r'$2x+1=%s$'%y0,xy=(x0,y0),xycoords='data',xytext=(+30,-30),textcoords='offset points',fontsize=16,arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))
####method 2
plt.text(-3.5,1,r'$This\ is\ the\ some\ text,\ \mu\ \sigma_i\ \alpha_i$',fontdict={'size':16,'color':'r'})
for label in ax.get_xticklabels()+ax.get_yticklabels():
    label.set_fontsize(12)
    label.set_bbox(dict(facecolor='white',edgecolor='None',alpha=0.7))

n=1024
X=np.random.normal(0,1,n)
Y=np.random.normal(0,1,n)
T=np.arctan2(Y,X)##for color value

plt.scatter(X,Y,s=75,c=T,alpha=0.5)
plt.scatter(np.arange(5),np.arange(5))
##bar
n=12
X=np.arange(n)
Y1=(1-X/float(n))*np.random.uniform(0.5,1.0,n)
Y2=(1-X/float(n))*np.random.uniform(0.5,1.0,n)

plt.bar(X,+Y1,facecolor='#9999ff',edgecolor='white')
plt.bar(X,-Y2,facecolor='#ff9999',edgecolor='white')

for x,y in zip(X,Y1):
    plt.text(x,y+0.05,'%.2f'%y,ha='center',va='bottom')
for x,y in zip(X,Y2):
    plt.text(x,-y-0.05,'%.2f'%y,ha='center',va='top')

plt.xlim((-0.5,n))
plt.ylim((-1.25,1.25))
##等高线绘制
def f(x,y):
    ##height function
    return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)
n=256
x=np.linspace(-3,3,n)
y=np.linspace(-3,3,n)
X,Y=np.meshgrid(x,y)
##use plt.contourf to filling contours
##X,Y and value for (X,Y) point
plt.contourf(X,Y,f(X,Y),8,alpha=0.75,cmap=plt.cm.hot)
##use plt.contour to add contour lines
C=plt.contour(X,Y,f(X,Y),8,color='black',linewidth=0.5)
##adding labels
plt.clabel(C,inline=True,fontsize=10)
plt.yticks(())
plt.xticks(())
##打印3D图像
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
X=np.arange(-4,4,0.25)
Y=np.arange(-4,4,0.25)
X,Y=np.meshgrid(X,Y)
R=np.sqrt(X**2+Y**2)
Z=np.sin(R)

ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))##行跨，列跨
ax.contourf(X,Y,Z,zdir='z',offset=-2,cmap='rainbow')
ax.set_zlim((-2,2))
##多个plot
plt.figure()

plt.subplot(211)
plt.plot([0,1],[0,1])
plt.subplot(234)
plt.plot([0,1],[0,2])
plt.subplot(235)
plt.plot([0,1],[0,3])
plt.subplot(236)
plt.plot([0,1],[0,4])
#分格显示
import matplotlib.gridspec as gridspec
##method 1:subplot2grid
plt.figure()
ax1=plt.subplot2grid((3,3),(0,0),colspan=3,rowspan=1)
ax1.plot([1,3],[1,3])##plot([x1,x2],[y1,y2])
ax1.set_title('ax1')
ax2=plt.subplot2grid((3,3),(1,0),colspan=2,rowspan=1)
ax3=plt.subplot2grid((3,3),(1,2),colspan=1,rowspan=2)
ax4=plt.subplot2grid((3,3),(2,0))
ax4=plt.subplot2grid((3,3),(2,1))
##method 2:gridspec
plt.figure()
gs=gridspec.GridSpec(3,3)
ax1=plt.subplot(gs[0,:])
ax2=plt.subplot(gs[1:,2])
ax3=plt.subplot(gs[1,:2])
ax4=plt.subplot(gs[2,0])
ax5=plt.subplot(gs[2,1])
##method 3:easy to define structure
figure,((ax11,ax12),(ax21,ax22))=plt.subplots(2, 2, sharex=True, sharey=True)
##
fig = plt.figure()
x = [1,2,3,4,5,6,7]
y = [1,3,4,2,5,8,6]

left, bottom, width, height = 0.1,0.1,0.8,0.8
ax1 = fig.add_axes([left,bottom,width,height])
ax1.plot(x,y,'r')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('title')

left, bottom, width, height = 0.2,0.6,0.25,0.25
ax2 = fig.add_axes([left,bottom,width,height])
ax2.plot(y,x,'b')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('inside title 1')

plt.axes([0.6, 0.2, 0.25, 0.25])
plt.plot(y[::-1],x,'g')
plt.xlabel('x')
plt.ylabel('y')
plt.title('inside title 2')

x = np.arange(0,10,0.1)
y1 = 0.05*x**2
y2 = -1*y1

fig, ax1 = plt.subplots()##新用法
ax2 = ax1.twinx()##镜面对称
ax1.plot(x,y1,'g-')
ax2.plot(x,y2,'b--')

ax1.set_xlabel('X data')
ax1.set_ylabel('Y1',color='b')
ax2.set_ylabel('Y2',color='g')

##animation
from matplotlib import animation

fig, ax = plt.subplots()
x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x,np.sin(x))

def animate(i):
    line.set_ydata(np.sin(x+i/10))
    return line,

def init():
    line.set_ydata(np.sin(x))
    return line,

ani = animation.FuncAnimation(fig=fig,func=animate,frames=100,init_func=init,interval=1,blit=False)

plt.show()
