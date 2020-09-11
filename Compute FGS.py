import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.axisartist.axislines import SubplotZero
import matplotlib
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA


txt_data_path = ""  #ecognition导出路径
files = [f for f in os.listdir(txt_data_path) if f.endswith('.txt')]
txtfiles=[os.path.join(txt_data_path, f) for f in files if f.endswith('.txt')]
results=[]
for i,file in enumerate(txtfiles):
    data = np.loadtxt(file, skiprows=1,delimiter=';')
    data=data.reshape(1,-1)
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=5) #设精度为15
    results.append(data)
results=np.array(results)
print(results)
results=results[:,0,:]
wv_norm=((results[:,1]-np.min(results[:,1]))/(np.max(results[:,1])-np.min(results[:,1]))).reshape(-1,1)
DTNP_norm=((results[:,2]-np.min(results[:,2]))/(np.max(results[:,2])-np.min(results[:,2]))).reshape(-1,1)
FGS=DTNP_norm-wv_norm
print(results[np.argmax(FGS),0])
plt.plot(results[:,0],FGS,label="FGS",color='black')
plt.xlabel("scale")
plt.ylabel("FGS")
plt.legend()
plt.show()