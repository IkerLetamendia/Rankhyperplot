# author: Iker Letamendia (Ikerletabara@gmail.com)
# created: 2/2023
# summary: Scripts for the visualization of manipulated hypergraphs, taking into account the specifications and rankings selected by the user.


import matplotlib.pyplot as plt, random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import itertools
import toolbag.toolbag.read_write
import hyperplot
from matplotlib.ticker import (MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)

def rankplot(data,rankvariable,dataset,nodevarpr=20,thresholdval=-0.0002,select_EdgeorNode='Edge'):
 savefig = True
 SAVE_DIR = Path.cwd() / 'figs'
 #allplots = ['polygons']
 allplots =['areas','two_rows','planar','polygons']
 n_plots = len(data['orders'])
 nodelabels=None
 if 'node2labels' in data:
  nodelabels = data['node2labels']
 if 'nodeorder' in data:
  nodeorder = data['nodeorder']
 if 'node2colors' in data:
    nodecolors = data['node2colors']

 RSedges=['']
 if select_EdgeorNode is 'Node':
       
       n_high=round(nodevarpr*len(rankvariable)/100)
       idx = np.argpartition(rankvariable, -n_high)[-n_high:] 
       sortrank=sorted(rankvariable)

       if sortrank[len(rankvariable)-1]<=0 and sortrank[0]  <=0:  
                  idx = np.argpartition(rankvariable, n_high)[:n_high] 
       
       RSnodes=list(idx+1)
       ranknames= [n+1 for n in range(0,len(rankvariable))]
       RSnodesnumb=[n+1 for n in  range(0,len(rankvariable))]

       if nodelabels is not None:
          for n in range(0,len(rankvariable)):
           ranknames[n] = nodelabels[n+1] 
       for n in range(0,n_high):
          numb=RSnodes[n]-1   
          RSnodesnumb[n]=RSnodesnumb[numb]
          RSnodes[n]=ranknames[numb]   
       if nodevarpr==0:
          idx=[0] 
          RSnodes=None
          RSnodesnumb=None
 
 edgecolors='gray'

 for plots in allplots:        
  if plots=='polygons':
      fig, axs = plt.subplots(nrows=2, ncols=n_plots , figsize=(n_plots * 4, 8))
      
  if plots=='planar':
      fig, axs = plt.subplots(2, n_plots, figsize=(5 * n_plots, 10))
    
  if plots=='areas':
      fig, axs = plt.subplots(2, n_plots, figsize=(5 * n_plots, 10))
  if plots=='two_rows':
      subplot_width=20
      subplot_height=5
      fig, axs = plt.subplots(n_plots, 2, figsize=(subplot_width, n_plots * subplot_height))      
  for i, kind in enumerate(['red', 'syn']):
    
     for n, order in enumerate(data['edges'][kind].keys()): 
       
         
        if plots=='polygons':
            ax = axs[i, n ]
        if plots =='planar' :   
           ax = axs[i] if n_plots == 1 else axs[i,n ]
        if plots == 'areas':
            ax = axs[i] if n_plots == 1 else axs[i, n]
        if plots =='two_rows': 
            ax = axs[i] if n_plots == 1 else axs[n, i]
      
        edges = data['edges'][kind][order]

        if len(edges) > 0:
        
            data_norm={}
            
            data_min=abs(data['edge2vals'][kind][data['edges'][kind][order][0]])
            data_max=abs(data['edge2vals'][kind][data['edges'][kind][order][0]])
            for numb in range(0,len(data['edges'][kind][order])):
               data_min=abs(data['edge2vals'][kind][data['edges'][kind][order][numb]]) if abs(data['edge2vals'][kind][data['edges'][kind][order][numb]])<data_min else data_min
               data_max=abs(data['edge2vals'][kind][data['edges'][kind][order][numb]]) if abs(data['edge2vals'][kind][data['edges'][kind][order][numb]])>data_max else data_max  
            for numb in range(0,len(data['edges'][kind][order])):
             data_norm[data['edges'][kind][order][numb]]=[(abs(data['edge2vals'][kind][data['edges'][kind][order][numb]])-data_min)/(data_max-data_min)]
            if select_EdgeorNode is 'Edge':
              RSnodes=['']
              RSedges=['a']*len(data["edges"][kind][order])
              numbcount=0
              for numb in range(0,len(data['edges'][kind][order])):  
                if len(data['edges'][kind][order])>numb: 
                  if thresholdval==0 or thresholdval<=0: 
                   if data['edge2vals'][kind][data['edges'][kind][order][numb] ]<thresholdval:
                    RSedges[numb] = data['edges'][kind][order][numb]
                    numbcount=numbcount+1
                  if thresholdval==0 or thresholdval>=0: 
                   if data['edge2vals'][kind][data['edges'][kind][order][numb] ]>thresholdval:
                       RSedges[numb] = data['edges'][kind][order][numb] 
                       numbcount=numbcount+1
                if RSedges[0]=='a':
                     idx=[0] 
                     RSedges=None
                     RSedgesnumb=None
                     break
              if RSedges is not None: 
               RSedges=RSedges[0:numbcount] 
               
            print(RSnodes)

            if plots=='planar':
                hyperplot.rankplanar(edges,RSnodes,RSedges,select_EdgeorNode, nodelabels=nodelabels, ax=ax,data_norm=data_norm)
            if plots=='areas':
                hyperplot.rankareas(edges,RSnodes,RSedges,select_EdgeorNode,nodelabels=nodelabels,nodecolors=nodecolors,edgecolors=edgecolors,ax=ax,data_norm=data_norm,)
            if plots=='two_rows':
               
                hyperplot.ranktwo_rows(edges,RSnodes,RSedges,select_EdgeorNode, nodelabels=nodelabels,nodecolors=nodecolors,nodeorder=nodeorder,ax=ax,column_spacing=2.5, nodesize=0.2,data_norm=data_norm )
            if plots=='polygons':
                hyperplot.rankpolygons(edges,RSnodes,RSedges,select_EdgeorNode,data_norm=data_norm, ax=ax, nodelabels=nodelabels, nodecolors=nodecolors,**{'node_size':0.035})
        else:
            ax.axis('off')
        ax.set_title(f'{kind.upper()} Multiplet Order: {order}', fontsize=16,)
       # ax.set_title(f'Multiplet Order: {order}')
        plt.suptitle(f"{dataset.upper()} Dataset in {plots}", fontsize=20)
        if savefig:
           plt.savefig(SAVE_DIR / f"{dataset}_{plots}.png", dpi=300)

  plt.show()       


