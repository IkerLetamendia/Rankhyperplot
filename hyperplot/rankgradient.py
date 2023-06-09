# author: Iker Letamendia (Ikerletabara@gmail.com)
# created: 2/2023
# summary: Script for loading and plotting the analysis of O-info gradients using dictionaries and bar charts.


import hypernetx as hnx
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import xgi
import itertools
import scipy.io as sio
import networkx as nx
from networkx import NetworkXException
import hyperplot
import matplotlib as mpl

def rawdata2data(rawdata, min_ord=1, max_ord=2):
    '''
    Input: rawdata (dict with '__header__', 'Otot', 'data' fields).
    Output: dict with Otot fields (sorted_red, index_red, bootsig_red, etc) and 'orders'.
    '''

    data = {'O_val': {}, 'index_var': {}}

    for order in range(min_ord, max_ord+1):
        for key in data.keys():
            tmp = rawdata['goi'][order - 1][key]
            
            if not hasattr(tmp, '__len__'):  # convert matlab singletons to array
                tmp = np.array([tmp])

            if key == 'index_var' :  # VERY IMPORTANT!! matlab to python indexing
                data[key][order] = tmp-1
            else:
                data[key][order] = tmp  # python indexing

    data['orders'] = list(range(min_ord, max_ord+1))
     
    return data
def add_datainfo2data(data, datainfo):
    '''
    Adds 'data', 'n_dims' and 'n_points' fields to data.
    Valid for 'empathy', 'eating', 'PTSD' datasets only.
    '''
    print('Adding data info...')
    data_shape = datainfo.shape
    print(f"shape: {data_shape}")
    data['data'] = datainfo
    data['n_dims'] = data_shape[1]
    data['n_points'] = data_shape[0]

def load_dataset(fpath, min_ord, max_ord, n_dims=None, n_points=None):
    '''
    Load dataset (output from O-info analysis, i.e. Otot structure)
    Returns
    -------
    data : dict ('sorted_red', 'index_red', 'bootsig_red', 'sorted_syn', 'index_syn', 'bootsig_syn')
    '''
    rawdata = loadmat(fpath)
    rawdata['goi'] = [todict(x) for x in rawdata['goi']]
    data = rawdata2data(rawdata, min_ord, max_ord)
    if 'data' in rawdata.keys():
        add_datainfo2data(data, rawdata['data'])
   # else:
    #    data['n_points'] = n_points
     #   data['n_dims'] = n_dims
   
    return data

# functions to load .mat
def loadmat(fname):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    fname = str(fname)
    data = sio.loadmat(fname, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def todict(matobj):
    """
    A recursive function which constructs from mat objects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = todict(elem)
        else:
            dict[strg] = elem
    return dict

def _check_keys(dict_in):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict_in:
        print(key,dict_in['goi'])
        if isinstance(dict_in[key], sio.matlab.mio5_params.mat_struct):
            dict_in[key] = todict(dict_in[key])
    return dict_in

def rankgradient(info,dataset):
 
   DATASET_DIR = Path.cwd() / 'data'
   SAVE_DIR = Path.cwd() / 'figs'
 # LOAD DATA
   print(f'DATASET: {dataset.upper()}')
   print(8 * '=')
   if dataset=='eating':
         fpath = DATASET_DIR / 'GradientsEating.mat'
         data = load_dataset(fpath, min_ord=1, max_ord=2)

   elif dataset=='empathy':
        fpath = DATASET_DIR / 'GradientsBriganti.mat'
        data = load_dataset(fpath, min_ord=1, max_ord=2)
   else:
         raise ValueError('Dataset not accepted.') 
   
   data_Oinfo = { 'syn_index_var':{}, 'red_index_var': {},'syn_O_val':{},'red_O_val':{}}
   data.update(data_Oinfo)
   lennodes=len(data['index_var'][1])
   gradient = {'redcount':[0]* lennodes ,'syncount':[0]*lennodes,'redappear':[0]* lennodes,'synappear':[0]* lennodes}  
   
  
   nodelabels=[(k, info["node2labels"][k]) for k in sorted(info["node2labels"])]
   nodelabels=[ info["node2labels"][k] for k in sorted(info["node2labels"])]

   print(nodelabels )

   
   for order in range(1,len(data['orders'])+1):
             a=0
             
             lst = list(range(1,len(data["index_var"][1])+1))
             all_combos = list(itertools.combinations(lst,order ))
             #print((data["index_var"][2][211:217]))
             #print((all_combos[211:217]))
             #print((data["O_val"][order]))
             pltedg=np.zeros((len(all_combos),order))            
             for z in range(0,len(all_combos)): 
                  #a=data['index_var'][1][z]
                  for t in range(0,order):
                   pltedg[z,t]=int(all_combos[z][t])
                   # pltedg =data['index_var_' + stat][order] 
                   
             pltvls = data['O_val'][order]
             #print(pltvls)
             t=0
             sortpos=0
             sortneg=0
             for n,val in zip(range(0, len(pltvls)),pltvls): 
                 if 0<val:   
                     sortpos=sortpos+1
                 if 0>val:   
                     sortneg=sortneg+1
             
             p1=0
             p2=0
             data['syn_O_val'][order]=np.zeros((sortneg,1))
             data['red_O_val'][order]=np.zeros((sortpos,1))
             data['syn_index_var'][order]=np.zeros((sortneg,order))
             data['red_index_var'][order]=np.zeros((sortpos,order))
             data["adjacency"]=np.zeros((lennodes+1,lennodes+1))
 
             for edge, val in zip(pltedg, pltvls): 
                 #print(edge)
                 [i,j]=pltedg.shape
                
                 for z in range(0,len(data["index_var"][order])+1):

                     for a in range(0,j): 
                              if (z+1)==pltedg[t][a]:
                                  if 211<=t<=217:
                                   
                                   print(val,pltvls[t],pltedg[t],edge,z,pltedg[t][a])
                                  if 0>val:
                                      gradient['syncount'][z]= gradient['syncount'][z]+val
                                      gradient['synappear'][z]= gradient['synappear'][z]+1
                                      
                                  if 0<val:
                                      gradient['redcount'][z]= gradient['redcount'][z]+val
                                      gradient['redappear'][z]= gradient['redappear'][z]+1    
                 
                 if 0>val:                   
                                   data['syn_O_val'][order][p1][0]=val
                                   if order==1:  
                                       data['syn_index_var'][order][p1]=data['index_var'][order][t]+1  
                                      #f=data['index_var'][order][t]+1
                                     # data['syn_index_var'][order]={p1:info['node2labels'][f]}
                                   else:
                                    for a in range(0,j):
                                      #  f=data['index_var'][order][t][a]+1
                                      #  data['syn_index_var'][order]={(p1,a):info['node2labels'][f]}
                                       data['syn_index_var'][order][p1][a]=data['index_var'][order][t][a] +1                     

                                   p1=p1+1
                 if 0<val:                                 
                                   data['red_O_val'][order][p2][0]=val
                                   if order==1:
                                      # f=data['index_var'][order][t]+1
                                       #data['red_index_var'][order]={p2:info['node2labels'][f]}
                                    data['red_index_var'][order][p2]=data['index_var'][order][t]+1 
                                   else:
                                    for a in range(0,j):
                                        #f=data['index_var'][order][t][a]+1
                                       # data['red_index_var'][order]={(p2,a):info['node2labels'][f]}
                                       data['red_index_var'][order][p2][a]=data['index_var'][order][t][a]+1

                                   p2=p2+1
                                       
                 t=t+1 
   
             if order==2:
                for edge, val in zip(pltedg, pltvls): 
                  
                  x =int(edge[0]) 
                  y=int(edge[1])     
                  data['adjacency'][x,y]=val
                  data['adjacency'][y,x]=val 
                  plt.text(x,y,round(val*100),ha='center',va='center')
                  plt.text(y,x,round(val*100),ha='center',va='center')
                mask = np.triu(data['adjacency'])
                z_min, z_max = np.abs(data['O_val'][order]).min(), np.abs(data['O_val'][order]).max()
                print(z_min, z_max)
                plt.imshow( data['adjacency'], cmap = 'RdBu',       vmin = z_min, vmax = z_max,)
                plt.xticks(np.arange(1, lennodes+1, 1),labels=nodelabels,rotation=45, horizontalalignment='right')
                plt.yticks(np.arange(1,lennodes+1, 1),labels=nodelabels,rotation=45, horizontalalignment='right')
                plt.suptitle(f"{dataset.upper()} Dataset in pairs gradients",fontweight ='bold', fontsize = 20)
                plt.xlabel("Node number",fontweight ='bold', fontsize = 15)
                plt.ylabel("Node number",fontweight ='bold', fontsize = 15)
                plt.colorbar()
                plt.show()
                
                plt.bar([x+1 + 0.2 for x in data['index_var'][1]], gradient['redappear'][0:lennodes],color='cyan',width=0.4)
                plt.bar([x+1 - 0.2 for x in data['index_var'][1]],gradient['synappear'][0:lennodes],color='orange',width=0.4)
                plt.xticks(np.arange(1, lennodes+1, 1),labels=nodelabels,rotation=45, horizontalalignment='right')
                plt.yticks(np.arange(0,max([max(gradient['redappear']), max(gradient['synappear'])])+3,3))
                plt.xlabel("Node number",fontweight ='bold', fontsize = 15)
                plt.suptitle(f"{dataset.upper()} Dataset in Nodes appearance", fontweight ='bold', fontsize=20)
                plt.ylabel("Amount of appearances",fontweight ='bold', fontsize = 15)
                plt.legend([ "Redundancy","Synergy"],bbox_to_anchor=(1.05, 1.0), loc='upper left')
                plt.show()
                
                
                
                datawidths=np.zeros(len(data['O_val'][order]))
                for n,i in zip(range(0,len(data['O_val'][order])),data['O_val'][order]):
                 datawidths[n] = float(200*(float(i)/sum(data['O_val'][order])))
                high, *_, low = sorted(data['O_val'][order],reverse=True)
                lim=max([-low,high])
                print(lim,-low,high,sorted(data['O_val'][order]))
                norm = mpl.colors.Normalize(vmin=-lim, vmax=lim, clip=True)
                mapper = mpl.cm.ScalarMappable(norm=norm, cmap='RdBu_r',)
                G = nx.Graph()
                G.add_nodes_from([x + 1 for x in data['index_var'][1]])
                color=np.zeros(len(data['O_val'][order]))
                for i in range(len( data['index_var'][2])):
                       G.add_edges_from([(data['index_var'][2][i,0]+1, data['index_var'][2][i,1]+1)])
                       
                    # G.add_edges_from([(data[stat+'_index_var'][2][i,0], data[stat+'_index_var'][2][i,1])],color=(0.1,0.3,0.3)*data[stat+'_O_val'][2][i])
                nx.draw_networkx(G,pos = nx.spring_layout(G),labels=info['node2labels'],node_color='Green',font_size=10, with_labels=True, edge_color=[mapper.to_rgba(i)  for i in data['O_val'][order]],width=datawidths)
                plt.colorbar(mappable=mapper)
                plt.suptitle(f"{dataset.upper()} Dataset in 2. order Gradients ", fontsize=20)
                plt.show()


                
                for stat in  [ 'red','syn']:

                   datawidths=np.zeros(len(data[stat+'_O_val'][order]))
                   nodewidths=np.zeros(len(gradient[stat+'count']))

                   for n,i in zip(range(0,len(data[stat+'_O_val'][order])),data[stat+'_O_val'][order]):
                     
                    datawidths[n] = float(100*(float(i)/sum(data[stat+'_O_val'][order])))
                   for n,i in zip(range(0,len(nodewidths)),gradient[stat+'count'][0:len(gradient[stat+'count'])]):
                    if stat=='syn':
                      nodewidths[n]= abs(float(5000*(float(i)/sum(gradient[stat+'count'][1:len(gradient[stat+'count'])]))))
                    if stat=='red':
                      nodewidths[n]= abs(float(8000*(float(i)/sum(gradient[stat+'count'][1:len(gradient[stat+'count'])]))))
                    
                
                   low, *_, high = sorted(data[stat+'_O_val'][order])
                   norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
                   mapper = mpl.cm.ScalarMappable(norm=norm, cmap='RdBu')
                   G = nx.Graph()
                   G.add_nodes_from([x + 1 for x in data['index_var'][1]])
                   color=np.zeros(len(data[stat+'_O_val'][order]))
                   for i in range(len( data[stat+'_index_var'][2])):
                       G.add_edges_from([(data[stat+'_index_var'][2][i,0], data[stat+'_index_var'][2][i,1])])
                       
                    # G.add_edges_from([(data[stat+'_index_var'][2][i,0], data[stat+'_index_var'][2][i,1])],color=(0.1,0.3,0.3)*data[stat+'_O_val'][2][i])
                   nx.draw_networkx(G,pos = nx.circular_layout(G),labels=info['node2labels'],font_size=10, with_labels=True, node_size=nodewidths, edge_color=[mapper.to_rgba(i)  for i in data[stat+'_O_val'][order]],width=datawidths)
                   plt.colorbar(mappable=mapper)
                   plt.suptitle(f"{stat.upper()}:{dataset.upper()} Dataset in 2. order Gradients ", fontsize=20)
                   plt.show()
                   
                   G = nx.Graph()
                   G.add_nodes_from([x + 1 for x in data['index_var'][1]])
                   for i in range(len( data[stat+'_index_var'][2])):
                     G.add_edges_from([(data[stat+'_index_var'][2][i,0], data[stat+'_index_var'][2][i,1])])
                   nx.draw_networkx(G,pos = nx.bipartite_layout(G,[x + 1 for x in data['index_var'][1][1:len(data['index_var'][1][1:15])]]), labels=info['node2labels'],with_labels=True, node_size=200,width=datawidths,edge_color=[mapper.to_rgba(i)  for i in data[stat+'_O_val'][order]] )
                   plt.suptitle(f"{stat.upper()}:{dataset.upper()} Dataset in 2. order Gradients ", fontsize=20)
                   plt.show()
                   
                   low, *_, high = sorted(data[stat+'_O_val'][order])
                   norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
                   mapper = mpl.cm.ScalarMappable(norm=norm, cmap='RdBu')
                   G = nx.Graph()
                   G.add_nodes_from([x + 1 for x in data['index_var'][1]])
                   color=np.zeros(len(data[stat+'_O_val'][order]))
                   for i in range(len( data[stat+'_index_var'][2])):
                       G.add_edges_from([(data[stat+'_index_var'][2][i,0], data[stat+'_index_var'][2][i,1])])
                       
                    # G.add_edges_from([(data[stat+'_index_var'][2][i,0], data[stat+'_index_var'][2][i,1])],color=(0.1,0.3,0.3)*data[stat+'_O_val'][2][i])
                   nx.draw_networkx(G,pos = nx.kamada_kawai_layout(G),labels=info['node2labels'],font_size=10, with_labels=True, node_size=nodewidths, edge_color=[mapper.to_rgba(i)  for i in data[stat+'_O_val'][order]],width=datawidths)
                   plt.colorbar(mappable=mapper)
                   plt.suptitle(f"{stat.upper()}:{dataset.upper()} Dataset in 2. order Gradients ", fontsize=20)
                   plt.show()

                   step=((max(gradient[stat+'count'])-min(gradient[stat+'count']))/10)

                   plt.bar([x + 1 for x in data['index_var'][1]],gradient[stat+'count'][0:lennodes],edgecolor="green")
                   plt.xticks(np.arange(1, lennodes+1, 1),labels=nodelabels,rotation=45, horizontalalignment='right')
                   if stat=='red':
                    plt.yticks(np.arange(min(gradient[stat+'count']),max(gradient[stat+'count'])+step,step))
                   if stat=='syn':
                       plt.yticks(np.arange(max(gradient[stat+'count']),min(gradient[stat+'count'])+step,step))
                   plt.xlabel("Node number",fontweight ='bold', fontsize = 15)
                   plt.suptitle(f"{stat.upper()}:{dataset.upper()} Dataset in Values collection",fontweight ='bold', fontsize=20)
                   plt.ylabel("Amount of quantity",fontweight ='bold', fontsize = 15)
                   plt.show()
                   

                   
                   for n in range(0,len(data[stat+'_index_var'][order])):           
                           plt.scatter([data[stat+'_index_var'][order][n][0],data[stat+'_index_var'][order][n][1]],[data[stat+'_index_var'][order][n][1],data[stat+'_index_var'][order][n][0]],linewidth=datawidths[n], marker='o')  
                   plt.xticks(np.arange(1, lennodes+1, 1),labels=nodelabels,rotation=45, horizontalalignment='right')
                   plt.yticks(np.arange(1,lennodes+1, 1),labels=nodelabels,rotation=45, horizontalalignment='right')
                   plt.suptitle(f"{stat.upper()}:{dataset.upper()} Dataset in pairs gradients",fontweight ='bold', fontsize = 20)
                   plt.xlabel("Node number",fontweight ='bold', fontsize = 15)
                   plt.ylabel("Node number",fontweight ='bold', fontsize = 15)
                   plt.grid()
                   #plt.tick_params(axis='y', which='both',step=1, labelleft='on', labelright='on')
                   plt.show()
                   


   
   return gradient,data    


#gradient['synappear']=np.zeros(len(data["index_var"][1]))
