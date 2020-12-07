import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial import distance

def do_kdtree(combined_x_y_arrays,points):
    mytree = cKDTree(combined_x_y_arrays)
    dist, indexes = mytree.query(points)
    print(dist, indexes)
    return indexes


testing = False

CMAP = 'bwr'

x_pts = 1000

xl = np.array((-4,1))
x_range =  np.linspace(xl[0], xl[1], x_pts)
x_phy_grid_size = sum(abs(xl))/x_pts

yl = np.array((0,2.5))
# makes roughly equidistant
y_nbpts = int(sum(abs(yl))//x_phy_grid_size)

y_range =  np.linspace(yl[0], yl[1], y_nbpts)

x_grid, y_grid = np.meshgrid(x_range, y_range)

r1 = 1
r2 = 2
jet_h = 5

# block
#z = np.where((x_grid<jet_rad) & (x_grid>-jet_rad) & (y_grid<jet_h), 1, 0)

# anulus
h = -(r1+r2)/2
k = 0
circle_z_grid = (x_grid-h)**2+(y_grid-k)**2

z = np.where((np.sqrt(circle_z_grid)>r1) & 
             (np.sqrt(circle_z_grid)<r2), 1, 0)

slice_height =0.1

edges_y = np.diff(z)
edges_x = np.diff(z, axis=0)
#
## fancier edge detection
#sx = ndimage.sobel(z, axis=0, mode='constant')
#sy = ndimage.sobel(z, axis=1, mode='constant')

edges = abs(edges_y)
edges_index= np.argwhere(edges>0)
# plt.scatter(edges_index[:,1],edges_index[:,0])
edges_phyical_value = (x_grid[(edges_index[:,0],edges_index[:,1])],
                       y_grid[(edges_index[:,0],edges_index[:,1])])
# plt.scatter(edges_phyical_value[0],edges_phyical_value[1]) 

combined_x_y_arrays = np.dstack([y_grid[(edges_index[:,0],edges_index[:,1])].ravel(),
                  x_grid[(edges_index[:,0],edges_index[:,1])].ravel()])[0]
                  
foot_pts = np.argwhere(edges_phyical_value[1]==0)
# for actual problem use better method based ont he jet radius location
rhs_foot_point = [edges_phyical_value[0][foot_pts[-1]][0],
                  edges_phyical_value[1][foot_pts[-1]][0]]
lhs_foot_point = [edges_phyical_value[0][foot_pts[-2]][0],
                  edges_phyical_value[1][foot_pts[-2]][0]]
#rhs_foot_pt = np.argsort(abs(edges_phyical_value[0][foot_pts]),2)



#lhs_foot_pt = 

#point = list([combined_x_y_arrays[0].transpose()])
point = list([rhs_foot_point])
tree_index_array = np.zeros(np.shape(combined_x_y_arrays))
tree_index_array[0] = point[0]
combined_x_y_arrays = combined_x_y_arrays[1:]
for i in range(len(combined_x_y_arrays)):
    print(point, combined_x_y_arrays[i-1])
    tree_index = do_kdtree(combined_x_y_arrays, point)
    point = combined_x_y_arrays[tree_index]
    tree_index_array[i+1] = point[0]
    combined_x_y_arrays[tree_index[0]] =  np.nan
    print(i)

#plt.plot(tree_index_array[:,1],tree_index_array[:,0]) 
    

##orig_format
#Opoints = np.random.random(10).reshape(2,5)
#Opoints_list = list(Opoints.transpose())
#print(Opoints_list)




##plt.pcolormesh(x_range, y_range, edges, cmap=CMAP)
#plt.pcolormesh(edges, cmap=CMAP)
#plt.colorbar()
##plt.scatter(edges_phyical_value[0][0],edges_phyical_value[1][0]) 
#plt.show()



## indetifies the slice index
#y_slice_index = np.argmin(abs(y_grid[:,0]-slice_height))
#
## get jet sides indexs
#width_indexs = np.argwhere(abs(edges_y[y_slice_index])==1)
#if testing==True:
#    plt.pcolormesh(x_range, y_range, edges)
#    plt.colorbar()
#    plt.scatter(width_indexs*x_phy_grid_size-abs(min(xl)),
#                y_grid[y_slice_index][:len(width_indexs)], s=50, c='pink')
#    plt.show()


#plt.pcolormesh(x_range, y_range, z)
#plt.colorbar()
#plt.show()


# failed method

#from sklearn.neighbors import NearestNeighbors
#from sklearn.cluster import KMeans

#nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(edges_y)
#distances, indices = nbrs.kneighbors(edges_y)
#data = nbrs.kneighbors_graph(edges_y).toarray()
#
#k_means = KMeans(n_clusters=3)
#k_means.fit(edges_y)
#k_means_predicted = k_means.predict(edges_y)
#
#plt.scatter(edges_y[k_means_predicted!=edges_y,3],c='b', s=50)

#plt.pcolormesh(test, cmap=CMAP)
#plt.colorbar()
#plt.show()