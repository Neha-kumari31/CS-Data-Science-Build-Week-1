import numpy as np
import scipy as sp

def dis_fun(p1, p2):
    """Returns the L2 distance between two arrays."""
    return sum((p1-p2)**2)**0.5


class DB_SCAN:
    '''
    Perform DBSCAN clustering
    eps : float, default=0.5
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other.
     min_pts : int, default=5
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point.
    '''
    def __init__(self, eps_radius, min_points, dis_fun = dis_fun):
        self.eps_radius = eps_radius
        self.min_points = min_points
        self.dis_fun = dis_fun
    def get_neighborhood(self, D, point, radius):
        '''Find all points in dataset D within distance 'eps' of point p'''
        return np.array([index for index, element in enumerate(D) if ((self.dis_fun(element, D[point]) <= radius) and (index != point))])
    def expand_cluster(self, point_ind, neighbors, label):

        """ Recursive method which expands the cluster until we have reached the border
        of the dense area (density determined by eps and min_samples) """
        
        self.labels[point_ind] == label

        # Iterate through neighbors - each "neighbor" is an index
        for neighbor in neighbors:
            
            if self.labels[neighbor] == -1: # if the point was labeled noise: we determined it didn't have enough neighbors to be a seed
                self.labels[neighbor] = label # but it is a neighbor, so it must be the boundary of a cluster

            elif self.labels[neighbor] == 0: # if the point has yet to be labeled:
                self.labels[neighbor] = label # label it with the cluster label
                
                # then get the neighbor's neighbors
                neighbors_of_neighbor = self.get_neighborhood(D=self.X, point=neighbor, radius=self.eps_radius)

                if len(neighbors_of_neighbor) >= self.min_points: # if the neighbor has more neighbors than the threshold
                    self.expand_cluster(point_ind=neighbor, neighbors=neighbors_of_neighbor, label=label)


    def fit(self, X):
        self.X = X # input data
        self.labels = [0] * self.X.shape[0] # set all labels for each point to unassigned.
        
        cluster_id = 1 # the id of the current cluster we're adding points to. initialize at 1.

        for point in range(0, self.X.shape[0]): # for each point in dataset:

            if self.labels[point] == 0: # if the point's label is unassigned:

                # get the neighborhood for the point (not including the point itself.)
                neighbors = self.get_neighborhood(D=self.X, point=point, radius=self.eps_radius)

                if len(neighbors) < self.min_points: # if the number of neighbors is below the threshold, label it noise.
                    self.labels[point] = -1
                else: # otherwise, start a cluster from the seed point.
                    self.expand_cluster(point, neighbors=neighbors, label=cluster_id)
                    cluster_id += 1
    
    def fit_predict(self, X):

        self.fit(X)
        return self.labels
    

    
if __name__ == "__main__":

    pass

