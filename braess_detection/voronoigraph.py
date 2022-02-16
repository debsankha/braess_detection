from scipy.spatial import Voronoi
import numpy as np
import networkx as nx

class VoronoiPlanarGraph():
    """
    add docs here
    """
    def __init__(self, size, points=None, with_coords=True):

        if points:
            vor=Voronoi(points)
        else:
            vor=self._gen_voronoi(size)

        self.with_coords=with_coords

        self.graph=self._voronoi_graph(vor, with_coords=with_coords)
        self.dual_graph=self._voronoi_loopy_dual(vor, with_coords=with_coords)

    def _gen_voronoi(self, size):
        """
        Generate a voronoi tessalation with `size` seed points in the unit cube
        """
        points=np.random.rand(size,2)
        return Voronoi(points)


    def _voronoi_graph(self, vor, with_coords=True):
        """
        Create a planar graph and it's loopy dual from a voronoi tessalation
        """
        G=nx.Graph()
        self.closed_cycles = {}

        #We select all the voronoi ridges that doesn't stretch to infinity
        for num,region in enumerate(vor.regions):
            if region and -1 not in region:
                region_point=np.where(vor.point_region==num)[0][0]

                # detect if the region is clockwise or not
                verts_coords = np.array([vor.vertices[idx] for idx in region])
                # source: https://en.wikipedia.org/wiki/Shoelace_formula
                area_shoelace = np.sum(verts_coords[:,0]*np.roll(verts_coords[:,1],1))\
                                 - np.sum(verts_coords[:,1]*np.roll(verts_coords[:,0],1))

                if area_shoelace < 0:
                    cycle = region
                else:
                    cycle = list(reversed(region))

                nx.add_cycle(G, cycle)
                self.closed_cycles[region_point] = cycle+[cycle[0]] # so, append [2,7,3,2],
                                                            # not [2,7,3]

        if with_coords:
            for node in G.nodes():
                G.nodes[node]['pos']=tuple(vor.vertices[node])

        return G


    def _voronoi_loopy_dual(self, vor, with_coords=True):
        H=nx.MultiGraph()

        points_with_bounded_regions=[]
        for idx, pt in enumerate(vor.points):
            region=vor.regions[vor.point_region[idx]]

            if region and (-1 not in region):
                points_with_bounded_regions.append(idx)

        for pt1,pt2 in vor.ridge_points:
            if pt1 in points_with_bounded_regions:
                if pt2 in points_with_bounded_regions:
                    H.add_edge(pt1,pt2)
                else:
                    H.add_edge(pt1,pt1)
            elif pt2 in points_with_bounded_regions:
                H.add_edge(pt2,pt2)
            else:
                pass

        if with_coords:
            for node in H.nodes():
                H.nodes[node]['pos']=tuple(vor.points[node])

        return H

    def delete_num_edges(self,num):
        edgeidx_todel=np.random.choice(np.arange(self.graph.number_of_edges()), size=num, replace=False)
        alledges=self.graph.edges()
        edges_todel=[alledges[idx] for idx in edgeidx_todel]

        for u,v in edges_todel:
            try:    #we don't do anything if it's a boundary edge
                c1=self.graph[u][v]['cycle1']
                c2=self.graph[u][v]['cycle2']

                self.graph.remove_edge(u,v)

                for nei in self.dual_graph.neighbors(c2):
                   self.dual_graph.add_edge(c1, nei)

                self.dual_graph.remove_node(c2)

                for na,nb in self.graph.edges():
                    if self.graph[na][nb]['cycle1']==c2:
                        self.graph[na][nb]['cycle1']==c1
                    try:
                        if self.graph[na][nb]['cycle2']==c2:
                            self.graph[na][nb]['cycle2']==c1
                    except:
                        pass

            except KeyError:
                pass

    def laplacian(self):
        return np.asarray(nx.laplacian_matrix(self.graph))

    def loopy_laplacian(self):
        A = np.asarray(nx.to_numpy_matrix(self.dual_graph))
        I = np.identity(A.shape[0])
        D = I*np.sum(A,axis=1)
        return D - A + np.diag(np.diag(A))
