import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import itertools
import vtk
import scipy.spatial
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

class SproutingFront:
    def __init__(self, file_name):
        import yaml
        with open(file_name) as f:
            config = yaml.safe_load(f)

        self.px_per_um = config['px_per_um']
        sp = config['sprouting_front']
        self.sp_point_a = np.array((sp['point_a']['x'], sp['point_a']['y'])) / self.px_per_um
        self.sp_point_b = np.array((sp['point_b']['x'], sp['point_b']['y'])) / self.px_per_um
    
    def distance(self, plexus_point):
        assert len(plexus_point) == 2
        dist = np.abs(np.cross(self.sp_point_b-self.sp_point_a, np.array(plexus_point)-self.sp_point_a)/np.linalg.norm(self.sp_point_b-self.sp_point_a))
        return dist

    def is_distance_in_range(self, plexus_point, min_max_range):
        assert len(min_max_range) == 2
        assert (min_max_range[0] < min_max_range[1])
        point_dist_sp = self.distance(plexus_point)
        return point_dist_sp >= min_max_range[0] and point_dist_sp < min_max_range[1]


class NetworkSkeleton:
    def __init__(self, file_name, sprouting_front):
        vtp_reader = vtk.vtkXMLPolyDataReader()
        vtp_reader.SetFileName(file_name)
        vtp_reader.Update()
        self.vtk_centreline = vtp_reader.GetOutput()
        self.sprouting_front = sprouting_front
        self.bifurcation_ids = self._get_bifurcation_ids()

    def _get_bifurcation_ids(self):
        network_graph = self._get_network_graph()
        bifurcation_ids = []
        for point_id in range(self.vtk_centreline.GetNumberOfPoints()):
            node_degree = network_graph.GetDegree(point_id)
            assert node_degree in [1, 2, 3]
            if node_degree == 3:
                bifurcation_ids.append(point_id)

        return bifurcation_ids

    def _get_network_graph(self):
        network_graph = vtk.vtkMutableUndirectedGraph()

        num_vertices = self.vtk_centreline.GetNumberOfPoints()
        for vertex_id in range(num_vertices):
            network_graph.AddVertex()

        edges = self.vtk_centreline.GetLines()
        edge = vtk.vtkIdList()
        edges.InitTraversal()
        while edges.GetNextCell(edge):
            network_graph.AddGraphEdge(edge.GetId(0), edge.GetId(1));

        return network_graph

    def plot_nodes_in_range(self, min_max_range):
        coord_list = []
        coord_list_region = []
        for point_id in range(self.vtk_centreline.GetNumberOfPoints()):
            point_coord = self.vtk_centreline.GetPoints().GetPoint(point_id)[:2]
            coord_list.append(point_coord)

            point_dist_sp = self.sprouting_front.distance(point_coord)

            if point_dist_sp >= min_max_range[0] and point_dist_sp < min_max_range[1]:
                coord_list_region.append(point_coord)

        #import pdb; pdb.set_trace()
        coord_list = np.array(coord_list)
        coord_list_region = np.array(coord_list_region)
        plt.plot(coord_list[:, 0], coord_list[:, 1], 'b+')
        plt.plot(coord_list_region[:, 0], coord_list_region[:, 1], 'r+')
        plt.plot(coord_list[self.bifurcation_ids, 0], coord_list[self.bifurcation_ids, 1], 'k+')
        plt.show()


    def bifurcation_density_for_range(self, min_max_range, plot_network=False):
        assert len(min_max_range) == 2

        if plot_network:
            self.plot_nodes_in_range(min_max_range)

        # coord_list, area = coords_area_for_range(min_max_range)
        coord_list = []
        for bifurcation_id in self.bifurcation_ids:
            point_coord = self.vtk_centreline.GetPoints().GetPoint(bifurcation_id)[:2]
            point_dist_sp = self.sprouting_front.distance(point_coord)

            if point_dist_sp >= min_max_range[0] and point_dist_sp < min_max_range[1]:
                coord_list.append(point_coord)

        if len(coord_list) < 3:
            return 0.0

        # We are passing to ConvexHull a polygon defined in a 2D space, therefore .volume its the area of the polygon (while .area would be its perimeter)
        # area = scipy.spatial.ConvexHull(coord_list).volume
        area = self.area_for_range(min_max_range)

        return len(coord_list) / area

    def coords_area_for_range(self, min_max_range):
        assert len(min_max_range) == 2

        coord_list = []
        # for bifurcation_id in self.bifurcation_ids:
        #     point_coord = self.vtk_centreline.GetPoints().GetPoint(bifurcation_id)[:2]
        for point_id in range(self.vtk_centreline.GetNumberOfPoints()):
            point_coord = self.vtk_centreline.GetPoints().GetPoint(point_id)[:2]
            point_dist_sp = self.sprouting_front.distance(point_coord)

            if point_dist_sp >= min_max_range[0] and point_dist_sp < min_max_range[1]:
                coord_list.append(point_coord)

        if len(coord_list) < 3:
            return coord_list, 0.0

        # hull = scipy.spatial.ConvexHull(coord_list)
        # points = np.array(coord_list)
        # plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'b--', lw=2)
        # print 'area', hull.area
        # print 'volume', hull.volume
        # plt.show()

        # We are passing to ConvexHull a polygon defined in a 2D space, therefore .volume its the area of the polygon (while .area would be its perimeter)
        return coord_list, scipy.spatial.ConvexHull(coord_list).volume

    def area_for_range(self, min_max_range):
        return self.coords_area_for_range(min_max_range)[1]

    def vascularised_density_for_range(self, min_max_range, plot_network=False):
        assert len(min_max_range) == 2

        if plot_network:
            self.plot_nodes_in_range(min_max_range)

        num_vertices = self.vtk_centreline.GetNumberOfPoints()
        vertex_coordinates = [np.array(self.vtk_centreline.GetPoints().GetPoint(vertex_id)) for vertex_id in range(num_vertices)]
        vertex_radius = vtk_to_numpy(self.vtk_centreline.GetPointData().GetArray('Radius'))

        vessel_areas = []
        edges = self.vtk_centreline.GetLines()
        edge = vtk.vtkIdList()
        edges.InitTraversal()
        coord_list_region = []
        unique_pairs = set()   
        unique_sorted_pairs = set()
        while edges.GetNextCell(edge):
            point_a_id = edge.GetId(0)
            point_b_id = edge.GetId(1)

            point_a_coords = vertex_coordinates[point_a_id]
            point_b_coords = vertex_coordinates[point_b_id]

            assert(point_a_coords[2] == point_b_coords[2])
            is_edge_in_range = self.sprouting_front.is_distance_in_range(point_a_coords[:2], min_max_range) and self.sprouting_front.is_distance_in_range(point_b_coords[:2], min_max_range)

            if is_edge_in_range:
                pair = (point_a_id,point_b_id)
                unique_pairs.add(pair)
                unique_sorted_pairs.add(tuple(sorted(pair)))
                coord_list_region.append(point_a_coords)
                coord_list_region.append(point_b_coords)
                distance = np.sqrt(vtk.vtkMath.Distance2BetweenPoints(point_a_coords, point_b_coords))
                mean_diameter = 2 * np.mean([vertex_radius[point_a_id], vertex_radius[point_b_id]])
                vessel_areas.append(distance*mean_diameter)

        # import pdb; pdb.set_trace()
        # coord_list_region = np.array(coord_list_region)
        # plt.plot(coord_list_region[:, 0], coord_list_region[:, 1], 'r+')
        # for coord1, coord2 in zip(coord_list_region[:-1:2], coord_list_region[1::2]):
        #     tmp_coords = np.array([coord1, coord2])
        #     plt.plot(tmp_coords[:,0], tmp_coords[:,1])
        # plt.xlim(0, 1000)
        # plt.ylim(0, 1000)
        #plt.show()

        total_vessel_area = np.sum(vessel_areas)
        total_area = self.area_for_range(min_max_range)
        assert(total_area > 0.)

        print "here", len(vessel_areas), '/', self.vtk_centreline.GetNumberOfLines(), total_vessel_area, total_area
        return total_vessel_area / total_area

    def diameters_for_range(self, min_max_range):
        num_vertices = self.vtk_centreline.GetNumberOfPoints()
        vertex_coordinates = [np.array(self.vtk_centreline.GetPoints().GetPoint(vertex_id)) for vertex_id in range(num_vertices)]
        vertex_radii = vtk_to_numpy(self.vtk_centreline.GetPointData().GetArray('Radius'))

        diameters = [vertex_radius for vertex_coord, vertex_radius in zip(vertex_coordinates, vertex_radii) if self.sprouting_front.is_distance_in_range(vertex_coord[:2], min_max_range)]
        return diameters
