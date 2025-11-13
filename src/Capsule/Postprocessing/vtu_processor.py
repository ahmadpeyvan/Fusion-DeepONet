import numpy as np
import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import matplotlib.pyplot as plt

class VTUProcessor:
    def __init__(self, input_vtu_file, variable):
        """
        Initialize the MeshProcessor by loading the first VTU file from the given directory,
        building the cell locator, and precomputing edge connectivity.
        """
        self.input_vtu_file = input_vtu_file
        # Read VTU file.
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(self.input_vtu_file)
        reader.Update()
        self.original_grid = reader.GetOutput()
        
        # Build cell locator and precompute connectivity.
        self.cellLocator = self.build_cell_locator(self.original_grid)
        self.edge_counts = self.precompute_edge_counts(self.original_grid)
        self.variable = variable
    
    def distance_point_to_segment(self, p, a, b):
        """Compute the shortest distance from point p to the line segment [a, b]."""
        p = np.array(p)
        a = np.array(a)
        b = np.array(b)
        v = b - a
        w = p - a
        c1 = np.dot(w, v)
        if c1 <= 0:
            return np.linalg.norm(p - a)
        c2 = np.dot(v, v)
        if c2 <= c1:
            return np.linalg.norm(p - b)
        t = c1 / c2
        projection = a + t * v
        return np.linalg.norm(p - projection)
    
    def projection_param(self, p, p0, p1):
        """
        Compute the scalar projection (parameter t) of point p onto the line defined by p0 and p1.
        If the line is parameterized as p0 + t*(p1-p0), this returns t.
        """
        p = np.array(p)
        p0 = np.array(p0)
        p1 = np.array(p1)
        d = p1 - p0
        norm_d = np.linalg.norm(d)
        if norm_d == 0:
            return 0.0
        direction = d / norm_d
        t = np.dot(p - p0, direction)
        return t
    
    def build_cell_locator(self, grid):
        """Build and return a cell locator for the given grid."""
        cellLocator = vtk.vtkCellLocator()
        cellLocator.SetDataSet(grid)
        cellLocator.BuildLocator()
        return cellLocator
    
    def find_cells_along_line(self, cellLocator, points, tol):
        """
        Find cell IDs along the line from line_start to line_end using the pre-built cell locator.
        """
        intersectedCellIds = vtk.vtkIdList()
        cellLocator.FindCellsAlongLine(line_start, line_end, tol, intersectedCellIds)
        return intersectedCellIds
    
    def extract_cells_from_locator(self, cellLocator, grid, line_start, line_end, tol):
        """
        Extract cells intersecting the line using the provided cell locator and grid.
        """
        intersectedCellIds = self.find_cells_along_line(cellLocator, line_start, line_end, tol)
        num_cells = intersectedCellIds.GetNumberOfIds()
        # print(f"Number of cells intersecting the line: {num_cells}")
        if num_cells == 0:
            print("No cells found intersecting the line.")
            return None
        extract = vtk.vtkExtractCells()
        extract.SetInputData(grid)
        extract.SetCellList(intersectedCellIds)
        extract.Update()
        return extract.GetOutput()
    
    def precompute_edge_counts(self, grid, tol=1e-5):
        """
        Precompute a dictionary mapping each edge (represented by its sorted endpoint tuples)
        to the number of cells in which it appears.
        """
        edge_counts = {}
        num_cells = grid.GetNumberOfCells()
        for i in range(num_cells):
            cell = grid.GetCell(i)
            for j in range(cell.GetNumberOfEdges()):
                edge = cell.GetEdge(j)
                pts = edge.GetPoints()
                if pts is None or pts.GetNumberOfPoints() < 2:
                    continue
                pt0 = pts.GetPoint(0)
                pt1 = pts.GetPoint(pts.GetNumberOfPoints()-1)
                decimals = 8  
                candidate = tuple(sorted([tuple(np.round(pt0, decimals=decimals)),
                                          tuple(np.round(pt1, decimals=decimals))]))
                edge_counts[candidate] = edge_counts.get(candidate, 0) + 1
        return edge_counts
    
    def find_boundary_edge(self, cell, edge_counts, tol=1e-6):
        """
        For a given cell, examine its edges and return the first boundary edge (one that appears
        only in one cell in the original grid).
        """
        num_edges = cell.GetNumberOfEdges()
        for edge_index in range(num_edges):
            edge = cell.GetEdge(edge_index)
            pts = edge.GetPoints()
            if pts is None or pts.GetNumberOfPoints() < 2:
                continue
            pt0 = pts.GetPoint(0)
            pt1 = pts.GetPoint(pts.GetNumberOfPoints()-1)
            decimals = 8  
            candidate = tuple(sorted([tuple(np.round(pt0, decimals=decimals)),
                                      tuple(np.round(pt1, decimals=decimals))]))
            count = edge_counts.get(candidate, 0)
            if count == 1:
                return edge_index, vtk_to_numpy(pts.GetData())
        return None, None
    
    def find_closest_cell_and_boundary_edge(self, extracted_grid, edge_counts, line_start, line_end, original_grid, tol=1e-6):
        """
        Among the extracted cells, find the cell whose centroid projects closest along the line
        (starting from line_start). Then, in that cell, find a boundary edge using the precomputed
        edge connectivity. Additionally, retrieve the "v1" values for the edge's points from the original grid.
    
        Returns a tuple:
          (closest_cell_id, boundary_edge_index, boundary_edge_points, v1_values)
        """
        num_cells = extracted_grid.GetNumberOfCells()
        if num_cells == 0:
            print("No cells in the extracted grid.")
            return None, None, None, None

        min_t = float('inf')
        closest_cell_id = None
        # Loop over cells to find the one whose centroid projects closest to the line_start along the line.
        for i in range(num_cells):
            cell = extracted_grid.GetCell(i)
            pts = cell.GetPoints()
            npts = pts.GetNumberOfPoints()
            centroid = np.zeros(3)
            for j in range(npts):
                centroid += np.array(pts.GetPoint(j))
            centroid /= npts
            t_val = self.projection_param(centroid, line_start, line_end)
            if t_val < min_t:
                min_t = t_val
                closest_cell_id = i

        # print(f"Closest cell ID (by centroid projection): {closest_cell_id}")
        closest_cell = extracted_grid.GetCell(closest_cell_id)
        boundary_edge_index, boundary_edge_points = self.find_boundary_edge(closest_cell, edge_counts, tol)
        
        if boundary_edge_points is None:
            print("No boundary edge found in the closest cell.")
            return closest_cell_id, None, None, None

        # Retrieve the "v1" values for each point on the boundary edge.
        # print("self.variable=\t",self.variable)
        v1_data = original_grid.GetPointData().GetArray(self.variable)
        if v1_data is None:
            raise ValueError("v1 variable not found in the original grid's point data!")
        v1_array = vtk_to_numpy(v1_data)
        
        v1_values = []
        for pt in boundary_edge_points:
            pt_id = original_grid.FindPoint(pt)
            if pt_id == -1:
                v1_values.append(np.nan)
            else:
                v1_values.append(v1_array[pt_id])
        v1_values = np.array(v1_values)
        
        return closest_cell_id, boundary_edge_index, boundary_edge_points, v1_values

    # def find_index_vectorized(self,pt, pts, tol):
    #     # Compute distances from pt to all points in pts
    #     print("pts=\t",pts.shape)
    #     print("pt=\t",pt.shape)
    #     dists = np.mean(np.abs(pts - pt[None,:]),axis=1)
    #     print("dists=\t",dists.shape)
    #     idx = np.argmin(dists)
    #     if dists[idx] < tol:
    #         return idx
    #     else:
    #         return None

    def find_index_vectorized(self,pt, pts, tol):
        # Compute distances from pt to all points in pts
        # print("pts=\t",pts.shape)
        # print("pt=\t",pt.shape)
        dists1 = np.mean(np.abs(pts - pt[None,0,:]),axis=-1)
        dists2 = np.mean(np.abs(pts - pt[None,1,:]),axis=-1)
        # print("dists=\t",dists.shape)
        min_val1 = np.min(dists1)
        indices1 = np.argwhere(dists1 == min_val1)
        min_val2 = np.min(dists2)
        indices2 = np.argwhere(dists2 == min_val2)

        for vec1 in indices1:
            for vec2 in indices2:
                if vec1[0]==vec2[0]:
                    target_cell = vec1[0]
                    target_nodes=np.array([vec1[1],vec2[1]])

        return target_cell,target_nodes
        # print("Minimum indices for each row:", row_min_indices)
        # col, row = np.unravel_index(flat_index, dists.shape)
        # print("Minimum indices for each row:", row)
        # print("Minimum indices for each column:", col)
        # print("idx=\t",idx)
        # if dists[idx] < tol:
        #     return idx
        # else:
        #     return None

    def find_cell_index(self, matrix, vec):
        a, b = vec
        # print("matrix=\t",matrix.shape)
        # print("a,b=\t",vec)
        # Check which rows contain the integer 'a'
        rows_with_a = np.any(matrix == int(a), axis=1)
        # Check which rows contain the integer 'b'
        rows_with_b = np.any(matrix == int(b), axis=1)
        
        # Find rows where both conditions are True
        valid_rows = np.where(rows_with_a )
        valid_rowsb = np.where(rows_with_b )
        
        # Return the first valid row index, or -1 if none exist
        return valid_rows[0], valid_rowsb[0]
    
    def process_boundaries(self, edge_points, tol=1e-15):
        """
        Process boundaries over a range of angles and return the collected points and solution values.
        """
        points_on_boundaries = []
        solution_on_boundaries = []

        points = vtk_to_numpy(self.original_grid.GetPoints().GetData())
        cells_np = vtk_to_numpy(self.original_grid.GetCells().GetData())
        num_cells = self.original_grid.GetNumberOfCells()
        cell = self.original_grid.GetCell(0)
        num_points_in_cell = cell.GetNumberOfPoints()

        cell_points_coord = np.zeros((num_cells,num_points_in_cell,3))
        cell_points_ids = np.zeros((num_cells,num_points_in_cell),dtype=int)

        sol_data = vtk_to_numpy(self.original_grid.GetPointData().GetArray(self.variable))

        for cell_index in range(num_cells):
            cell = self.original_grid.GetCell(cell_index)
            cell_type = cell.GetCellType()
            

            # Gather the point indices for this cell.
            num_points_in_cell = cell.GetNumberOfPoints()
            point_ids = [cell.GetPointId(i) for i in range(num_points_in_cell)]
            
            # Use vectorized numpy indexing to get the coordinates of the cell's nodes.
            point_in_cell= points[point_ids, :]
            cell_points_ids[cell_index,:] = point_ids
            # print("point_incell=\t",point_incell.shape)
            cell_points_coord[cell_index,:,:]=point_in_cell
        
        # print("cell_points_coord=\t",cell_points_coord.shape)


        for vec in edge_points:
            cell_id,nodes_id = self.find_index_vectorized(vec, cell_points_coord, tol)
            # print("cell_id=\t",cell_id)
            # print("nodes_id=\t",nodes_id)

            cell_coords= cell_points_coord[cell_id]
            # print("cell_coords=\t",cell_coords.shape)

            cell = self.original_grid.GetCell(cell_id)
            num_edges = cell.GetNumberOfEdges()
            points_ids = cell_points_ids[cell_id,:]
            for edge_index in range(num_edges):
                edge = cell.GetEdge(edge_index)
                pts_ids = []
                for i in range(edge.GetNumberOfPoints()):
                    pts_ids.append(edge.GetPointId(i))
                edge_pts_ids= np.array(pts_ids)

                # print("edge_pts_ids=\t",edge_pts_ids)

                if points_ids[nodes_id[0]] in edge_pts_ids and points_ids[nodes_id[1]] in edge_pts_ids:
                    # print("edge=\t",edge_index)
                    pts = vtk_to_numpy(edge.GetPoints().GetData())
                    # print("edge_points1=\t",pts1[:,0:2])
                    points_on_boundaries.append(pts.copy())
                    sol_val=[]
                    for pt in pts:
                        pt_id = self.original_grid.FindPoint(pt)
                        sol_val.append(sol_data[pt_id])

                    sol_values = np.array(sol_val)
                    solution_on_boundaries.append(sol_values)
        
        points_on_boundaries = np.array(points_on_boundaries)
        solution_on_boundaries = np.array(solution_on_boundaries)
        # print("points_on_boundaries =", points_on_boundaries)
        # print("solution_on_boundaries =", solution_on_boundaries.shape)
        
        return points_on_boundaries, solution_on_boundaries
    
    def plot_boundaries(self, points, solution):
        """
        Plot the boundary points colored by solution values.
        """
        plt.scatter(points[:, 0], points[:,1], s=2)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Boundary Points")
        plt.show()
