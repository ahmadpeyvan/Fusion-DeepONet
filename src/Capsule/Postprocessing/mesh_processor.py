import numpy as np
import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import matplotlib.pyplot as plt

class MeshProcessor:
    """
    A class to parse a mesh file, process mesh elements, compute normals,
    and plot the mesh data.
    """
    
    def __init__(self, mesh_file):
        """
        Initialize the MeshProcessor with a mesh file.
        
        Parameters:
            mesh_file (str): Path to the mesh file.
        """
        self.mesh_file = mesh_file
        self.heading = None
        self.nodes = None          # numpy array: each row is [node_id, x, y, z]
        self.elements = {}         # dict keyed by (elset_name, element_type) with numpy arrays
        self.element_sets = {}     # dict keyed by ELSET names with numpy arrays of element IDs
        self.node_sets = {}        # dict keyed by NSET names with numpy arrays of node IDs
        self.filtered_data = None  # Filtered elements after applying blunt element filtering
        self.coords = None         # Node coordinates corresponding to the blunt set
        self.normals = None        # Computed normals for filtered elements
        self.edge_points = None    # Edge points corresponding to filtered elements
        
        # Parse the mesh file upon initialization
        self.parse_mesh_file()

    def parse_mesh_file(self):
        """
        Parses the mesh file specified in self.mesh_file.
        
        Sets:
            self.heading: The heading information from the mesh file.
            self.nodes: Numpy array of node data.
            self.elements: Dictionary of element data.
            self.element_sets: Dictionary of element set IDs.
            self.node_sets: Dictionary of node set IDs.
        """
        heading = None
        nodes = []
        elements = {}       # key: (elset_name, element_type)
        element_sets = {}   # key: ELSET name
        node_sets = {}      # key: NSET name

        with open(self.mesh_file, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            if line.startswith("*"):
                # Split keyword and parameters
                parts = [p.strip() for p in line.split(",")]
                keyword = parts[0].upper()  # e.g., "*NODE", "*ELEMENT", etc.
                params = {}
                for p in parts[1:]:
                    if "=" in p:
                        key, val = p.split("=")
                        params[key.upper()] = val

                if keyword == "*HEADING":
                    i += 1  # Next line should contain heading info
                    if i < len(lines):
                        heading = lines[i].strip()
                    i += 1
                    continue

                elif keyword == "*NODE":
                    i += 1
                    while i < len(lines) and not lines[i].strip().startswith("*"):
                        node_line = lines[i].strip()
                        parts = [p for p in node_line.split(",") if p.strip() != ""]
                        if len(parts) >= 4:
                            node_id = int(parts[0])
                            coords = [float(coord) for coord in parts[1:4]]
                            nodes.append([node_id] + coords)
                        i += 1
                    continue

                elif keyword == "*ELEMENT":
                    # Get element type and ELSET name from parameters
                    element_type = params.get("TYPE", "")
                    elset_name = params.get("ELSET", "default")
                    key = (elset_name, element_type)
                    if key not in elements:
                        elements[key] = []
                    i += 1
                    while i < len(lines) and not lines[i].strip().startswith("*"):
                        elem_line = lines[i].strip()
                        parts = [p for p in elem_line.split(",") if p.strip() != ""]
                        if parts:
                            elem_id = int(parts[0])
                            conn = []
                            for p in parts[1:]:
                                if '.' in p:
                                    try:
                                        conn.append(float(p))
                                    except ValueError:
                                        conn.append(p)
                                else:
                                    try:
                                        conn.append(int(p))
                                    except ValueError:
                                        conn.append(p)
                            elements[key].append([elem_id] + conn)
                        i += 1
                    continue

                elif keyword == "*ELSET":
                    elset_name = params.get("ELSET", "default")
                    if elset_name not in element_sets:
                        element_sets[elset_name] = []
                    i += 1
                    while i < len(lines) and not lines[i].strip().startswith("*"):
                        group_line = lines[i].strip()
                        parts = [p for p in group_line.split(",") if p.strip() != ""]
                        for p in parts:
                            try:
                                element_sets[elset_name].append(int(p))
                            except ValueError:
                                pass
                        i += 1
                    continue

                elif keyword == "*NSET":
                    nset_name = params.get("NSET", "default")
                    if nset_name not in node_sets:
                        node_sets[nset_name] = []
                    i += 1
                    while i < len(lines) and not lines[i].strip().startswith("*"):
                        group_line = lines[i].strip()
                        parts = [p for p in group_line.split(",") if p.strip() != ""]
                        for p in parts:
                            try:
                                node_sets[nset_name].append(int(p))
                            except ValueError:
                                pass
                        i += 1
                    continue

                else:
                    i += 1
                    continue
            else:
                i += 1

        self.heading = heading
        self.nodes = np.array(nodes) if nodes else None
        self.elements = {key: np.array(val) for key, val in elements.items()}
        self.element_sets = {name: np.array(val) for name, val in element_sets.items()}
        self.node_sets = {name: np.array(val) for name, val in node_sets.items()}

    def filter_blunt_elements(self, blunt_key="Blunt", line_identifier="Line"):
        """
        Filters elements and node coordinates corresponding to the 'blunt' part of the mesh.
        
        Parameters:
            blunt_key (str): Key in element_sets and node_sets that identifies the blunt part.
            line_identifier (str): Substring to identify line elements in the elements keys.
        
        Sets:
            self.filtered_data: Numpy array of filtered element data.
            self.coords: Numpy array of node coordinates for the blunt part.
        """
        blunt_elem_no = self.element_sets.get(blunt_key)
        if blunt_elem_no is None:
            raise ValueError(f"Blunt element set '{blunt_key}' not found in element_sets.")

        elem_list = []
        for key, arr in self.elements.items():
            elset, etype = key
            if line_identifier in elset:
                elem_list.append(arr)
        if not elem_list:
            raise ValueError(f"No elements found with identifier '{line_identifier}' in element set keys.")

        elems = np.concatenate(elem_list)
        mask = np.isin(elems[:, 0], blunt_elem_no)
        self.filtered_data = elems[mask]

        print("self.filtered_data=\t",self.filtered_data.shape)

        blunt_nodes = self.node_sets.get(blunt_key)
        if blunt_nodes is None:
            raise ValueError(f"Blunt node set '{blunt_key}' not found in node_sets.")
        mask_nodes = np.isin(self.nodes[:, 0].astype(int), blunt_nodes)
        self.coords = self.nodes[mask_nodes, :]

    def compute_normals(self):
        """
        Computes normal vectors for each filtered element using node coordinates.
        
        Returns:
            normals (numpy array): Array of computed normal vectors.
            edge_points (numpy array): Array of edge point arrays corresponding to each element.
            
        Note:
            This method must be called after filter_blunt_elements().
        """
        if self.filtered_data is None or self.coords is None:
            raise ValueError("filtered_data and coords must be set. Run filter_blunt_elements() first.")

        normals = []
        edge_points = []
        for vec in self.filtered_data:
            mask = np.isin(self.coords[:, 0].astype(int), vec[1:])
            points = self.coords[mask, 1:]
            edge_points.append(points)
            
            # Determine the order of the two points
            if points[0, 0] <= points[1, 0]:
                p1 = points[0]
                p2 = points[1]
            else:
                p1 = points[1]
                p2 = points[0]

            x0, y0 = p1[0], p1[1]
            x1, y1 = p2[0], p2[1]

            # Compute the midpoint for placing the normal vector
            mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2

            # Compute the direction vector of the line
            dx = x1 - x0
            dy = y1 - y0
            norm_val = np.hypot(dx, dy)

            if norm_val != 0:
                if mid_y < 0.0:
                    nx, ny = dy / norm_val, -dx / norm_val
                    if x0 == x1:
                        nx, ny = 1.0, 0.0
                elif mid_y >= 0.0:
                    nx, ny = -dy / norm_val, +dx / norm_val
                    if x0 == x1:
                        nx, ny = 1.0, 0.0
            else:
                nx, ny = 0, 0

            normals.append([nx, ny])
        self.normals = np.array(normals)
        self.edge_points = np.array(edge_points)
        return self.normals, self.edge_points

    def plot_mesh(self):
        """
        Plots the average edge points of the mesh as a scatter plot.
        
        Note:
            Requires that compute_normals() has been called.
        """
        if self.edge_points is None:
            raise ValueError("Edge points not computed. Run compute_normals() first.")
        # Average the edge points for each element (axis=1)
        avg_edge_points = np.mean(self.edge_points, axis=1)
        plt.figure()
        plt.scatter(avg_edge_points[:, 0], avg_edge_points[:, 1], s=2, label="MESH")
        plt.legend()
        plt.show()



