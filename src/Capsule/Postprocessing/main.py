from vtu_processor import VTUProcessor
from mesh_processor import MeshProcessor
from variable_processor import VarProcessor
import os
import matplotlib.pyplot as plt
import numpy as np

import numpy as np

def compute_heat_flux(coordinates, normals, gradT, k=1.0):
	# Compute the local heat flux at each point on every segment.
	# local_flux has shape (96, 4)
	local_flux = -k * np.sum(gradT * normals, axis=2)
	
	# Compute the differential arc lengths (ds) between consecutive points along each segment.
	# diff_coords has shape (96, 3, 2)
	diff_coords = np.diff(coordinates, axis=1)
	ds = np.linalg.norm(diff_coords, axis=2)  # shape (96, 3)
	
	# Create an integration variable s (cumulative arc length) for each segment.
	# Start with 0 for the first point.
	s = np.concatenate([np.zeros((coordinates.shape[0], 1)), np.cumsum(ds, axis=1)], axis=1)
	
	# Now integrate local_flux over s using the trapezoidal rule.
	# This gives one integrated flux value per segment (shape (96,))
	integrated_flux = np.trapezoid(local_flux, s, axis=1)
	
	return np.sum(integrated_flux)

def construct_linear_line(constructors, point):
	# Unpack coordinates for the two endpoints
	x0, y0,_ = constructors[0, :]
	x1, y1,_ = constructors[1, :]
	x, y,_ = point

	# Handle vertical line case
	if x1 - x0 == 0.0:
		# Check if the point lies along the vertical line segment
		if x == x0 and y >= min(y0, y1) and y < max(y0, y1):
			res = 0.0
		else:
			res = 1e4  # Large penalty value
	else:
		slope = (y1 - y0) / (x1 - x0)
		res = np.abs(y - y0 - slope * (x - x0))
	return res

def vectorized_linear_line(constructors, points):
	# Unpack endpoints
	x0, y0,_ = constructors[0]
	x1, y1,_ = constructors[1]
	
	# Difference in x for the line segment
	dx = x1 - x0
	
	# Initialize result array
	if dx == 0:  # vertical line
		# Define vertical line boundaries
		y_min = min(y0, y1)
		y_max = max(y0, y1)
		# Create a boolean mask for points that lie on the vertical line segment
		on_line = (points[:, 0] == x0) & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
		# Initialize all values with a large penalty (e.g., 1e4)
		res = np.full(points.shape[0], 1e4)
		# Set residual to zero for points that lie exactly on the line segment
		res[on_line] = 0.0
	else:
		# Compute the slope of the line
		slope = (y1 - y0) / dx
		# Compute the residual for each point as the absolute difference
		# between the actual y and the y on the line
		res = np.abs(points[:, 1] - y0 - slope * (points[:, 0] - x0))
	
	return res

def points_on_line_segment_np(A, B, P, tol=1e-8):
	# Ensure P is at least 2D in case a single point is passed
	P = np.atleast_2d(P)
	
	# Compute vector from A to B and from A to each point in P
	AB = B - A         # Vector from A to B, shape (2,)
	AP = P - A         # Vectors from A to each point, shape (N,2)
	
	# Compute the 2D cross product (determinant) for each point.
	# For 2D vectors, cross product reduces to a scalar: AB[0]*AP[:,1] - AB[1]*AP[:,0]
	cross = np.abs(AB[0] * AP[:, 1] - AB[1] * AP[:, 0])
	
	# Check collinearity: points must have a near-zero cross product.
	collinear = cross <= tol
	
	# Compute the dot product for each point (projection of AP onto AB)
	dot = np.sum(AP * AB, axis=1)
	
	# Compute the squared length of AB
	AB_sq = np.dot(AB, AB)
	
	# Check that the projection lies between 0 and AB_sq, meaning between A and B.
	between = (dot >= 0) & (dot <= AB_sq)
	
	# A point is on the segment if it's collinear and lies between A and B.
	return collinear & between


if __name__ == "__main__":

	# Example usage:
	index_test = [23, 30, 46,  2, 27, 17, 51, 16, 21, 49,  0,  4]
	# index_test = [34]
	errors =[]
	pred_deriv=[]
	true_deriv=[]
	boundary_nodes=[]
	heat_flux_vec = []
	heat_flux_predicted = []
	for index in index_test:
	# index = index_test[2]
		params = np.loadtxt("params.txt")
		rhoInf = 0.014103202
		mach = params[index,-1]
		p_freestream =  850
		Pr = 0.72
		cInf = np.sqrt(1.4 * p_freestream / rhoInf)
		# mach = sqrt(1.4*p/rho)
		uInf = mach*cInf
		print("mach=\t",mach)
		Re=0.04 * 10**6
		R = 287.058
		cp = 1.4*287.058/(1.4-1)
		mu = rhoInf*uInf/Re
		kappa = cp*mu/Pr
		print("kappa=\t",kappa)

		meshname = "mesh_rotated" + str(index) + ".inp"
		mesh_path = "../mesh_folder/"
		mesh_file = os.path.join(mesh_path, meshname)

		# Create an instance of MeshProcessor
		meshprocessor = MeshProcessor(mesh_file)

		# Filter blunt elements and corresponding node coordinates
		meshprocessor.filter_blunt_elements(blunt_key="Blunt", line_identifier="Line")

		# Compute normals and edge points
		normals, edge_points = meshprocessor.compute_normals()
		print("Blunt Normals shape:", normals.shape)
		print("Blunt Edge points shape:", edge_points.shape)
		# meshprocessor.plot_mesh()
		# meshprocessor.filter_blunt_elements(blunt_key="Duct", line_identifier="Line")

		# # Compute normals and edge points
		# normals_duct, edge_points_duct = meshprocessor.compute_normals()
		# print("Duct Normals shape:", normals_duct.shape)
		# print("Duct Edge points shape:", edge_points_duct.shape)

		# meshprocessor.plot_mesh()
		#vtu_files/
		direc = "../Inference/"
		
		output_filename = "case_"+str(index)+".vtu"
		variableprocessor=VarProcessor(input_vtu_file=direc+output_filename, index=index,mu=mu)
		variableprocessor.process(output_filename)

		processor = VTUProcessor(input_vtu_file=output_filename,variable="gradT")
		processor_pred = VTUProcessor(input_vtu_file=output_filename,variable="gradT_pred")

		# Process boundaries.
		points, solution = processor.process_boundaries(edge_points)
		print("points=\t",points.shape)
		print("solution=\t",solution.shape)
		points,indices=np.unique(points, axis=0, return_index=True)
		solution = solution[indices,:]
		print("pointsunique=\t",points.shape)
		print("solutionunique=\t",solution.shape)
		points, solution_pred = processor_pred.process_boundaries(edge_points)
		print("points=\t",points.shape)
		print("solution_pred=\t",solution.shape)
		points,indices=np.unique(points, axis=0, return_index=True)
		solution_pred= solution_pred[indices,:]
		print("pointsunique=\t",points.shape)
		print("solution_predunique=\t",solution_pred.shape)

		boundary_elements_nodes = points[:,:,0:2] 
		boundary_elements_normals = np.zeros((edge_points.shape[0],points.shape[1],2))
		boundary_elements_sol=solution[:,:,0:2]
		boundary_elements_sol_pred=solution_pred[:,:,0:2]

		for i in range(edge_points.shape[0]):
			for j in range(points.shape[1]):
				boundary_elements_normals[i,j,:] = normals[i,0:2]
		
	# 	error =np.abs(boundary_elements_sol-boundary_elements_sol_pred)/np.abs(boundary_elements_sol)
	# 	print("error in temperature gradient=\t",np.mean(error,axis=(0,1)),"\t",np.min(error,axis=(0,1)),"\t",np.max(error,axis=(0,1)))

		boundary_elements_nodes = boundary_elements_nodes[:,:,:]
		boundary_elements_normals = boundary_elements_normals[:,:,:]
		boundary_elements_sol = boundary_elements_sol[:,:,:]
		boundary_elements_sol_pred = boundary_elements_sol_pred[:,:,:]

		print("boundary_elements_nodes=\t",boundary_elements_nodes.shape)

		heat_flux = compute_heat_flux(boundary_elements_nodes, boundary_elements_normals, boundary_elements_sol, kappa)
		heat_flux_pred = compute_heat_flux(boundary_elements_nodes, boundary_elements_normals, boundary_elements_sol_pred, kappa)
		print("heat_flux_truth=\t",heat_flux)
		print("heat_flux_pred=\t",heat_flux_pred)
		heat_flux_vec.append(heat_flux)
		heat_flux_predicted.append(heat_flux_pred)
		print("error_precentage=\t",np.abs(heat_flux-heat_flux_pred)*100/np.abs(heat_flux),"\tindex=\t",index)
		errors.append(np.abs(heat_flux-heat_flux_pred)*100/np.abs(heat_flux))
		pred_deriv.append(boundary_elements_sol_pred)
		true_deriv.append(boundary_elements_sol)
		boundary_nodes.append(boundary_elements_nodes)
		# val_err=np.abs(boundary_elements_sol-boundary_elements_sol_pred)
		# print("value_error=\t",np.min(val_err),"\t",np.max(val_err))
		# value_err.append(val_err.flatten())
		# print("heat_flux_pred=\t",heat_flux_pred)
		# print("error_precentage=\t",np.abs(heat_flux-heat_flux_pred)*100/np.abs(heat_flux),"\tindex=\t",index)

		# pointsed = np.mean(edge_points,axis=1)[:,0:2]

		# x = pointsed[:, 0]
		# y = pointsed[:, 1]

		# # Extract the u and v components of the normal vectors
		# u = normals[:, 0]
		# v = normals[:, 1]

		# plt.figure()
		# plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1, color='r')
		# plt.scatter(x, y, color='b')  # Optional: plot the points as well
		# plt.xlabel("X")
		# plt.ylabel("Y")
		# plt.title("Normal Vectors at Points")
		# plt.grid(True)
		# plt.xlim(min(x) - 1, max(x) + 1)
		# plt.ylim(min(y) - 1, max(y) + 1)
		# plt.show()
	
	

	errors=np.array(errors)
	index_test=np.array(index_test)
	pred_deriv=np.array(pred_deriv)
	true_deriv=np.array(true_deriv)
	boundary_nodes=np.array(boundary_nodes)
	heat_flux_vec=np.array(heat_flux_vec)
	heat_flux_predicted=np.array(heat_flux_predicted)

	for i in range(boundary_nodes.shape[0]):
		x = boundary_nodes[i,:,:,0]
		y = boundary_nodes[i,:,:,1]

		plt.scatter(x,y,s=8,c='r' ,label="Mesh")
		plt.legend()
		plt.savefig("boundary"+str(i)+".png",dpi=300)
		plt.close()

	np.savetxt("eroors.txt",errors)
	np.savetxt("index_test.txt",index_test)

	np.savetxt("heat_flux.txt",heat_flux_vec)
	np.savetxt("heat_flux_predicted.txt",heat_flux_predicted)
	
	np.savez("data.npz",pred=pred_deriv,true=true_deriv,coords=boundary_nodes)
	# np.savetxt("pred_deriv.txt",pred_deriv)


