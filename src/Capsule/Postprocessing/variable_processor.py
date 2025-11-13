import vtk
import numpy as np
import os
import sys
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

class VarProcessor:
	def __init__(self, input_vtu_file, index,mu=1.0):
		"""
		Initialize the VTUProcessor.
		
		Parameters:
			mu (float): The dynamic viscosity to use in the shear stress calculation.
		"""
		self.mu = mu
		self.input_vtu_file = input_vtu_file

		print("input_vtu_file=\t",input_vtu_file)

		reader = vtk.vtkXMLUnstructuredGridReader()
		reader.SetFileName(self.input_vtu_file)
		reader.Update()
		self.grid = reader.GetOutput()


	def get_vtu_files(self, directory):
		"""Recursively search for all .vtu files in the given directory."""
		vtu_files = []
		for root, dirs, files in os.walk(directory):
			for file in files:
				# print("file=\t",file)
				if file.lower().endswith(".vtu"):
					vtu_files.append(os.path.join(root, file))
		return vtu_files

	def add_temperature_field(self, grid):
		"""Compute the temperature field and add it to the grid's point data."""
		# pointData = grid.GetPointData()

		# # Retrieve the required arrays
		# rho_vtk = pointData.GetArray("rho")
		# p_vtk   = pointData.GetArray("p")
		# v1_vtk  = pointData.GetArray("v1")
		# v2_vtk  = pointData.GetArray("v2")

		# rho_pred_vtk = pointData.GetArray("rho_pred")
		# p_pred_vtk   = pointData.GetArray("p_pred")
		# v1_pred_vtk  = pointData.GetArray("u_pred")
		# v2_pred_vtk  = pointData.GetArray("v_pred")

		# if None in (rho_vtk, p_vtk, v1_vtk, v2_vtk):
		# 	raise ValueError("One or more required arrays ('rho', 'p', 'v1', 'v2') are missing.")

		# # Convert VTK arrays to NumPy arrays
		# rho = vtk_to_numpy(rho_vtk)
		# p   = vtk_to_numpy(p_vtk)
		# v1  = vtk_to_numpy(v1_vtk)
		# v2  = vtk_to_numpy(v2_vtk)

		# rho_pred = vtk_to_numpy(rho_pred_vtk)
		# p_pred   = vtk_to_numpy(p_pred_vtk)
		# v1_pred  = vtk_to_numpy(v1_pred_vtk)
		# v2_pred  = vtk_to_numpy(v2_pred_vtk)

		# # Compute temperature T with protection against division by zero.
		# # The formula is: T = p * 1.4 * 850 / (rho * 0.014103202 * 287.058)
		# T = np.where(rho == 0, 0.0, p * 1.4 * 850 / (rho * 0.014103202 * 287.058))
		# T_pred = np.where(rho_pred == 0, 0.0, p_pred * 1.4 * 850 / (rho_pred * 0.014103202 * 287.058))
		# # Convert temperature array back to a VTK array and add to point data.
		# T_vtk = numpy_to_vtk(T, deep=True, array_type=vtk.VTK_DOUBLE)
		# T_vtk.SetName("T")
		# pointData.AddArray(T_vtk)

		# T_pred_vtk = numpy_to_vtk(T_pred, deep=True, array_type=vtk.VTK_DOUBLE)
		# T_pred_vtk.SetName("T_pred")
		# pointData.AddArray(T_pred_vtk)
		return grid

	# def add_velocity_field(self, grid):
	# 	"""Create a velocity vector field from v1 and v2 and add it to the grid."""
	# 	pointData = grid.GetPointData()

	# 	# Retrieve v1 and v2 arrays
	# 	v1_vtk = pointData.GetArray("v1")
	# 	v2_vtk = pointData.GetArray("v2")

	# 	v1_pred_vtk = pointData.GetArray("u_pred")
	# 	v2_pred_vtk = pointData.GetArray("v_pred")
	# 	if None in (v1_vtk, v2_vtk):
	# 		raise ValueError("One or more required arrays ('v1', 'v2') are missing for velocity computation.")

	# 	v1 = vtk_to_numpy(v1_vtk)
	# 	v2 = vtk_to_numpy(v2_vtk)

	# 	v1_pred = vtk_to_numpy(v1_pred_vtk)
	# 	v2_pred = vtk_to_numpy(v2_pred_vtk)
	# 	num_points = grid.GetNumberOfPoints()

	# 	# Create a velocity array with a zero z-component.
	# 	velocity = np.column_stack((v1, v2, np.zeros(num_points)))
	# 	velocity_vtk = numpy_to_vtk(velocity, deep=True, array_type=vtk.VTK_DOUBLE)
	# 	velocity_vtk.SetNumberOfComponents(3)
	# 	velocity_vtk.SetName("velocity")
	# 	pointData.AddArray(velocity_vtk)

	# 	velocity_pred = np.column_stack((v1_pred, v2_pred, np.zeros(num_points)))
	# 	velocity_pred_vtk = numpy_to_vtk(velocity_pred, deep=True, array_type=vtk.VTK_DOUBLE)
	# 	velocity_pred_vtk.SetNumberOfComponents(3)
	# 	velocity_pred_vtk.SetName("velocity_pred")
	# 	pointData.AddArray(velocity_pred_vtk)
	# 	return grid

	def compute_gradients(self, grid):
		"""Compute the gradients of the temperature and velocity fields."""
		# Compute gradient for temperature (T)
		gradT_filter = vtk.vtkGradientFilter()
		gradT_filter.SetInputData(grid)
		gradT_filter.SetResultArrayName("gradT")
		gradT_filter.SetInputArrayToProcess(0, 0, 0,
		    vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "T")
		gradT_filter.Update()
		grid_T = gradT_filter.GetOutput()

		# Compute gradient for first velocity field (v1)
		gradV_filter = vtk.vtkGradientFilter()
		gradV_filter.SetInputData(grid_T)  # using grid_T which now has gradT arrays
		gradV_filter.SetResultArrayName("gradu")
		gradV_filter.SetInputArrayToProcess(0, 0, 0,
		    vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "v1")
		gradV_filter.Update()
		grid_V = gradV_filter.GetOutput()

		# Compute gradient for second velocity field (v2) using a new filter instance
		gradV_filter = vtk.vtkGradientFilter()
		gradV_filter.SetInputData(grid_V)  # or use grid_T if they share the same point data
		gradV_filter.SetResultArrayName("gradv")
		gradV_filter.SetInputArrayToProcess(0, 0, 0,
		    vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "v2")
		gradV_filter.Update()
		grid_V = gradV_filter.GetOutput()

		# # Merge the velocity gradient arrays into the temperature grid
		pd_T = grid_T.GetPointData()
		pd_V = grid_V.GetPointData()

		# # Retrieve the velocity gradient arrays from grid_V
		gradu_array = pd_V.GetArray("gradu")
		gradv_array = pd_V.GetArray("gradv")

		# # Add the velocity gradients to grid_T's point data

		pd_T.AddArray(gradu_array)
		pd_T.AddArray(gradv_array)

		# Now, grid_T contains "gradT", "gradu", and "gradv"
		return grid_T



	def compute_shear_stress(self, grid,suffix):
		pd = grid.GetPointData()
    
		# Retrieve the gradient arrays for u and v
		gradu = pd.GetArray("gradu"+suffix)
		gradv = pd.GetArray("gradv"+suffix)

		if gradu is None or gradv is None:
		    raise ValueError("Gradient arrays 'gradu' and/or 'gradv' not found in the grid.")

		# Convert the VTK arrays to NumPy arrays for vectorized operations.
		np_gradu = vtk_to_numpy(gradu)
		np_gradv = vtk_to_numpy(gradv)

		du_dy = np_gradu[:, 1]
		dv_dx = np_gradv[:, 0]

		# Compute the shear stress component using the given dynamic viscosity.
		tau_xy_np = self.mu * (du_dy + dv_dx)

		# Convert the NumPy array back to a VTK array.
		vtk_tau_xy = numpy_to_vtk(tau_xy_np, deep=True)
		vtk_tau_xy.SetName("tau_xy"+suffix)

		# Compute the normal stresses using:
		# tau_xx = 2 * mu * (du/dx)    => first component of gradu
		# tau_yy = 2 * mu * (dv/dy)    => second component of gradv
		tau_xx_np = 2 * self.mu * np_gradu[:, 0]
		tau_yy_np = 2 * self.mu * np_gradv[:, 1]

		# Convert the NumPy arrays back to VTK arrays.
		vtk_tau_xx = numpy_to_vtk(tau_xx_np, deep=True)
		vtk_tau_xx.SetName("tau_xx"+suffix)

		vtk_tau_yy = numpy_to_vtk(tau_yy_np, deep=True)
		vtk_tau_yy.SetName("tau_yy"+suffix)

		# Add the computed normal stress arrays to the grid's point data.
		pd.AddArray(vtk_tau_xx)
		pd.AddArray(vtk_tau_yy)
		pd.AddArray(vtk_tau_xy)

		return grid

	def write_output(self, grid, output_filename):
		"""Write the grid (with all computed fields) to the output VTU file."""
		writer = vtk.vtkXMLUnstructuredGridWriter()
		writer.SetFileName(output_filename)
		writer.SetInputData(grid)
		writer.Write()

	def process(self,output_filename):
		"""
		Process the input VTU file to compute temperature, velocity, gradients,
		and shear stress fields, then write the modified grid to the output file.
		"""
		# grid = self.read_input(input_filename)
		# grid = self.add_temperature_field(self.grid)
		# grid = self.add_velocity_field(grid)
		grid = self.compute_gradients(self.grid)
		# grid = self.compute_shear_stress(grid,"")
		# grid = self.compute_shear_stress(grid,"_pred")
		self.write_output(grid, output_filename)

# if __name__ == '__main__':
#     if len(sys.argv) < 3:
#         print("Usage: python script.py <input.vtu> <output.vtu>")
#         sys.exit(1)

#     input_file = sys.argv[1]
#     output_file = sys.argv[2]
#     processor = VarProcessor()
#     processor.process(input_file, output_file)

