import numpy as np
import os
import pyvista as pv
from pyvista import examples
import pymeshfix



def makeMeshes(data_dir,N_meshes,noise_level=0.0001, deformation_scale=0.1):
    data_dir=data_dir+"/raw"
   
    base_mesh =examples.download_bunny()

    v = base_mesh.points
    f = base_mesh.faces.reshape(-1, 4)[:, 1:]

    # Fix the mesh using pymeshfix
    meshfix = pymeshfix.MeshFix(v, f)
    meshfix.repair()

    # Create a PyVista mesh from the repaired vertices and faces
    base_mesh = pv.PolyData(meshfix.v, np.hstack((np.full((meshfix.f.shape[0], 1), 3), meshfix.f)))


    for i in range(N_meshes):
        # Copy the base mesh to avoid modifying the original
        new_mesh = base_mesh.copy()

        # Apply random rotation
        angle_x, angle_y, angle_z = np.random.uniform(0, 360, 3)
        new_mesh.rotate_x(angle_x)
        new_mesh.rotate_y(angle_y)
        new_mesh.rotate_z(angle_z)

        # Apply random deformation by scaling each axis with factors slightly away from 1
        points = new_mesh.points
        scale_factors = np.random.uniform(1 - deformation_scale, 1 + deformation_scale, 3)
        points[:, 0] *= scale_factors[0]  # Scale x-axis
        points[:, 1] *= scale_factors[1]  # Scale y-axis
        points[:, 2] *= scale_factors[2]  # Scale z-axis
        new_mesh.points = points

        # Add noise
        noise = np.random.normal(0, noise_level, points.shape)
        new_mesh.points += noise

        # Save the new mesh
        file_path = os.path.join(data_dir, f"mesh{i}.vtp")
        new_mesh.save(file_path)
        new_mesh.plot()
        print(f"Saved: {file_path}")


#TODO: remove this later
#TODO: specify enviromnet.yml
if __name__ == "__main__":
    from micro_match.pipeline import run_micromatch_test, run_microMatch,shape_analysis

    #run_micromatch_test()

    example_data_dir = "/home/max/Documents/02_Data/random/test_muMatchBC"

    # makeMeshes(example_data_dir,3)
    
    run_microMatch(
        data_dir=str(example_data_dir),
        mesh_pairs=[("20240421mAG-zGemH2a-mcherry78hpfLM6nucleisurface", "20240422mAG-zGemH2a-mcherry96hpfLM5nucleisurface"), ("20240421mAG-zGemH2a-mcherry78hpfLM6nucleisurface", "20240428mAG-zGemH2a-mcherry144hpfControl4nucleisurface")],
        dataset_id="bunny",
        use_deep_learning=True
    )

    # data_dir = os.path.join(example_data_dir, "processed_data")
    # match_dir = os.path.join(data_dir, "match_results")
    # shape_analysis(
    #         data_dir=data_dir, match_dir=match_dir, dataset_id="bunny"
    # )

    