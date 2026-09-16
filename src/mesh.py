import open3d as o3d
import numpy as np

def points_to_subdivided_stl(pts, output_path, depth=7, smooth_iter=5):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

   
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20)
    )

    pts_arr = np.asarray(pcd.points)
    normals_arr = np.asarray(pcd.normals)
    center = np.mean(pts_arr, axis=0)
    
    outward_mask = np.sum((pts_arr - center) * normals_arr, axis=1) < 0
    normals_arr[outward_mask] = -normals_arr[outward_mask]
    pcd.normals = o3d.utility.Vector3dVector(normals_arr)

    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, 
        depth=depth, 
        n_threads=1
    )

   
    mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iter)
    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(output_path, mesh)
    return output_path