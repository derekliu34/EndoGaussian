import plyfile
import numpy as np

# Load original point cloud
plydata = plyfile.PlyData.read('point_cloud.ply')
vertex_data = plydata['vertex'].data
field_names = vertex_data.dtype.names

print(f"Detected PLY fields: {field_names}")

# 1. Extract Positions (x, y, z)
x = vertex_data['x'].astype(np.float32)
y = vertex_data['y'].astype(np.float32)
z = vertex_data['z'].astype(np.float32)

# 2. Automatically find and extract color channels
if 'red' in field_names:
    r = vertex_data['red']
    g = vertex_data['green']
    b = vertex_data['blue']
elif 'f_dc_0' in field_names:
    # 3D Gaussian Splatting Spherical Harmonics format (f_dc_0, f_dc_1, f_dc_2)
    # SH degree 0 to RGB formula: RGB = 0.5 + C0 * SH
    C0 = 0.28209479177387814
    r = np.clip((0.5 + C0 * vertex_data['f_dc_0']) * 255, 0, 255)
    g = np.clip((0.5 + C0 * vertex_data['f_dc_1']) * 255, 0, 255)
    b = np.clip((0.5 + C0 * vertex_data['f_dc_2']) * 255, 0, 255)
elif 'diffuse_red' in field_names:
    r = vertex_data['diffuse_red']
    g = vertex_data['diffuse_green']
    b = vertex_data['diffuse_blue']
else:
    # Fallback: search for any fields containing 'r', 'g', 'b' or 'color'
    color_fields = [f for f in field_names if any(c in f.lower() for c in ['red', 'green', 'blue', 'color', 'f_dc'])]
    raise KeyError(f"Could not automatically map color fields. Available fields are: {field_names}")

# Ensure uint8 [0, 255]
r = r.astype(np.uint8)
g = g.astype(np.uint8)
b = b.astype(np.uint8)

# 3. Construct clean Blender-compatible structure
vertices = np.empty(len(x), dtype=[
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
])

vertices['x'] = x
vertices['y'] = y
vertices['z'] = z
vertices['red'] = r
vertices['green'] = g
vertices['blue'] = b

# 4. Save
el = plyfile.PlyElement.describe(vertices, 'vertex')
plyfile.PlyData([el], text=False).write('blender_ready.ply')