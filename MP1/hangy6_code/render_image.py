import numpy as np
from generate_scene import get_ball
import matplotlib.pyplot as plt

# specular exponent
k_e = 50

def render(Z, N, A, S,
           point_light_loc, point_light_strength,
           directional_light_dirn, directional_light_strength,
           ambient_light, k_e):
  # To render the images you will need the camera parameters, you can assume
  # the following parameters. (cx, cy) denote the center of the image (point
  # where the optical axis intersects with the image, f is the focal length.
  # These parameters along with the depth image will be useful for you to
  # estimate the 3D points on the surface of the sphere for computing the
  # angles between the different directions.
  h, w = A.shape
  cx, cy = w / 2, h /2
  f = 128.

  #Get the camera coordinates of all the points in the pixel coordinates
  coordinates = np.zeros((h, w, 3))
  for y in range(h):
      for x in range(w):
          depth = Z[y, x]
          coord = [(x - cx)*depth/f, (y - cy)*depth/f, depth]
          coordinates[y, x, :] = np.array(coord)

  #Normalize the directional light
  directional_light_dirn = np.array(directional_light_dirn)/np.linalg.norm(np.array(directional_light_dirn))


  # Ambient Term
  I = A * ambient_light

  # Diffuse Term
  #Get the direction vectors of point light and normalize
  point_light_dirns = np.array(point_light_loc) - coordinates
  point_light_dirns = (point_light_dirns.T/np.linalg.norm(point_light_dirns, axis=2).T).T

  #Dot product of direction vectors with the surface normal vectors
  point_diffuse = np.sum(point_light_dirns*N, axis=2)
  directional_diffuse = np.sum(directional_light_dirn*N, axis=2)
  #Filter out the values that are negative in the diffuse matrix
  filter = point_diffuse < 0
  point_diffuse[filter] = 0
  filter = directional_diffuse < 0
  directional_diffuse[filter] = 0
  I += A*(point_diffuse*np.array(point_light_strength) + directional_diffuse*np.array(directional_light_strength))

  # Specular Term
  #Get the viewing directions and normalize
  v_r = np.zeros(3) - coordinates
  v_r = (v_r.T/np.linalg.norm(v_r, axis=2).T).T

  #Reflection = (2x[dot]N)N - x, x is the reverse of the incident light
  #No need to normalize again because the norm of reflection vectors are already 1
  point_reflection = (2*np.sum(point_light_dirns*N, axis=2).T*N.T).T - point_light_dirns
  directional_reflection = (2*np.sum(directional_light_dirn*N, axis=2).T*N.T).T - directional_light_dirn

  #Get the specular matrices
  point_specular = np.sum(point_reflection*v_r, axis=2)
  directional_specular = np.sum(directional_reflection*v_r, axis=2)
  #Filter out the negative values in the specular matrices
  filter = point_specular < 0
  point_specular[filter] = 0
  filter = directional_specular < 0
  directional_specular[filter] = 0
  I += S*(np.array(point_light_strength)*(point_specular)**k_e + np.array(directional_light_strength)*(directional_specular)**k_e)

  I = np.minimum(I, 1)*255
  I = I.astype(np.uint8)
  I = np.repeat(I[:,:,np.newaxis], 3, axis=2)
  return I

def main():
  for specular in [True, False]:
    # get_ball function returns:
    # - Z (depth image: distance to scene point from camera center, along the
    # Z-axis)
    # - N is the per pixel surface normals (N[:,:,0] component along X-axis
    # (pointing right), N[:,:,1] component along X-axis (pointing down),
    # N[:,:,0] component along X-axis (pointing into the scene)),
    # - A is the per pixel ambient and diffuse reflection coefficient per pixel,
    # - S is the per pixel specular reflection coefficient.
    Z, N, A, S = get_ball(specular=specular)

    # Strength of the ambient light.
    ambient_light = 0.5

    # For the following code, you can assume that the point sources are located
    # at point_light_loc and have a strength of point_light_strength. For the
    # directional light sources, you can assume that the light is coming _from_
    # the direction indicated by directional_light_dirn, and with strength
    # directional_light_strength. The coordinate frame is centered at the
    # camera, X axis points to the right, Y-axis point down, and Z-axis points
    # into the scene.

    # Case I: No directional light, only point light source that moves around
    # the object.
    point_light_strength = [1.5]
    directional_light_dirn = [[1, 0, 0]]
    directional_light_strength = [0.0]

    fig, axes = plt.subplots(4, 4, figsize=(15,10))
    axes = axes.ravel()[::-1].tolist()
    for theta in np.linspace(0, np.pi*2, 16):
      point_light_loc = [[10*np.cos(theta), 10*np.sin(theta), -3]]
      I = render(Z, N, A, S, point_light_loc, point_light_strength,
                 directional_light_dirn, directional_light_strength,
                 ambient_light, k_e)
      ax = axes.pop()
      ax.imshow(I)
      ax.set_axis_off()
    plt.savefig(f'specular{specular:d}_move_point.png', bbox_inches='tight')
    plt.close()

    # Case II: No point source, just a directional light source that moves
    # around the object.
    point_light_loc = [[0, -10, 2]]
    point_light_strength = [0.0]
    directional_light_strength = [2.5]

    fig, axes = plt.subplots(4, 4, figsize=(15,10))
    axes = axes.ravel()[::-1].tolist()
    for theta in np.linspace(0, np.pi*2, 16):
      directional_light_dirn = [np.array([np.cos(theta), np.sin(theta), .1])]
      directional_light_dirn[0] = \
        directional_light_dirn[0] / np.linalg.norm(directional_light_dirn[0])
      I = render(Z, N, A, S, point_light_loc, point_light_strength,
                 directional_light_dirn, directional_light_strength,
                 ambient_light, k_e)
      ax = axes.pop()
      ax.imshow(I)
      ax.set_axis_off()
    plt.savefig(f'specular{specular:d}_move_direction.png', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
  main()
