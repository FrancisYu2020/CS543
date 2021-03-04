import numpy as np
import matplotlib.pyplot as plt


def get_wall_z_image(Z_val, fx, fy, cx, cy, szx, szy):
    Z = Z_val*np.ones((szy, szx), dtype=np.float32)
    return Z


def get_road_z_image(H_val, fx, fy, cx, cy, szx, szy):
    y = np.arange(szy).reshape(-1,1)*1.
    y = np.tile(y, (1, szx))
    Z = np.zeros((szy, szx), dtype=np.float32)
    Z[y > cy] = H_val*fy / (y[y>cy]-cy)
    Z[y <= cy] = np.NaN
    return Z


def plot_optical_flow(ax, Z, u, v, cx, cy, szx, szy, s=16):
    # Here is a function for plotting the optical flow. Feel free to modify this
    # function to work well with your inputs, for example if your predictions are
    # in a different coordinate frame, etc.

    x, y = np.meshgrid(np.arange(szx), np.arange(szy))
    ax.imshow(Z, alpha=0.5, origin='upper')
    q = ax.quiver(x[::s,::s], y[::s,::s], u[::s,::s], -v[::s, ::s])
    # ax.quiverkey(q, X=0.5, Y=0.9, U=20,
    #              label='Quiver key length = 20', labelpos='N')
    ax.axvline(cx)
    ax.axhline(cy)
    ax.set_xlim([0, szx])
    ax.set_ylim([szy, 0])
    ax.axis('equal')


if __name__ == "__main__":
    # Focal length along X and Y axis. In class we assumed the same focal length
    # for X and Y axis. but in general they could be different. We are denoting
    # these by fx and fy, and assume that they are the same for the purpose of
    # this MP.
    fx = fy = 128.

    # Size of the image
    szy = 256
    szx = 384

    # Center of the image. We are going to assume that the principal point is at
    # the center of the image.
    cx = 192
    cy = 128

    # Gets the image of a wall 2m in front of the camera.
    Z1 = get_wall_z_image(2., fx, fy, cx, cy, szx, szy)


    # Gets the image of the ground plane that is 3m below the camera.
    Z2 = get_road_z_image(3., fx, fy, cx, cy, szx, szy)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,7))
    ax1.imshow(Z1)
    ax2.imshow(Z2)

    coordinates = np.zeros((szy, szx, 2))
    for y in range(szy):
        for x in range(szx):
            coordinates[y, x, :] = np.array([x - cx, y - cy])*1.0

    # Plotting function.
    #4.2.1
    f = plt.figure(figsize=(13.5,9))
    t_x, t_y, t_z = 0., 0., 1.
    Z = Z2
    u = -t_x/Z + t_z/Z*coordinates[:, :, 0]
    v = -t_y/Z + t_z/Z*coordinates[:, :, 1]
    plot_optical_flow(f.gca(), Z1, u, v, cx, cy, szx, szy, s=16)
    f.savefig('optical_flow_output_1.pdf', bbox_inches='tight')

    #4.2.2
    f = plt.figure(figsize=(13.5,9))
    t_x, t_y, t_z = 1., 0., 0.
    Z = Z1
    u = -t_x/Z + t_z/Z*coordinates[:, :, 0]
    v = -t_y/Z + t_z/Z*coordinates[:, :, 1]
    plot_optical_flow(f.gca(), Z1, u, v, cx, cy, szx, szy, s=16)
    f.savefig('optical_flow_output_2.pdf', bbox_inches='tight')

    #4.2.3
    f = plt.figure(figsize=(13.5,9))
    t_x, t_y, t_z = 0., 0., 1.
    Z = Z1
    u = -t_x/Z + t_z/Z*coordinates[:, :, 0]
    v = -t_y/Z + t_z/Z*coordinates[:, :, 1]
    plot_optical_flow(f.gca(), Z1, u, v, cx, cy, szx, szy, s=16)
    f.savefig('optical_flow_output_3.pdf', bbox_inches='tight')

    #4.2.4
    f = plt.figure(figsize=(13.5,9))
    t_x, t_y, t_z = 1., 1., 1.
    Z = Z1
    u = -t_x/Z + t_z/Z*coordinates[:, :, 0]
    v = -t_y/Z + t_z/Z*coordinates[:, :, 1]
    plot_optical_flow(f.gca(), Z1, u, v, cx, cy, szx, szy, s=16)
    f.savefig('optical_flow_output_4.pdf', bbox_inches='tight')

    #4.2.5
    f = plt.figure(figsize=(13.5,9))
    # t_x, t_y, t_z = 0., 0., 0.
    w_x, w_y, w_z = 0., 1., 0.
    Z = Z1
    # u = -t_x/Z + t_z/Z*coordinates[:, :, 0]
    # v = -t_y/Z + t_z/Z*coordinates[:, :, 1]
    u = -(1 + coordinates[:, :, 0]**2)*w_y
    v = -coordinates[:, :, 0]*coordinates[:, :, 1]*w_y
    plot_optical_flow(f.gca(), Z1, u, v, cx, cy, szx, szy, s=16)
    f.savefig('optical_flow_output_5.pdf', bbox_inches='tight')
