import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "map_output.csv"
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
        return

    pts = df[df['type'] == 'point']
    kfs = df[df['type'] == 'kf']

    fig = plt.figure(figsize=(15, 10))

    # 3D Plot
    ax3d = fig.add_subplot(221, projection='3d')
    ax3d.scatter(pts['x'], pts['y'], pts['z'], c='g', s=1, alpha=0.3, label='Map Points')
    ax3d.plot(kfs['x'], kfs['y'], kfs['z'], c='r', marker='o', markersize=2, label='Trajectory')
    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')
    ax3d.set_title('3D Reconstruction')

    # X-Y Plane (Top-Down)
    ax_xy = fig.add_subplot(222)
    ax_xy.scatter(pts['x'], pts['y'], c='g', s=1, alpha=0.3)
    ax_xy.plot(kfs['x'], kfs['y'], c='r', marker='o', markersize=2)
    ax_xy.set_xlabel('X (m)')
    ax_xy.set_ylabel('Y (m)')
    ax_xy.set_title('Top-Down View (X-Y Plane)')
    ax_xy.grid(True)
    ax_xy.axis('equal')

    # X-Z Plane (Front-Back)
    ax_xz = fig.add_subplot(223)
    ax_xz.scatter(pts['x'], pts['z'], c='g', s=1, alpha=0.3)
    ax_xz.plot(kfs['x'], kfs['z'], c='r', marker='o', markersize=2)
    ax_xz.set_xlabel('X (m)')
    ax_xz.set_ylabel('Z (m)')
    ax_xz.set_title('Side View (X-Z Plane)')
    ax_xz.grid(True)
    ax_xz.axis('equal')

    # Y-Z Plane
    ax_yz = fig.add_subplot(224)
    ax_yz.scatter(pts['y'], pts['z'], c='g', s=1, alpha=0.3)
    ax_yz.plot(kfs['y'], kfs['z'], c='r', marker='o', markersize=2)
    ax_yz.set_xlabel('Y (m)')
    ax_yz.set_ylabel('Z (m)')
    ax_yz.set_title('Side View (Y-Z Plane)')
    ax_yz.grid(True)
    ax_yz.axis('equal')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
