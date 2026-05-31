import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.cm as cm

# caricamento dati generati dal C++
print("Caricamento dati...")
df = pd.read_csv('traiettorie_sequenziali.csv')
frames_count = df['frame'].nunique()
n_agents = df['agent_id'].nunique()
W, H = 20, 20 #dimensioni dell'area

# caricamento matrice di adiacenza
A = np.loadtxt('adiacenza.csv', delimiter=',')

print(f"Dati caricati: {n_agents} agenti, {frames_count} frame.")

# generazione colori casuali per riempire le celle
colors_fill = cm.get_cmap('Set3', n_agents)(np.linspace(0, 1, n_agents))
# trasparenza = 0.3
colors_fill[:, 3] = 0.3

# setup grafico
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect('equal')

# funzione geometrica
def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Function only supports 2D")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue
        ridges = all_ridges.get(p1, [])
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0 or v1 < 0:
                v_finite = v1 if v1 >= 0 else v2
                finite_vertex = vor.vertices[v_finite]
                tangent = vor.points[p2] - vor.points[p1]
                tangent /= np.linalg.norm(tangent)
                normal = np.array([-tangent[1], tangent[0]])
                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, normal)) * normal
                far_point = finite_vertex + direction * radius
                new_vertices.append(far_point.tolist())
                new_region.append(len(new_vertices) - 1)
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)].tolist()
        new_regions.append(new_region)
    return new_regions, np.asarray(new_vertices)

# animazione
def update(frame):
    ax.clear()
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.set_title(f"Visualizzazione - Frame {frame}")

    # estrazione posizioni agenti per il frame corrente
    frame_data = df[df['frame'] == frame]
    agents = frame_data[['x', 'y']].values
    agent_ids = frame_data['agent_id'].values

    bbox = box(0, 0, W, H)

    # calcolo geometrico di Voronoi
    try:
        vor = Voronoi(agents)
        regions, vertices = voronoi_finite_polygons_2d(vor)

        # disegno i poligoni
        for i, region in enumerate(regions):
            polygon_coords = vertices[region]
            poly = Polygon(polygon_coords)

            clipped = poly.intersection(bbox)

            if not clipped.is_empty and clipped.geom_type == "Polygon":

                current_agent_id = agent_ids[i]

                patch = MplPolygon(
                    list(clipped.exterior.coords),
                    facecolor=colors_fill[current_agent_id],
                    edgecolor='forestgreen',
                    linewidth=2.0,
                    alpha=1.0,
                    zorder=1
                )
                ax.add_patch(patch)
    except Exception as e:
        #in caso di posizioni collidenti e fallimento di Scipy
        print(f"Errore Voronoi al frame {frame}: {e}")

    # disegno linee di connettività
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            if A[i, j] == 1:
                # posizioni attuali dei due agenti connessi
                pos_i = agents[df[df['frame']==frame]['agent_id']==i][0]
                pos_j = agents[df[df['frame']==frame]['agent_id']==j][0]
                ax.plot([pos_i[0], pos_j[0]],
                        [pos_i[1], pos_j[1]],
                        c='magenta', lw=0.8, alpha=0.8, zorder=2)

    # disegno agenti e scritta ID
    ax.scatter(agents[:, 0], agents[:, 1], c='red', s=60, edgecolors='black', zorder=3)
    for idx, (x, y) in enumerate(agents):
        real_id = agent_ids[idx]
        ax.text(x + 0.2, y + 0.2, str(real_id), fontsize=9,
                color='blue', weight='bold', zorder=4)

    info_text = f"Frame: {frame}/{frames_count-1}\n Agenti: {n_agents}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, va='top', ha='left', weight='bold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round'))

print("Avvio animazione...")
ani = FuncAnimation(fig, update, frames=frames_count,
                    interval=80, blit=False, repeat=False)

print("Generazione e salvataggio della GIF in corso...")
ani.save('simulazione_voronoi.gif', writer='pillow', fps=12)
print("GIF salvata con successo come 'simulazione_voronoi.gif'!")

plt.tight_layout()
plt.show()