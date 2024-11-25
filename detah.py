from flask import Flask, render_template, request
from random import random
app = Flask(__name__)

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401
from matplotlib import ticker
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from PIL import Image

from sklearn import datasets, manifold

n_samples = 400
S_points, S_color = datasets.make_s_curve(n_samples, random_state=0)

boeing_airflow = np.loadtxt("boeing.csv", delimiter=",")

datasets = {
    "artificial": {
        "real_data":S_points,
        "data": S_points,
        "colors": S_color
    },
    "boeing_airflow": {
        "real_data":boeing_airflow,
        "data":boeing_airflow[:,:3],
        "colors":boeing_airflow[:,3:]
    }
}
def plot_3d(points, points_color, title):
    x, y, z = points.T

    fig, ax = plt.subplots(
        figsize=(10, 10),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = None
    if points_color is not None:
        col = ax.scatter(x, y, z, c=points_color, s=50, alpha=0.8)
    else:
        col = ax.scatter(x, y, z, s=50, alpha=0.8)
    canvas = FigureCanvasAgg(fig)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    canvas.draw()
    buf = canvas.buffer_rgba()
    X = np.asarray(buf)
    im = Image.fromarray(X)
    n = int(random()*20000)
    im.save(f"./static/plot{n}.png")
    return(f"static/plot{n}.png")

def plot_2d(points, points_color, title):

    # make a Figure and attach it to a canvas.
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    canvas = FigureCanvasAgg(fig)
    add_2d_scatter(ax, points, points_color)

    # Retrieve a view on the renderer buffer
    canvas.draw()
    buf = canvas.buffer_rgba()
    # convert to a NumPy array
    X = np.asarray(buf)
    im = Image.fromarray(X)
    n = int(random()*20000)
    im.save(f"./static/plot{n}.png")
    return(f"static/plot{n}.png")

def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    if points_color is not None:
        ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    else:
        ax.scatter(x, y, s=50, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

def MDS(dataset, max_iter=12):
    md_scaling = manifold.MDS(
        n_components=2,
        max_iter=int(max_iter),
        n_init=4,
        random_state=0,
        normalized_stress=False,
    )
    S_scaling = md_scaling.fit_transform(dataset["real_data"])
    return plot_2d(S_scaling, dataset["colors"], "Multidimensional scaling")

def LLE(dataset, n_neighbors=12, method="standard"):
    lle = manifold.LocallyLinearEmbedding(method=method, n_neighbors=12, eigen_solver="auto", n_components=2)
    S_standard = lle.fit_transform(dataset["real_data"])
    return plot_2d(S_standard, dataset["colors"], "LLE")

@app.route("/detah-picture")
def get_detah_picture():
    dataset_name = request.args.get("dataset", "artificial")
    detah_picture_path = plot_3d(datasets[dataset_name]["data"], datasets[dataset_name]["colors"], "Original S-curve samples")
    return detah_picture_path

@app.route("/gimmee-a-graph")
def gimmeeeeee():
    algo = request.args.get("algo", "MDS")
    dataset = datasets[request.args.get("dataset", "standard")]
    if algo == "MDS":
        max_iter = request.args.get("max_iter", 12)
        return MDS(dataset, max_iter=max_iter)
    elif algo == "LLE":
        n_neighbors = request.args.get("n_neighbors", 12)
        method = request.args.get("method", "standard")
        return LLE(dataset, n_neighbors, method)


@app.route("/")
def hello_world():
    return render_template('index.html')
