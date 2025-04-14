import numpy as np
import pandas as pd


def transform_points(coords, transform):
    transformed_coords = np.zeros(coords.shape)
    transform = np.loadtxt(transform)

    M = transform[:3, :3]
    abc = transform[:3, 3]

    for i in range(len(coords)):
        vec = coords[i, :]
        tvec = M.dot(vec) + abc
        transformed_coords[i, :] = tvec[:3]

    transformed_coords = np.round(transformed_coords, 1).astype(float)

    return transformed_coords


def array_to_fcsv(coord_array, output_fcsv):
    with open(output_fcsv, "w") as fid:
        fid.write("# Markups fiducial file version = 4.11\n")
        fid.write("# CoordinateSystem = 0\n")
        fid.write(
            "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n"
        )

    out_df = {
        "node_id": [],
        "x": [],
        "y": [],
        "z": [],
        "ow": [],
        "ox": [],
        "oy": [],
        "oz": [],
        "vis": [],
        "sel": [],
        "lock": [],
        "label": [],
        "description": [],
        "associatedNodeID": [],
    }

    for i, coord in enumerate(coord_array):
        out_df["node_id"].append(i + 1)
        out_df["x"].append(coord[0])
        out_df["y"].append(coord[1])
        out_df["z"].append(coord[2])
        out_df["ow"].append(0)
        out_df["ox"].append(0)
        out_df["oy"].append(0)
        out_df["oz"].append(0)
        out_df["vis"].append(1)
        out_df["sel"].append(1)
        out_df["lock"].append(1)
        out_df["label"].append(str(i + 1))
        out_df["description"].append("elec_type")
        out_df["associatedNodeID"].append("vtkMRMLScalarVolumeNode2")

    out_df = pd.DataFrame(out_df)
    out_df.to_csv(
        output_fcsv,
        sep=",",
        index=False,
        lineterminator="",
        mode="a",
        header=False,
        float_format="%.3f",
    )


def apply_transform(orig_fcsv, transform_path, out_fcsv):
    coords = pd.read_csv(orig_fcsv, skiprows=3, header=None)
    coords = coords[[1, 2, 3]].to_numpy()

    transformed = transform_points(coords=coords, transform=transform_path)

    array_to_fcsv(transformed, out_fcsv)


if __name__ == "__main__":
    apply_transform(
        orig_fcsv=snakemake.input["coords"],
        transform_path=snakemake.input["transformation_matrix"],
        out_fcsv=snakemake.output["transformed_coords"],
    )
