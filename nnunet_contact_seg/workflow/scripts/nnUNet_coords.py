import os
import numpy as np
import pandas as pd
import skimage.measure
import nibabel as nib


def vox2world(vox_coords, img_aff):
    M = img_aff[:3, :3]
    abc = img_aff[:3, 3]

    transform_coords = np.zeros(vox_coords.shape)

    for i in range(len(transform_coords)):
        vec = vox_coords[i, :]
        tvec = M.dot(vec) + abc
        transform_coords[i, :] = tvec[:3]

    transform_coords = np.round(transform_coords, 1).astype(float)

    return transform_coords


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


def determine_coords(pred_path, out_fcsv):
    # load image
    pred_img = nib.load(pred_path)
    pred_data = np.asarray(pred_img.dataobj)
    pred_aff = pred_img.affine
    # threshold for determining individual coordinates
    thresh = np.where(pred_data > 0.9, 1.0, 0.0)

    # #see https://scikit-image.org/docs/0.24.x/api/skimage.measure.html#skimage.measure.label
    # #label connected regions (i.e. connected components of predicted image array - identify individual segmentations)
    labelled = skimage.measure.label(thresh)

    regions = skimage.measure.regionprops(labelled)
    voxel_coords = np.array([region.centroid for region in regions])

    world_coords = vox2world(voxel_coords, pred_aff)

    array_to_fcsv(world_coords, out_fcsv)


if __name__ == "__main__":
    determine_coords(
        pred_path=snakemake.input["model_seg"],
        out_fcsv=snakemake.output["model_coords"],
    )
