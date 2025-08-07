#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 15:15:58 2025

@author: arun
"""

import pandas as pd
import numpy as np
import re
import os
import json
import math


def custom_round(value):
    if value % 1 >= 0.8 or value % 1 <= 0.1:
        return round(value)
    elif 0.45 <= value % 1 <= 0.6:
        return math.floor(value) + 0.5


def determineFCSVCoordSystem(input_fcsv, overwrite_fcsv=False):
    # need to determine if file is in RAS or LPS
    # loop through header to find coordinate system
    coordFlag = re.compile("# CoordinateSystem")
    verFlag = re.compile("# Markups fiducial file version")
    headFlag = re.compile("# columns")
    coord_sys = None
    headFin = None
    ver_fin = None

    with open(input_fcsv, "r") as myfile:
        firstNlines = myfile.readlines()[0:3]

    for row in firstNlines:
        row = re.sub("[\s\,]+[\,]", "", row).replace("\n", "")
        cleaned_dict = {row.split("=")[0].strip(): row.split("=")[1].strip()}
        if None in list(cleaned_dict):
            cleaned_dict["# columns"] = cleaned_dict.pop(None)
        if any(coordFlag.match(x) for x in list(cleaned_dict)):
            coord_sys = list(cleaned_dict.values())[0]
        if any(verFlag.match(x) for x in list(cleaned_dict)):
            verString = list(filter(verFlag.match, list(cleaned_dict)))
            assert len(verString) == 1
            ver_fin = verString[0].split("=")[-1].strip()
        if any(headFlag.match(x) for x in list(cleaned_dict)):
            headFin = list(cleaned_dict.values())[0].split(",")

    if any(x in coord_sys for x in {"LPS", "1"}):
        df = pd.read_csv(input_fcsv, skiprows=3, header=None)

        if df.shape[1] != 13:
            df = df.iloc[:, :14]

        df[1] = -1 * df[1]  # flip orientation in x
        df[2] = -1 * df[2]  # flip orientation in y

        if overwrite_fcsv:
            with open(input_fcsv, "w") as fid:
                fid.write("# Markups fiducial file version = 4.11\n")
                fid.write("# CoordinateSystem = 0\n")
                fid.write(
                    "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n"
                )

            df.rename(
                columns={
                    0: "node_id",
                    1: "x",
                    2: "y",
                    3: "z",
                    4: "ow",
                    5: "ox",
                    6: "oy",
                    7: "oz",
                    8: "vis",
                    9: "sel",
                    10: "lock",
                    11: "label",
                    12: "description",
                    13: "associatedNodeID",
                },
                inplace=True,
            )

            df["associatedNodeID"] = pd.Series(np.repeat("", df.shape[0]))
            df.round(6).to_csv(
                input_fcsv,
                sep=",",
                index=False,
                lineterminator="",
                mode="a",
                header=False,
                float_format="%.6f",
            )

            # print(
            #     f"Converted LPS to RAS: {os.path.dirname(input_fcsv)}/{os.path.basename(input_fcsv)}"
            # )
    return coord_sys, headFin


def create_electrode_list(df_te):
    """create a list of dictionaries for each electrode.
    Dictionaries contain the target and entry point, the label for that electrode,
    and an empty list for the contacts belonging to that electrode.
    Reads in a 3D Slicer .fcsv file containing the target and entry points
    (in this order) for each electrode trajectory.
    """
    electrodes = []
    for i in range(0, len(df_te), 2):
        target_point = df_te.iloc[i][[1, 2, 3]].values
        entry_point = df_te.iloc[i + 1][[1, 2, 3]].values
        electrode_label = df_te.iloc[i][11]

        electrodes.append(
            {
                "entry_point": entry_point,
                "target_point": target_point,
                "contacts": [],
                "elec_label": electrode_label,
            }
        )
    return electrodes


def calculate_distance_to_line(point, entry, trajectory_vec):
    """
    Returns the magnitude of the orthogonal vector between the contact and the
    planned electrode trajectory. see https://en.wikipedia.org/wiki/Vector_projection.
    Constrained to be along the length of trajectory (prevent incorrect labelling).

    """
    entry = np.array(entry)
    trajectory_vec = np.array(trajectory_vec)
    point = np.array(point)

    unit_vector = trajectory_vec / np.linalg.norm(trajectory_vec)
    vector_to_point = point - entry
    projection_length = np.dot(vector_to_point, unit_vector)
    projection_point = entry + projection_length * unit_vector

    trajectory_length = np.linalg.norm(trajectory_vec)

    if projection_length <= trajectory_length:
        distance = np.linalg.norm(point - projection_point)
        return distance


def assign_contacts_to_electrodes(electrodes, contacts, max_distance_threshold=4.0):
    """
    Iterates through each "contact" or coordinate segmented by the nnUNet model.
    Uses the perpendicular distance of contact from the trajectory to assign a contact to an electrode.
    Distance must be within the threshold (set to 4 mm).
    Depending on the accuracy of implantation (according to plan) can result in missed contacts.
    """

    for contact in contacts:
        min_distance = float("inf")
        assigned_electrode = None

        for electrode in electrodes:
            target_point = electrode["target_point"]
            entry_point = electrode["entry_point"]
            elec_vector = target_point - entry_point

            distance = calculate_distance_to_line(contact, entry_point, elec_vector)
            if (
                distance
                and distance <= max_distance_threshold
                and distance < min_distance
            ):
                min_distance = distance
                assigned_electrode = electrode

        if assigned_electrode:
            assigned_electrode["contacts"].append(contact)

    return electrodes


def new_label_contacts(target_point, contacts):
    contacts = np.array(contacts, dtype=np.float64)

    target_point = np.array(target_point)

    distances = [np.sqrt(np.sum((point - target_point) ** 2)) for point in contacts]

    sorted_indices = np.argsort(distances)

    sorted_contacts = contacts[sorted_indices]

    inter_contact = [
        np.linalg.norm(sorted_contacts[i] - sorted_contacts[i - 1])
        for i, contact in enumerate(sorted_contacts)
        if i > 0
    ]

    spacing = custom_round(np.mean(inter_contact))

    return np.array(sorted_contacts), spacing


def label_multiple_electrodes(electrodes, manufacturer_dict):
    for electrode in electrodes:
        if electrode["contacts"]:
            num_contacts = len(electrode["contacts"])

            electrode["sorted_contacts"], spacing = new_label_contacts(
                electrode["target_point"], electrode["contacts"]
            )
            electrode["contact_labels"] = [
                f"{electrode['elec_label']}-{i+1:02}" for i in range(num_contacts)
            ]

            if manufacturer_dict.get((num_contacts, spacing)):
                electrode["elec_type"] = manufacturer_dict.get((num_contacts, spacing))[
                    0
                ]
            else:
                electrode["elec_type"] = "NA"
    return electrodes


def convert_to_df(labelled_final):
    new_contacts_dicts = []

    # Iterate through each electrode in the labeled electrodes list
    for electrode in labelled_final:
        contact_labels = electrode.get("contact_labels", [])
        contacts = electrode.get("sorted_contacts", [])
        elec_type = electrode.get("elec_type", [])

        # Iterate through each contact and corresponding label
        for label, contact in zip(contact_labels, contacts):
            contact_dict = {
                "x": contact[0],
                "y": contact[1],
                "z": contact[2],
                "contact_label": f"{label}",
                "elec_type": elec_type,
            }
            new_contacts_dicts.append(contact_dict)

    # Display the new list of contact dictionaries
    new_df = pd.DataFrame(new_contacts_dicts)
    new_df.sort_values(by=["contact_label"], inplace=True)
    return new_df


def df_to_fcsv(input_df, output_fcsv):
    """
    Convert dataframe object to Slicer-readable .fcsv file.
    """
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

    for idx, ifid in input_df.iterrows():
        out_df["node_id"].append(idx + 1)
        out_df["x"].append(ifid.iloc[0])
        out_df["y"].append(ifid.iloc[1])
        out_df["z"].append(ifid.iloc[2])
        out_df["ow"].append(0)
        out_df["ox"].append(0)
        out_df["oy"].append(0)
        out_df["oz"].append(0)
        out_df["vis"].append(1)
        out_df["sel"].append(1)
        out_df["lock"].append(1)
        out_df["label"].append(str(ifid.iloc[3]))
        out_df["description"].append(ifid.iloc[4])
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


if __name__ == "__main__":
    determineFCSVCoordSystem(snakemake.input["planned_fcsv"], overwrite_fcsv=True)
    determineFCSVCoordSystem(snakemake.input["coords"], overwrite_fcsv=True)

    df_te = pd.read_csv(snakemake.input["planned_fcsv"], skiprows=3, header=None)
    df_contacts = pd.read_csv(snakemake.input["coords"], skiprows=3, header=None)

    contact_array = df_contacts[[1, 2, 3]].values

    electrode_list = create_electrode_list(df_te)
    grouped = assign_contacts_to_electrodes(electrode_list, contact_array, 5.0)

    # Label contacts for each electrode
    with open(snakemake.params["electrode_type"]) as electrode_file:
        manufacturer_json = json.load(electrode_file)

    manufacturer_dict = {eval(k): v for k, v in manufacturer_json.items()}

    labelled_contacts = label_multiple_electrodes(electrode_list, manufacturer_dict)
    labelled_df = convert_to_df(labelled_contacts)

    df_to_fcsv(labelled_df, snakemake.output["labelled_coords"])