rule get_coords:
    input:
        model_seg=rules.model_inference.output.contact_seg,
    output:
        model_coords=bids(
            root=config["output_dir"],
            suffix="nnUNet.fcsv",
            **inputs["post_ct"].wildcards,
        ),
    group:
        "subj"
    script:
        "../scripts/nnUNet_coords.py"


if config["transform"]:

    rule transform_coords:
        input:
            coords=rules.get_coords.output.model_coords,
            transformation_matrix = bids(
                root=config["output_dir"],
                datatype="registration",
                space="native",
                suffix="xfm.txt",
                **inputs["post_ct"].wildcards,
            ),
        output:
            transformed_coords=bids(
                root=config["output_dir"],
                suffix="transformed_nnUNet.fcsv",
                **inputs["post_ct"].wildcards,
            ),
        group:
            "subj"
        script:
            "../scripts/transform_coords.py"


if config["label"]:

    rule label_coords:
        input:
            coords=rules.transform_coords.output.transformed_coords,
            planned_fcsv=bids(
                root=config["bids_dir"],
                suffix="planned",
                extension=".fcsv",
                **inputs["post_ct"].wildcards,
            ),
        output:
            labelled_coords=bids(
                root=config["output_dir"],
                suffix="labelled_nnUNet.fcsv",
                **inputs["post_ct"].wildcards,
            ),
        params:
            electrode_type=str(Path(workflow.basedir).parent / config["electrode_type"]),
        group:
            "subj"
        script:
            "../scripts/label.py"
