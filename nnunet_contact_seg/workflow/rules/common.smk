def get_reg_matrix():
    if not config["reg_matrix_path"]:
        return bids(
                root=config["output_dir"],
                datatype="registration",
                space="T1w",
                suffix="xfm.txt",
                **inputs["post_ct"].wildcards,
            )
    else:
        return config["reg_matrix_path"]

def get_registered_ct_image():
    if not config["reg_matrix_path"]:
        return bids(
            root=config["output_dir"],
            datatype="registration",
            space="T1w",
            suffix="ct.nii.gz",
            **inputs["post_ct"].wildcards,
        ),
    else:
        return bids(
            root=config["bids_dir"],
            suffix="ct",
            datatype="ct",
            session="post",
            acq="Electrode",
            extension=".nii.gz",
            **inputs["post_ct"].wildcards,
        )
    
def get_final_output():
    final = []
    final.extend(
        inputs["post_ct"].expand(
            bids(
                root=config["output_dir"],
                suffix="nnUNet.fcsv",
                datatype="coords",
                **inputs["post_ct"].wildcards,
            )
        )
    )
    if config["label"]:
        final.extend(
            inputs["post_ct"].expand(
                bids(
                    root=config["output_dir"],
                    datatype="coords",
                    suffix="labelled_nnUNet.fcsv",
                    **inputs["post_ct"].wildcards,
                )
            )
        )
    if config["transform"]:
        final.extend(
            inputs["post_ct"].expand(
                bids(
                    root=config["output_dir"],
                    datatype="coords",
                    suffix="transformed_nnUNet.fcsv",
                    **inputs["post_ct"].wildcards,
                )
            )
        )
    if config["reg_qc"]:
        final.extend(
            inputs["post_ct"].expand(
                bids(
                    root=config["output_dir"],
                    datatype="qc",
                    desc="reg_qc",
                    suffix=".html",
                    **inputs["post_ct"].wildcards,
                )
            )
        )
    if config["contacts_qc"]:
        final.extend(
            inputs["post_ct"].expand(
                bids(
                    root=config["output_dir"],
                    datatype="qc",
                    desc="contacts_qc",
                    suffix=".html",
                    **inputs["post_ct"].wildcards,
                )
            )
        )
    return final
