rule register_contacts:
    input: 
        in_im=bids(
            root=config["output_dir"],
            suffix="dseg.nii.gz",
            desc="contacts_nnUNet",
            datatype="contact_seg",
            **inputs["post_ct"].wildcards,
        ),
        t1w_img=bids(
            root=config["output_dir"],
            suffix="T1w",
            desc="n4biascorr",
            datatype="n4biascorr",
            extension=".nii.gz",
            **inputs["pre_t1w"].wildcards,
        ),
        transform_matrix=get_reg_matrix()
    output:
        out_im=bids(
            root=config["output_dir"],
            suffix="dseg.nii.gz",
            datatype="registration",
            space="T1w",
            desc="contacts_nnUNet",
            **inputs["post_ct"].wildcards,
        )
    script:
        '../scripts/apply_registration.py'


rule reg_qc:
    input: 
        ct_img=get_registered_ct_image(),
        t1w_img=bids(
            root=config["output_dir"],
            suffix="T1w",
            desc="n4biascorr",
            datatype="n4biascorr",
            extension=".nii.gz",
            **inputs["pre_t1w"].wildcards,
        ),
    output: 
        html=bids(
            root=config["output_dir"],
            datatype="qc",
            desc="reg_qc",
            suffix=".html",
            **inputs["post_ct"].wildcards,
        )

    script: '../scripts/reg_qc.py'

rule contacts_qc:
    input: 
        ct_img=get_registered_ct_image(),
        t1w_img=bids(
            root=config["output_dir"],
            suffix="T1w",
            desc="n4biascorr",
            datatype="n4biascorr",
            extension=".nii.gz",
            **inputs["pre_t1w"].wildcards,
        ),
        contact_fcsv_labelled=bids(
            root=config["output_dir"],
            datatype="coords",
            suffix="labelled_nnUNet.fcsv",
            **inputs["post_ct"].wildcards,
        ),
        contact_fcsv_planned=bids(
                root=config["bids_dir"],
                suffix="planned",
                extension=".fcsv",
                **inputs["post_ct"].wildcards,
            ),
    output: 
        html=bids(
            root=config["output_dir"],
            datatype="qc",
            desc="contacts_qc",
            suffix=".html",
            **inputs["post_ct"].wildcards,
        )
    script: '../scripts/contacts_qc.py'