from nnunet_contact_seg.workflow.lib import utils as utils


def get_model():
    local_model = config["resource_urls"].get("nnUNet_model")

    return (
        Path(utils.get_download_dir()) / "model" / Path(local_model).name
    ).absolute()

def get_cmd_copy_inputs(wildcards, input):
    in_img = input.in_img
    print(in_img)
    if isinstance(in_img, str):
        # we have one input image
        return f"cp {in_img} tempimg/temp_000_0000.nii.gz"
    else:
        cmd = []
        # we have multiple input images
        for i, img in enumerate(input.in_img):
            cmd.append(f"cp {img} tempimg/temp_{i:03d}_0000.nii.gz")
        return " && ".join(cmd)


rule download_model:
    params:
        url=config["resource_urls"]["nnUNet_model"],
        model_dir=Path(utils.get_download_dir()) / "model",
    output:
        nnUNet_model=get_model(),
    shell:
        "mkdir -p {params.model_dir} && wget https://{params.url} -O {output}"


rule model_inference:
    input:
        in_img=bids(
            root=config["output_dir"],
            datatype="registration",
            space="native",
            suffix="MNI.nii.gz",
            **inputs["post_ct"].wildcards,
        ),
        nnUNet_model=get_model(),
    params:
        device="gpu" if config["use_gpu"] else "cpu",
        cmd_copy_inputs=get_cmd_copy_inputs,
        temp_lbl="templbl/temp_000.nii.gz",
        model_dir="tempmodel",
        in_folder="tempimg",
        out_folder="templbl",
    output:
        # contact_seg =  inputs['post_ct'].expand(
        #     bids(
        #         root = config['output_dir'],
        #         suffix = 'dseg.nii.gz',
        #         desc = 'contacts_nnUNet',
        #         **inputs['post_ct'].wildcards
        #     )
        # ),
        contact_seg=bids(
            root=config["output_dir"],
            suffix="dseg.nii.gz",
            desc="contacts_nnUNet",
            **inputs["post_ct"].wildcards,
        ),
    log:
        bids(root="logs", suffix="nnUNet.txt", **inputs["post_ct"].wildcards),
        # inputs['post_ct'].expand(
        #     bids(
        #         root="logs",
        #         suffix="nnUNet.txt",
        #         **inputs['post_ct'].wildcards,
        #     ),
        # )
    # shadow:
    #     "minimal"
    threads: 16
    resources:
        gpus=1 if config["use_gpu"] else 0,
        mem_mb=16000,
        time=30 if config["use_gpu"] else 60,
    group:
        "subj"
    shell:
        #create temp folders
        #cp input image to temp folder
        #extract model
        #set nnunet env var to point to model
        #set threads
        # run inference
        #copy from temp output folder to final output
        "mkdir -p {params.model_dir} {params.in_folder} {params.out_folder} && "
        "{params.cmd_copy_inputs} && "
        "unzip -q -n {input.nnUNet_model} -d {params.model_dir} && "
        "export nnUNet_results={params.model_dir} && "
        "export nnUNet_raw={params.in_folder} && "
        "pwd && "
        "nnUNetv2_predict -device {params.device} -d Dataset011_seeg_contacts -i {params.in_folder} -o {params.out_folder} -f 0 -tr nnUNetTrainer_250epochs --disable_tta -c 3d_fullres -p nnUNetPlans &> {log} && "
        "echo 'nnUNet prediction complete' && "
        "cp {params.temp_lbl} {output.contact_seg}"
