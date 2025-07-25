import ants

# Code is currently not being used - included in case we want to use for the t1w image later on

def n4_bias_correction(input_image_path, output_image_path):
    """
    Performs n4 bias field correction.

    Parameters
    ----------
        input_image_path:: str
            Path to the input image

        output_image_path:: str
            Path to the output image

    Returns
    -------
        None
    """
    input_image = ants.image_read(input_image_path)

    # Perform the bias field correction
    output_image = ants.n4_bias_field_correction(input_image)

    ants.image_write(output_image, output_image_path)


n4_bias_correction(snakemake.input.t1w, snakemake.output.corrected_t1w)