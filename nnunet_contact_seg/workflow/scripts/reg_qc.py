import re
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
from nilearn import plotting
from svgutils.compose import Unit
from svgutils.transform import GroupElement, SVGFigure, fromstring

def svg2str(display_object, dpi):
    """Serialize a nilearn display object to string."""

    image_buf = StringIO()
    display_object.frame_axes.figure.savefig(
        image_buf, dpi=dpi, format="svg", facecolor="k", edgecolor="k"
    )
    return image_buf.getvalue()


def extract_svg(display_object, dpi=250):
    """Remove the preamble of the svg files generated with nilearn."""
    image_svg = svg2str(display_object, dpi)

    image_svg = re.sub(' height="[0-9]+[a-z]*"', "", image_svg, count=1)
    image_svg = re.sub(' width="[0-9]+[a-z]*"', "", image_svg, count=1)
    image_svg = re.sub(
        " viewBox", ' preseveAspectRation="xMidYMid meet" viewBox', image_svg, count=1
    )
    start_tag = "<svg "
    start_idx = image_svg.find(start_tag)
    end_tag = "</svg>"
    end_idx = image_svg.rfind(end_tag)

    # rfind gives the start index of the substr. We want this substr
    # included in our return value so we add its length to the index.
    end_idx += len(end_tag)

    return image_svg[start_idx:end_idx]


def clean_svg(bg1_svgs, bg2_svgs, ref=0):
    # Expecting exactly one slice per direction in each group
    svgs = bg1_svgs + bg2_svgs
    roots = [f.getroot() for f in svgs]

    sizes = []
    for f in svgs:
        viewbox = [float(v) for v in f.root.get("viewBox").split(" ")]
        width = int(viewbox[2])
        height = int(viewbox[3])
        sizes.append((width, height))

    sizes = np.array(sizes)
    width = sizes[ref, 0]
    scales = width / sizes[:, 0]
    heights = sizes[:, 1] * scales

    # Total height based on number of slices in bg1
    nsvgs = len(bg1_svgs)
    fig = SVGFigure(Unit(f"{width}px"), Unit(f"{heights[:nsvgs].sum()}px"))

    yoffset = 0
    for i, r in enumerate(roots):
        r.moveto(0, yoffset, scale_x=scales[i])
        if i < nsvgs - 1:
            yoffset += heights[i]

    # Create two background groups for blending
    newroots = [
        GroupElement(roots[:nsvgs], {"class": "background-svg ct"}),
        GroupElement(roots[nsvgs:], {"class": "background-svg t1w"}),
    ]

    fig.append(newroots)
    fig.root.attrib.pop("width", None)
    fig.root.attrib.pop("height", None)
    fig.root.set("preserveAspectRatio", "xMidYMid meet")

    with TemporaryDirectory() as tmpdirname:
        out_file = Path(tmpdirname) / "tmp.svg"
        fig.save(str(out_file))
        svg = out_file.read_text().splitlines()

    if svg[0].startswith("<?xml"):
        svg = svg[1:]

    svg.insert(
        2,
        """\
        <style type="text/css">
        .background-svg.ct {
            opacity: 1;
            transition: opacity 0.2s ease;
        }
        .background-svg.t1w {
            opacity: 0;
            transition: opacity 0.2s ease;
        }
        </style>
        <script>
        function updateBlend(slider) {
            const alpha = parseFloat(slider.value) / 100;
            document.querySelectorAll('.background-svg.ct').forEach(el => {
                el.style.opacity = 1 - alpha;
            });
            document.querySelectorAll('.background-svg.t1w').forEach(el => {
                el.style.opacity = alpha;
            });
        }
        </script>
        """
    )

    return svg

def output_html_file(ct_img_path, t1w_img_path,output_html):
    """Processes a single subject's images and generates an HTML output."""

    # Load CT image
    ct_img = nib.load(str(ct_img_path))
    ct_img = nib.as_closest_canonical(ct_img)

    match = re.search(r"(sub-P\d+)", str(ct_img_path))
    if match:
        subject_id = match.group(1)
        print(subject_id)
        
    # # t1w img
    t1w_img = nib.load(t1w_img_path)
    t1w_img = nib.as_closest_canonical(t1w_img)

    plot_args_ref = {"dim": -0.5} 
    plot_args_ct = {"dim": -0.5, "vmin": 0, "vmax": 2000}    

    # For CT (background group 1)
    display_x_ct_contacts = plotting.plot_anat(ct_img, display_mode="x", draw_cross=False, cut_coords=[-60, -40, -20, 0, 20, 40], **plot_args_ct)
    bg_x_ct_contacts_svgs = [fromstring(extract_svg(display_x_ct_contacts, 300))]
    display_x_ct_contacts.close()

    display_y_ct_contacts = plotting.plot_anat(ct_img, display_mode="y", draw_cross=False, cut_coords=[-60, -40, -20, 0, 20, 40], **plot_args_ct)
    bg_y_ct_contacts_svgs = [fromstring(extract_svg(display_y_ct_contacts, 300))]
    display_y_ct_contacts.close()

    display_z_ct_contacts = plotting.plot_anat(ct_img, display_mode="z", draw_cross=False, cut_coords=[-60, -40, -20, 0, 20, 40], **plot_args_ct)
    bg_z_ct_contacts_svgs = [fromstring(extract_svg(display_z_ct_contacts, 300))]
    display_z_ct_contacts.close()

    # For T1w (background group 2)
    display_x_t1w_contacts = plotting.plot_anat(t1w_img, display_mode="x", draw_cross=False, cut_coords=[-60, -40, -20, 0, 20, 40], **plot_args_ref)
    bg_x_t1w_contacts_svgs = [fromstring(extract_svg(display_x_t1w_contacts, 300))]
    display_x_t1w_contacts.close()

    display_y_t1w_contacts = plotting.plot_anat(t1w_img, display_mode="y", draw_cross=False, cut_coords=[-60, -40, -20, 0, 20, 40], **plot_args_ref)
    bg_y_t1w_contacts_svgs = [fromstring(extract_svg(display_y_t1w_contacts, 300))]
    display_y_t1w_contacts.close()

    display_z_t1w_contacts = plotting.plot_anat(t1w_img, display_mode="z", draw_cross=False, cut_coords=[-60, -40, -20, 0, 20, 40], **plot_args_ref)
    bg_z_t1w_contacts_svgs = [fromstring(extract_svg(display_z_t1w_contacts, 300))]
    display_z_t1w_contacts.close()

    final_svg_x = "\n".join(clean_svg(bg_x_ct_contacts_svgs, bg_x_t1w_contacts_svgs))
    final_svg_y = "\n".join(clean_svg(bg_y_ct_contacts_svgs, bg_y_t1w_contacts_svgs))
    final_svg_z = "\n".join(clean_svg(bg_z_ct_contacts_svgs, bg_z_t1w_contacts_svgs))

    #display results
    with open(output_html, "w") as f:
        f.write(f"""
            <html><body>
                <center>
                    <h3 style="font-size:42px">{subject_id} Registration QC</h3>
                    <p style="margin:20px;">
                        <label for="blendSlider" style="font-size:18px;">CT ↔ T1w:</label>
                        <input id="blendSlider" type="range" min="0" max="100" value="0" oninput="updateBlend(this)" style="width: 300px; vertical-align: middle;">
                    </p>
                    <p>{final_svg_x}</p>
                    <p>{final_svg_y}</p>
                    <p>{final_svg_z}</p>
                    <hr style="height:4px;border-width:0;color:black;background-color:black;margin:30px;">
                </center>
            </body></html>
        """)

    print(f"HTML output saved to {output_html}")

if __name__ == "__main__":
    ct_img_path = snakemake.input["ct_img"]
    t1w_img_path = snakemake.input["t1w_img"]
    output_html = snakemake.output["html"]

    output_html_file(ct_img_path,t1w_img_path,output_html)