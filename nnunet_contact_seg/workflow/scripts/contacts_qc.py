import nibabel as nib
import re
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import csv
from nilearn import plotting
from svgutils.compose import Unit
from svgutils.transform import GroupElement, SVGFigure, fromstring
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import io
from label_map import convert_acronym_to_words

# reused from degad code
def svg2str(display_object, dpi):
    """Serialize a nilearn display object to string."""
    image_buf = StringIO()
    display_object.frame_axes.figure.savefig(
        image_buf, dpi=dpi, format="svg", facecolor="k", edgecolor="k"
    )
    return image_buf.getvalue()

# reused from degad code 
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

# altered from degad code 
def clean_svgs(bg1_svgs_strs, bg2_svgs_strs, ref=0):
    """
    Clean and stack SVGs vertically with toggleable opacity.

    Args:
        bg1_svgs_strs: List of SVG strings for background 1 (e.g., CT).
        bg2_svgs_strs: List of SVG strings for background 2 (e.g., T1w).
        ref: Index to use as reference for width scaling.

    Returns:
        Final cleaned SVG lines for the html file.
    """  
    bg1_svgs = [fromstring(s) for s in bg1_svgs_strs]
    bg2_svgs = [fromstring(s) for s in bg2_svgs_strs]  

    svgs = bg1_svgs + bg2_svgs
    roots = [s.getroot() for s in svgs]

    sizes = []
    for f in svgs:
        vb = f.root.get("viewBox")
        if vb is None:
            raise ValueError("SVG missing viewBox attribute. Matplotlib SVGs must have it.")
        viewbox = list(map(float, vb.strip().split()))
        width = viewbox[2]
        height = viewbox[3]
        sizes.append((width, height))

    sizes = np.array(sizes)
    ref_width = sizes[ref, 0]
    scales = ref_width / sizes[:, 0]
    heights = sizes[:, 1] * scales

    nsvgs = len(bg1_svgs)
    fig = SVGFigure(Unit(f"{ref_width}px"), Unit(f"{heights[:nsvgs].sum()}px"))

    yoffset = 0
    for i, root in enumerate(roots):
        root.moveto(0, yoffset, scale_x=scales[i], scale_y=scales[i])
        if i < nsvgs - 1:
            yoffset += heights[i]

    newroots = [
        GroupElement(roots[:nsvgs], {"class": "background-svg ct"}),
        GroupElement(roots[nsvgs:], {"class": "background-svg t1w"}),
    ]

    fig.append(newroots)
    fig.root.attrib.pop("width", None)
    fig.root.attrib.pop("height", None)
    fig.root.set("preserveAspectRatio", "xMidYMid meet")

    # Save and inject styles/scripts
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

# dictionary with entry and exit coord for each label
# from the planned fcsv file
def find_entry_exit(contact_fcsv_planned_path):
    
    # {label: (x,y,z), (x,y,z) }
    entry_exit_dict = {}
    with open(contact_fcsv_planned_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            try:
                x, y, z = map(float, row[1:4])
                label = row[11]
                if label not in entry_exit_dict:
                    entry_exit_dict[label] = []
                entry_exit_dict[label].append((x, y, z))
            except ValueError:
                continue
    return entry_exit_dict

# dictionary containing the label with its contact coordinates and corresponding contact numbers
def group_contacts(contact_fcsv_labelled_path):
    
    # {label: [(x,y,z), label_num], ... }
    coords_dict = {}
    with open(contact_fcsv_labelled_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            try:
                x, y, z = map(float, row[1:4])
                label, label_num = row[11].split('-')
                if label not in coords_dict:
                    coords_dict[label] = []
                coords_dict[label].append(((x, y, z), label_num))
            except ValueError:
                continue
    return coords_dict

# extract a slice along the trajectory from the entrance to exit point 
def extract_trajectory_slice(nifti_img, entry, exit, width=40, num_points=256):
    
    # load image and affine
    data = nifti_img.get_fdata()
    affine = nifti_img.affine
    
    # for converting back to voxel from world
    inv_affine = np.linalg.inv(affine)

    # find direction vector
    direction = exit - entry
    
    # find length between the coordinates
    length = np.linalg.norm(direction)
    
    # normalize the direction vector with the length between entry and exit
    # unit vector 
    u = direction / length
    
    # find orthogonal vector for plane creation using the y-direction slice
    # v = np.cross(u, [0,1,0])
    v = np.cross(u, [1,0,0])
        
    # normalize the vector 
    v /= np.linalg.norm(v)
    w = np.cross(u, v)

    # extract the points along the width and length of the desired plane
    s_vals = np.linspace(-width/2, width/2, num_points)
    t_vals = np.linspace(-10, length + 10, num_points)
    
    # create a gride of these points 
    S, T = np.meshgrid(s_vals, t_vals)

    # world coordinates of the slice points
    # generates 3d coords for every pixel in the slice 
    # entry + (t * u) + (s * v)
    coords_world = entry[:, None, None] + T[None, :, :] * u[:, None, None] + S[None, :, :] * v[:, None, None]
    coords_world = coords_world.reshape(3, -1)

    # Convert to voxel space
    coords_voxel = nib.affines.apply_affine(inv_affine, coords_world.T).T

    slice_vals = map_coordinates(data, coords_voxel, order=1, mode='nearest')
    slice_img = slice_vals.reshape(num_points, num_points)

    # Return image and transformation info
    return slice_img, s_vals, t_vals, u, v, w

def project_point_to_slice(P, entry, u, v, w, thickness=20):
    
    # compute the points distance from the 2d plane
    distance_from_plane = np.dot(P - entry, w)
    print(f"distance from plane for {P}: {distance_from_plane}")
    
    # if close to the trajectory line (still within the contact size)
    # plot the point
    if abs(distance_from_plane) <= thickness:
        vec = P - entry
        s = np.dot(vec, v)
        t = np.dot(vec, u)
        return s, t
    else:
        return None, None

def render_oblique_slice_to_svg(img, entry_world, exit_world, points, **args):
    
    # extract the trajectory slice 
    slice_img, s_vals, t_vals, u, v, w = extract_trajectory_slice(img, entry_world, exit_world)

    # Project entry and exit to slice coords
    entry_st = project_point_to_slice(entry_world, entry_world, u, v, w)
    exit_st = project_point_to_slice(exit_world, entry_world, u, v, w)

    # rotate image 90 degrees clockwise
    rotated_img = np.rot90(slice_img, k=-1)

    # plot rotated image
    plt.imshow(rotated_img, cmap='gray', extent=[t_vals[0], t_vals[-1], s_vals[0], s_vals[-1]], **args)

    # Project entry and exit to rotated slice coords (s becomes y-axis, t becomes x-axis)
    plt.scatter(entry_st[1], entry_st[0], color='green', label='Entry Point', s=50)
    plt.scatter(exit_st[1], exit_st[0], color='red', label='Exit Point', s=50)

    # plot the contact coords
    for (pt, label) in points:
        s, t = project_point_to_slice(pt, entry_world, u, v, w)
        
        if s is not None and t is not None:
            plt.scatter(t, s, color='orange', s=50)
            plt.text(t - 2, s - 2, str(label), color='purple', fontsize=12, weight='bold')

    plt.legend()
    plt.axis('off')
    plt.tight_layout(pad=0)
    svg_buffer = io.StringIO()
    plt.savefig(svg_buffer, format='svg', bbox_inches='tight', pad_inches=0)
    svg_str = svg_buffer.getvalue()
    svg_buffer.close()
    plt.close()

    # remove matplotlib specifc svg formatting for the html compatibility
    svg_str = re.sub(r'<\?xml.*?\?>', '', svg_str, flags=re.DOTALL)
    svg_str = re.sub(r'<!DOCTYPE.*?>', '', svg_str, flags=re.DOTALL)
    svg_str = svg_str.lstrip()
    return svg_str

def output_html_file(ct_img_path,t1w_img_path,contact_fcsv_planned_path,contact_fcsv_labelled_path,output_html):

    # Load CT image
    ct_img = nib.load(str(ct_img_path))
    ct_img = nib.as_closest_canonical(ct_img)

    match = re.search(r"(sub-P\d+)", str(ct_img_path))
    if match:
        subject_id = match.group(1)
        print(subject_id)

    # t1w img
    t1w_img = nib.load(t1w_img_path)
    t1w_img = nib.as_closest_canonical(t1w_img)

    # get coordinate dictionaries for plotting
    entry_exit = find_entry_exit(contact_fcsv_planned_path)
    contacts = group_contacts(contact_fcsv_labelled_path)

    html_parts = []
    
    for i, (label, points) in enumerate(contacts.items()):
    # for label, points in contacts.items():
        if i == 3:
            break
        
        # axial, sagittal, coronal views taken at the middle contacts slice
        middle_index = len(points) // 2
        middle_point = points[middle_index][0]

        # arguements for nilearn and matplotlin
        plot_args_ref = {"dim": -0.5} 
        plot_args_ct = {"dim": -0.5, "vmin": 0, "vmax": 2000} 
        matpltlib_refs_t1w =  {} 
        matpltlib_refs_ct =  {"vmin": 0, "vmax": 2000} 

        # coordinates from dictionaries 
        marker_coords = [val[0] for val in points]
        entry_world, exit_world = map(np.array, entry_exit[label])

        # for CT + contacts overlays (background group 1)
        display_x_ct_contacts = plotting.plot_anat(ct_img, display_mode="x", draw_cross=False, cut_coords=[middle_point[0]], **plot_args_ct)
        display_x_ct_contacts.add_markers(marker_coords, marker_color="orange", marker_size=3)
        bg_x_ct_contacts_svgs = extract_svg(display_x_ct_contacts, 300)
        display_x_ct_contacts.close()

        display_y_ct_contacts = plotting.plot_anat(ct_img, display_mode="y", draw_cross=False, cut_coords=[middle_point[1]], **plot_args_ct)
        display_y_ct_contacts.add_markers(marker_coords, marker_color="orange", marker_size=3)
        bg_y_ct_contacts_svgs = extract_svg(display_y_ct_contacts, 300)
        display_y_ct_contacts.close()

        display_z_ct_contacts = plotting.plot_anat(ct_img, display_mode="z", draw_cross=False, cut_coords=[middle_point[2]], **plot_args_ct)
        display_z_ct_contacts.add_markers(marker_coords, marker_color="orange", marker_size=3)
        bg_z_ct_contacts_svgs = extract_svg(display_z_ct_contacts, 300)
        display_z_ct_contacts.close()

        # For T1w + contacts overlays (background group 2)
        display_x_t1w_contacts = plotting.plot_anat(t1w_img, display_mode="x", draw_cross=False, cut_coords=[middle_point[0]], **plot_args_ref)
        display_x_t1w_contacts.add_markers(marker_coords, marker_color="orange", marker_size=3)
        bg_x_t1w_contacts_svgs = extract_svg(display_x_t1w_contacts, 300)
        display_x_t1w_contacts.close()

        display_y_t1w_contacts = plotting.plot_anat(t1w_img, display_mode="y", draw_cross=False, cut_coords=[middle_point[1]], **plot_args_ref)
        display_y_t1w_contacts.add_markers(marker_coords, marker_color="orange", marker_size=3)
        bg_y_t1w_contacts_svgs = extract_svg(display_y_t1w_contacts, 300)
        display_y_t1w_contacts.close()

        display_z_t1w_contacts = plotting.plot_anat(t1w_img, display_mode="z", draw_cross=False, cut_coords=[middle_point[2]], **plot_args_ref)
        display_z_t1w_contacts.add_markers(marker_coords, marker_color="orange", marker_size=3)
        bg_z_t1w_contacts_svgs = extract_svg(display_z_t1w_contacts, 300)
        display_z_t1w_contacts.close()

        # For trajectory slice 
        svg_oblique_ct = render_oblique_slice_to_svg(ct_img, entry_world, exit_world, points, **matpltlib_refs_ct)
        svg_oblique_t1w = render_oblique_slice_to_svg(t1w_img, entry_world, exit_world, points, **matpltlib_refs_t1w)

        final_svg_x = "\n".join(clean_svgs([bg_x_ct_contacts_svgs], [bg_x_t1w_contacts_svgs]))
        final_svg_y = "\n".join(clean_svgs([bg_y_ct_contacts_svgs], [bg_y_t1w_contacts_svgs]))
        final_svg_z = "\n".join(clean_svgs([bg_z_ct_contacts_svgs], [bg_z_t1w_contacts_svgs]))
        final_svg_oblique = "\n".join(clean_svgs([svg_oblique_ct], [svg_oblique_t1w]))

        label_long = convert_acronym_to_words(label)

        html_parts.append(f"""
            <div style="margin-bottom: 40px;">
                <p style="font-size:20px;"><b>{label_long}</b></p>
                <div style="display: flex; justify-content: center; align-items: center; gap: 10px; text-align: center; background-color: black;">
                    <div style="width: 400px;">
                        {final_svg_x}
                    </div>
                    <div style="width: 400px;">
                        {final_svg_y}
                    </div>
                    <div style="width: 400px;">
                        {final_svg_z}
                    </div>
                    <div style="width: 400px;">
                        {final_svg_oblique}
                    </div>
                </div>
                <div style="display: flex; justify-content: center; align-items: center; gap: 10px; text-align: center;">
                    <div style="width: 400px; ">
                        <p>Sagittal Slice</p>
                    </div>
                    <div style="width: 400px;">
                        <p>Coronal Slice</p>
                    </div>
                    <div style="width: 400px;">
                        <p>Axial Slice</p>
                    </div>
                    <div style="width: 400px;">
                        <p>Slice along the trajectory with fixed coronal plane from planned entry -> exit</p>
                    </div>
                </div>
                <hr style="height:4px;border-width:0;color:black;background-color:black;margin-top:30px;">
            </div>
        """)

        print("finished label: ", label)

        i += 1
    with open(output_html, "w") as f:
        f.write(f"""
        <html>
        <head>
            <style>
                #sliderContainer {{
                    position: fixed;
                    top: 0;
                    left: 0;
                    right: 0;
                    background-color: white;
                    padding: 20px;
                    text-align: center;
                    z-index: 1000;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                body {{
                    margin-top: 100px;  
                }}
            </style>
        </head>
        <body>
            <div id="sliderContainer">
                <h3 style="font-size:42px; margin: 0;">{subject_id} Labelled Contacts</h3>
                <p style="margin:10px;">
                    <label for="blendSlider" style="font-size:18px;">CT ↔ T1w:</label>
                    <input id="blendSlider" type="range" min="0" max="100" value="0"
                        oninput="updateBlend(this)" style="width: 300px; vertical-align: middle;">
                </p>
            </div>
            <center style="padding-top:40px;">
                {''.join(html_parts)}
            </center>
        </body>
        </html>
        """)

    print(f"HTML output saved to {output_html}")

if __name__ == "__main__":
    ct_img_path = snakemake.input["ct_img"]
    t1w_img_path = snakemake.input["t1w_img"]
    contact_fcsv_planned_path = snakemake.input["contact_fcsv_planned"]
    contact_fcsv_labelled_path = snakemake.input["contact_fcsv_labelled"]
    output_html = snakemake.output["html"]

    output_html_file(ct_img_path,t1w_img_path,contact_fcsv_planned_path,contact_fcsv_labelled_path,output_html)