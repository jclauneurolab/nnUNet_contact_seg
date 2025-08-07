import nibabel as nib
import re
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import csv
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
def extract_svg(display_object, dpi=150):
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
            transition: opacity 0.2s linear;
        }
        .background-svg.t1w {
            opacity: 0;
            transition: opacity 0.2s linear;
        }
        </style>
        """
    )

    return svg

# dictionary with entry and exit coord for each label
# from the actual fcsv file
def find_entry_exit(contact_fcsv_actual_path):
    
    # {label: (x,y,z), (x,y,z) }
    entry_exit_dict = {}
    with open(contact_fcsv_actual_path, 'r') as f:
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
def extract_trajectory_slice(nifti_img, entry, exit, width=40, num_points=128):
    
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
    
    # find orthogonal vector for plane creation 
    # v = np.cross(u, [0,1,0])
    v = np.cross(u, [1,0,0])
        
    # normalize the vector 
    v /= np.linalg.norm(v)
    w = np.cross(u, v)

    s_vals = np.linspace(-width/2, width/2, num_points)
    t_vals = np.linspace(-10, length+10, num_points)
    
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

def project_point_to_slice(P, entry, u, v, w):

    # compute the points distance from the 2d plane
    w_normalized = w / np.linalg.norm(w)
    distance_from_plane = np.dot(P - entry, w_normalized)
    
    # if close to the trajectory line (still within the contact size)
    # plot the point
    # setting thickness to be 1mm 
    # if off my more than 1mm, the point will not be included 
    if abs(distance_from_plane) <= 20:
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

    plt.imshow(slice_img.T, cmap='gray', extent=[t_vals[0], t_vals[-1], s_vals[0], s_vals[-1]], origin='lower', **args)

    # Plot entry/exit
    plt.scatter(entry_st[1], entry_st[0], color='green', label='Entry Point', s=100)
    plt.scatter(exit_st[1], exit_st[0], color='red', label='Target Point', s=100)

    # Plot contacts
    for (pt, label) in points:
        s, t = project_point_to_slice(pt, entry_world, u, v, w)
        if s is not None and t is not None:
            plt.scatter(t, s, color='orange', s=100, edgecolors='white')
            plt.text(t + 1, s + 1, str(label), color='white', fontsize=10, weight='bold')

    plt.legend()
    plt.axis('off')
    plt.tight_layout(pad=0)
    svg_buffer = io.StringIO()
    plt.savefig(svg_buffer, format='svg', bbox_inches='tight', pad_inches=0, facecolor='black')
    svg_str = svg_buffer.getvalue()
    svg_buffer.close()
    plt.close()

    # remove matplotlib specifc svg formatting for the html compatibility
    svg_str = re.sub(r'<\?xml.*?\?>', '', svg_str, flags=re.DOTALL)
    svg_str = re.sub(r'<!DOCTYPE.*?>', '', svg_str, flags=re.DOTALL)
    svg_str = svg_str.lstrip()
    return svg_str

def render_plane_slice_to_svg(img, middle_point, points, axis, **args):
    
    data = img.get_fdata()
    affine = img.affine

    # for converting back to voxel from world
    inv_affine = np.linalg.inv(affine)

    # add 1 to make it middle point homogeneous to the affine matrix
    # apply inverse matrix for the conversion to voxel
    # remove the 1
    point_vox = inv_affine.dot(np.append(middle_point, 1))[:3]
    # convert to ints for slicing 
    point_vox = np.round(point_vox).astype(int)

    # compute voxel spacing from the world spacing
    # important for the images to not look squished with matplotlib
    # zooms[0] = x zoom etc. 
    zooms = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))

    # extent = [x_min, x_max, y_min, y_max]
    if axis == 'x':

        # [y, z]
        slice_2d = data[point_vox[0], :, :]
        extent = [0, data.shape[1] * zooms[1], 0, data.shape[2] * zooms[2]]
    elif axis == 'y':
        slice_2d = data[:, point_vox[1], :]
        # [x, z]
        extent = [0, data.shape[0] * zooms[0], 0, data.shape[2] * zooms[2]]
    elif axis == 'z':
        slice_2d = data[:, :, point_vox[2]]
        # [x, y]
        extent = [0, data.shape[0] * zooms[0], 0, data.shape[1] * zooms[1]]

    plt.imshow(slice_2d.T, cmap='gray', origin='lower', extent=extent, **args)

    # overlay points within 1mm of the selected slice
    for pt_entry in points:
        
        # take only the points not the label 
        pt_world, label = pt_entry

        pt_world = np.array(pt_world) # convert tuple
        # same conversion as above
        pt_vox = inv_affine.dot(np.append(pt_world, 1))[:3]
        
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        axis_idx = axis_map[axis]
        plane_axes = [i for i in [0, 1, 2] if i != axis_idx]

        # see is the point is within 1mm of the slice 
        if abs(pt_vox[axis_idx] - point_vox[axis_idx]) * zooms[axis_idx] < 1:
            x = pt_vox[plane_axes[0]] * zooms[plane_axes[0]]
            y = pt_vox[plane_axes[1]] * zooms[plane_axes[1]]
            plt.scatter(x, y, color='orange', s=40)

    plt.axis('off')
    plt.tight_layout(pad=0)
    svg_buffer = io.StringIO()
    plt.savefig(svg_buffer, format='svg', bbox_inches='tight', pad_inches=0, facecolor='black')
    svg_str = svg_buffer.getvalue()
    svg_buffer.close()
    plt.close()

    svg_str = re.sub(r'<\?xml.*?\?>', '', svg_str, flags=re.DOTALL)
    svg_str = re.sub(r'<!DOCTYPE.*?>', '', svg_str, flags=re.DOTALL)
    return svg_str.lstrip()

def output_html_file(ct_img_path,t1w_img_path,contact_fcsv_actual_path,contact_fcsv_labelled_path,output_html):

    # Load CT image
    ct_img = nib.load(str(ct_img_path))
    ct_img = nib.as_closest_canonical(ct_img)

    match = re.search(r"(sub-P\d+)", str(ct_img_path))
    if match:
        subject_id = match.group(1)

    # t1w img
    t1w_img = nib.load(t1w_img_path)
    t1w_img = nib.as_closest_canonical(t1w_img)
    
    # get coordinate dictionaries for plotting
    entry_exit = find_entry_exit(contact_fcsv_actual_path)
    contacts = group_contacts(contact_fcsv_labelled_path)

    html_parts = []

    for label, points in contacts.items():
        
        # axial, sagittal, coronal views taken at the middle contacts slice
        middle_index = len(points) // 2
        middle_point = points[middle_index][0]

        # arguements matplotlib
        matplotlib_refs_t1w =  {} 
        matplotlib_refs_ct =  {"vmin": 0, "vmax": 2000} 

        # coordinates from dictionaries 
        marker_coords = [val[0] for val in points]
        exit_world, entry_world = map(np.array, entry_exit[label])

        # for CT + contacts overlays (background group 1)
        bg_x_ct_contacts_svgs = render_plane_slice_to_svg(ct_img, middle_point, points, axis="x", **matplotlib_refs_ct)
        bg_y_ct_contacts_svgs = render_plane_slice_to_svg(ct_img, middle_point, points, axis="y", **matplotlib_refs_ct)
        bg_z_ct_contacts_svgs = render_plane_slice_to_svg(ct_img, middle_point, points, axis="z", **matplotlib_refs_ct)

        # For T1w + contacts overlays (background group 2)
        bg_x_t1w_contacts_svgs = render_plane_slice_to_svg(t1w_img, middle_point, points, axis="x", **matplotlib_refs_t1w)
        bg_y_t1w_contacts_svgs = render_plane_slice_to_svg(t1w_img, middle_point, points, axis="y", **matplotlib_refs_t1w)
        bg_z_t1w_contacts_svgs = render_plane_slice_to_svg(t1w_img, middle_point, points, axis="z", **matplotlib_refs_t1w)

        # For trajectory slice 
        svg_oblique_ct = render_oblique_slice_to_svg(ct_img, entry_world, exit_world, points, **matplotlib_refs_ct)
        svg_oblique_t1w = render_oblique_slice_to_svg(t1w_img, entry_world, exit_world, points, **matplotlib_refs_t1w)

        final_svg_x = "\n".join(clean_svgs([bg_x_ct_contacts_svgs], [bg_x_t1w_contacts_svgs]))
        final_svg_y = "\n".join(clean_svgs([bg_y_ct_contacts_svgs], [bg_y_t1w_contacts_svgs]))
        final_svg_z = "\n".join(clean_svgs([bg_z_ct_contacts_svgs], [bg_z_t1w_contacts_svgs]))
        final_svg_oblique = "\n".join(clean_svgs([svg_oblique_ct], [svg_oblique_t1w]))

        label_long = convert_acronym_to_words(label)

        html_parts.append(f"""
            <div style="margin-bottom: 40px;">
                <p style="font-size:20px;"><b>({label}) - {label_long}</b></p>
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
                        <p>Slice along the trajectory with fixed coronal plane from entry -> exit</p>
                    </div>
                </div>
                <hr style="height:4px;border-width:0;color:black;background-color:black;margin-top:30px;">
            </div>
        """)

    with open(output_html, "w") as f:
        f.write(f"""
            <html>
            <head>
                <style>
                    #headerContainer {{
                        position: fixed;
                        top: 0;
                        left: 0;
                        right: 0;
                        background-color: white;
                        padding: 10px;
                        text-align: center;
                        z-index: 1000;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    }}
                    body {{
                        margin-top: 100px;
                    }}
                    @keyframes flickerAnimation {{
                        0% {{ opacity: 1; }}
                        50% {{ opacity: 0; }}
                        100% {{ opacity: 1; }}
                    }}
                    .foreground-svg {{
                        animation: flickerAnimation 3s ease-in-out infinite;
                    }}
                    .button-style {{
                        font-size: 18px;
                        padding: 10px 20px;
                        margin: 5px;
                        border: solid;
                        border-radius: 8px;
                        background-color: white;
                        cursor: pointer;
                        transition: background-color 0.3s ease;
                    }}
                    .button-style:hover {{
                        background-color: #ECECEC;
                    }}
                </style>
            </head>
            <body>
                <div id="headerContainer">
                    <h3 style="font-size:42px; margin: 0;">{subject_id} Labelled Contacts</h3>
                    <p style="margin:10px;">
                        <button class="button-style" onclick="switchView()">Switch CT ↔ T1w</button>
                        <button class="button-style" onclick="blendView()">Blend CT ↔ T1w</button>
                    </p>
                </div>

                <center style="padding-top:40px;">
                    {''.join(html_parts)}
                </center>

                <script>
                    function switchView() {{
                        const ct = document.querySelectorAll('.background-svg.ct');
                        const t1w = document.querySelectorAll('.background-svg.t1w');
                        const isCTVisible = parseFloat(getComputedStyle(ct[0]).opacity) > 0.5;

                        t1w.forEach(el => el.classList.remove('foreground-svg'));

                        ct.forEach(el => el.style.opacity = isCTVisible ? 0 : 1);
                        t1w.forEach(el => el.style.opacity = isCTVisible ? 1 : 0);
                    }}
                    function blendView() {{
                        const ct = document.querySelectorAll('.background-svg.ct');
                        const t1w = document.querySelectorAll('.background-svg.t1w');

                        ct.forEach(el => el.style.opacity = 1);
                        t1w.forEach(el => {{
                            el.style.opacity = 1;
                            el.classList.toggle('foreground-svg');
                        }});
                    }}
                </script>
            </body>
            </html>
        """)

if __name__ == "__main__":
    ct_img_path = snakemake.input["ct_img"]
    t1w_img_path = snakemake.input["t1w_img"]
    contact_fcsv_actual_path = snakemake.input["contact_fcsv_actual"]
    contact_fcsv_labelled_path = snakemake.input["contact_fcsv_labelled"]
    output_html = snakemake.output["html"]

    output_html_file(ct_img_path,t1w_img_path,contact_fcsv_actual_path,contact_fcsv_labelled_path,output_html)