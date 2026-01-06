import config as cfg

def get_model_xml_explicit():
    p = cfg.PHYSICS_CONFIG
    
    xml_header = f"""
    <mujoco model="flag_manual">
        <option timestep="{p['timestep']}" gravity="{p['gravity']}" density="{p['density']}" viscosity="{p['viscosity']}" integrator="{p['integrator']}"/>
        <visual><map force="0.1" zfar="30"/></visual>
        
        <default>
            <geom type="sphere" size="{p['node_radius']}" mass="{p['node_mass']}" rgba=".8 .2 .2 1" friction="{p['friction']}" solref="{p['solref']}"/>
            <site type="sphere" size="0.002" rgba="1 1 1 0"/>
        </default>

        <worldbody>
            <light pos="0 0 10"/>
            <geom name="floor" type="plane" size="10 10 .1" pos="0 0 {p['floor_z']}" rgba=".9 .9 .9 1"/>
    """
    
    bodies_xml = ""
    for r in range(p['grid_h']): 
        for c in range(p['grid_w']):
            name = f"B_{r}_{c}"
            site_name = f"S_{r}_{c}"
            
            x = c * cfg.SPACING_W
            y = 0
            z = p['start_z'] - (r * cfg.SPACING_H)
            
            # --- STATIC POLE LOGIC ---
            # Column 0: No Joint = Static Body (Fixed to World)
            if c == 0:
                joint_xml = "" 
                color_xml = 'rgba=".5 .5 .5 1"'
            else:
                # Column 1+: Free Joint with high damping
                joint_xml = '<joint type="free" damping="0.00001"/>'
                color_xml = 'rgba=".8 .2 .2 1"'

            bodies_xml += f"""
            <body name="{name}" pos="{x} {y} {z}">
                <geom {color_xml}/> 
                <site name="{site_name}" pos="0 0 0"/> 
                {joint_xml}
            </body>
            """
    
    xml_footer_start = "</worldbody>\n<tendon>"
    
    tendons_xml = ""
    
    def make_tendon(s1_name, s2_name):
        return f"""
        <spatial stiffness="{p['stiffness']}" damping="{p['damping']}" width="{p['width']}">
            <site site="{s1_name}"/> <site site="{s2_name}"/>
        </spatial>"""

    # Horizontal
    for r in range(p['grid_h']):
        for c in range(p['grid_w'] - 1):
            tendons_xml += make_tendon(f"S_{r}_{c}", f"S_{r}_{c+1}")

    # Vertical
    for r in range(p['grid_h'] - 1):
        for c in range(p['grid_w']):
            tendons_xml += make_tendon(f"S_{r}_{c}", f"S_{r+1}_{c}")
    
    # Diagonals
    for r in range(p['grid_h'] - 1):
        for c in range(p['grid_w'] - 1):
            tendons_xml += make_tendon(f"S_{r}_{c}", f"S_{r+1}_{c+1}")
            tendons_xml += make_tendon(f"S_{r}_{c+1}", f"S_{r+1}_{c}")
    
    
    def make_bend_tendon(s1_name, s2_name):
        return f"""
        <spatial stiffness="{p['bending_stiffness']}" damping="{p['damping']}" width="{p['width']}">
            <site site="{s1_name}"/> <site site="{s2_name}"/>
        </spatial>"""

    # Horizontal Bending (Skip every other node)
    for r in range(p['grid_h']):
        for c in range(p['grid_w'] - 2): # Note the -2
            tendons_xml += make_bend_tendon(f"S_{r}_{c}", f"S_{r}_{c+2}")

    # Vertical Bending
    for r in range(p['grid_h'] - 2):
        for c in range(p['grid_w']):
            tendons_xml += make_bend_tendon(f"S_{r}_{c}", f"S_{r+2}_{c}")

    xml_end = "</tendon>\n</mujoco>"
    
    return xml_header + bodies_xml + xml_footer_start + tendons_xml + xml_end