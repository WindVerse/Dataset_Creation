import config as cfg

def get_model_xml_explicit():
    p = cfg.PHYSICS_CONFIG
    
    # 1. Spacing Calculation
    spacing_x = p['width_m'] / (p['grid_w'] - 1)
    spacing_y = p['height_m'] / (p['grid_h'] - 1)

    # 2. Position & Orientation Logic
    # ---------------------------------------------------------
    # GOAL: X range [0, 0.6], Z range [-0.2, 0.2]
    #
    # A. CENTER X:
    # Since the grid is width 0.6 centered at 0 (range -0.3 to 0.3),
    # we shift it right by half the width (0.3) to get range 0.0 to 0.6.
    center_x = p['width_m'] / 2.0  # = 0.3
    
    # B. CENTER Z:
    # We want Z range -0.2 to 0.2. The center of this range is 0.0.
    # (Config 'start_z' is 0.2, 'height' is 0.4.  0.2 - (0.4/2) = 0.0)
    center_z = p['start_z'] - (p['height_m'] / 2.0) # = 0.0
    
    # C. ROTATION:
    # The grid creates X (width) and Y (height). 
    # We rotate 90 degrees on X-axis so that Y points up/down (aligns with Z).
    # ---------------------------------------------------------

    xml = f"""
    <mujoco model="flag_flex">
        <compiler angle="degree"/>
        <option timestep="0.01" integrator="implicitfast" viscosity="{p['viscosity']}" solver="CG" tolerance="1e-6"/>
        
        <extension>
            <plugin plugin="mujoco.elasticity.shell"/>
        </extension>

        <worldbody>
            <light pos="0 0 10"/>
            <geom name="floor" type="plane" size="10 10 .1" pos="0 0 -1" rgba=".9 .9 .9 1"/>
            
            <body name="flag_root" pos="{center_x} 0 {center_z}" euler="90 0 0">
                
                <flexcomp type="grid" name="cloth"
                          count="{p['grid_w']} {p['grid_h']} 1" 
                          spacing="{spacing_x} {spacing_y} 0.01"
                          mass="{p['node_mass'] * p['grid_w'] * p['grid_h']}" 
                          radius="0.001" rgba="1 0 0 0.3">
                    
                    <edge equality="true" damping="{p['damping']}"/>
                    <contact condim="3" solref="{p['solref']}" solimp=".95 .99 .0001"/>
                    
                    <plugin plugin="mujoco.elasticity.shell">
                        <config key="poisson" value="{p['poisson']}"/>
                        <config key="thickness" value="{p['thickness']}"/>
                        <config key="young" value="{p['young']}"/> 
                    </plugin>
                </flexcomp>
            </body>
        </worldbody>
        
        <equality>
    """

    # 3. STATIC POLE LOGIC (Weld Constraints)
    # Even after rotation, the topology logic (indices) remains the same.
    # We pin Column 0 (the left edge).
    
    W = p['grid_w']
    H = p['grid_h']
    
    for r in range(H):
        # TRY THIS: Sequential indexing (0, 1, 2 ... H-1)
        # This assumes the nodes 0..H-1 represent the first vertical column (pole).
        node_idx = r
        
        # If the flag looks like it's pinned horizontally along the top instead,
        # then the grid is Row-Major and we need the old logic:
        # node_idx = r * W

        # Explicitly name the pin for easier debugging in the visualizer
        # xml += f'        <weld name="pin_{r}" body1="cloth_{node_idx}" />\n'
        xml += f'        <weld name="pin_{r}" body1="cloth_{node_idx}" solref="0.001 1" solimp="0.99 0.999 0.001"/>\n'

    xml += """
        </equality>
    </mujoco>
    """
    
    return xml