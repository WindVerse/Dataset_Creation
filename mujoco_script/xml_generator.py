import config as cfg

def get_model_xml_explicit():
    p = cfg.PHYSICS_CONFIG
    
    # 1. Calculate grid spacing based on physical dimensions
    # flexcomp requires spacing in "x y z" format
    spacing_x = p['width_m'] / (p['grid_w'] - 1)
    spacing_y = p['height_m'] / (p['grid_h'] - 1)
    
    xml = f"""
    <mujoco model="flag_flex">
        <option timestep="{p['timestep']}" gravity="{p['gravity']}" density="{p['density']}" 
                viscosity="{p['viscosity']}" integrator="implicitfast" solver="CG" tolerance="1e-6"/>
        
        <visual>
            <map force="0.1" zfar="30"/>
            <headlight ambient="0.6 0.6 0.6" diffuse="0.4 0.4 0.4" specular="0.1 0.1 0.1"/>
        </visual>
        
        <extension>
            <plugin plugin="mujoco.elasticity.cable"/>
        </extension>

        <default>
            <geom type="sphere" size="{p['node_radius']}" rgba=".8 .2 .2 1"/>
        </default>

        <worldbody>
            <light pos="0 0 10"/>
            <geom name="floor" type="plane" size="10 10 .1" pos="0 0 {p['floor_z']}" rgba=".9 .9 .9 1"/>
            
            <body name="flag_root" pos="0 0 {p['start_z']}">
                <flexcomp name="cloth" type="grid" 
                          count="{p['grid_w']} {p['grid_h']} 1" 
                          spacing="{spacing_x} {spacing_y} 0.01" 
                          mass="{p['node_mass'] * p['grid_w'] * p['grid_h']}" 
                          radius="0.001" rgba=".8 .2 .2 1">
                    
                    <edge equality="true" damping="{p['damping']}"/>
                    <contact condim="3" solref="{p['solref']}" solimp=".95 .99 .0001" selfcollide="none"/>
                    
                    <plugin plugin="mujoco.elasticity.cable">
                        <config key="poisson" value="0.403"/>
                        <config key="thickness" value="0.0005"/>
                        <config key="young" value="85242.0"/> 
                    </plugin>
                </flexcomp>
            </body>
        </worldbody>
        
        <equality>
    """

    # 3. STATIC POLE LOGIC (Weld Constraints)
    # In flexcomp type="grid", bodies are generated with names "cloth_0", "cloth_1", etc.
    # The indices usually iterate X first, then Y.
    # We want to pin the first column (x=0) for every row.
    # Indices for first column: 0, W, 2W, 3W ...
    
    W = p['grid_w']
    H = p['grid_h']
    
    for r in range(H):
        # The index of the node on the left edge (Column 0) for this row
        node_idx = r * W
        
        # We weld this node to the world (no body2 means body2="world")
        # 'anchor' isn't strictly needed for a weld at current pos, but helps stability
        xml += f'        <weld name="pin_{r}" body1="cloth_{node_idx}" />\n'

    xml += """
        </equality>
    </mujoco>
    """
    
    return xml