import streamlit as st
import io
from calculators.absolute_perm import match_permeability

st.set_page_config(page_title="Permeability Calculator", layout="wide")

st.title("🧮 Permeability Calculator")
st.markdown("Reservoir simulation tool for Automated History Matching")

tab1, tab2 = st.tabs(["Absolute Permeability", "Relative Permeability"])

with tab1:
    st.header("Automated History Matching (1D FD Core Flood)")
    st.write("Enter your lab parameters below. The Nelder-Mead algorithm will automatically find the exact permeability required to match your target differential pressure.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        L = st.number_input("Sample Length (cm)", value=6.0)
        A = st.number_input("Cross-section Area (cm²)", value=11.4)
        phi = st.number_input("Porosity (fraction)", value=0.20)
        
    with col2:
        mu = st.number_input("Viscosity (cP)", value=1.0)
        c_Pa = st.number_input("Compressibility (1/Pa)", value=4.5e-10, format="%.2e")
        q_inj = st.number_input("Injection Rate (ml/min)", value=1.0)
        
    with col3:
        P_out = st.number_input("Back-Pressure (bar)", value=10.0)
        target_dp = st.number_input("Target ΔP (mbar)", value=50.0)
        total_time = st.number_input("Simulation Time (mins)", value=60.0)

    # A hidden dt value for stability, but changeable if needed
    dt_mins = st.number_input("Timestep size (mins)", value=1.0)

    if st.button("Run Optimizer & Calculate K", type="primary"):
        with st.spinner('Running 1D Finite-Difference Simulation (Nelder-Mead is thinking)...'):
            
            # 1. Trigger your pure Python math engine
            optimised_k, fig = match_permeability(
                L_cm=L, A_cm2=A, phi=phi, mu_cP=mu, c_Pa=c_Pa, 
                q_ml_min=q_inj, P_out_bar=P_out, total_time_mins=total_time, 
                dt_mins=dt_mins, target_dp_mbar=target_dp
            )
            
            # 2. Display the final calculated number
            st.success(f"History Matching Complete! Optimised Permeability K = **{optimised_k:.4f} mD**")
            
            # 3. Display the Matplotlib graph directly on the website
            st.pyplot(fig)
            
            # 4. Claude's Addition: Create an in-memory buffer to allow downloading the graph
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            
            st.download_button(
                label="📥 Download Graph as PNG",
                data=buf,
                file_name=f"Delta_Pressure_Match_{optimised_k:.2f}mD.png",
                mime="image/png"
            )

with tab2:
    st.header("Relative Permeability")
    st.write("JutulDarcy gas-brine relative permeability module will be integrated here.")