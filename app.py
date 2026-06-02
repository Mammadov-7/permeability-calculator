import streamlit as st
# from calculators.absolute_perm import calculate_absolute_permeability

st.set_page_config(page_title="Permeability Calculator", layout="wide")

st.title("🧮 Permeability Calculator")
st.markdown("Reservoir simulation tool for Absolute & Relative Permeability")

# Create two tabs for the interface
tab1, tab2 = st.tabs(["Absolute Permeability", "Relative Permeability"])

# Everything inside 'with tab1:' goes into the first tab
with tab1:
    st.header("Absolute Permeability (Darcy's Law)")
    
    # Split the screen into two columns
    col1, col2 = st.columns(2)
    
    with col1:
        Q = st.number_input("Flow Rate Q (cm³/s)", value=1.0)
        mu = st.number_input("Viscosity μ (cp)", value=1.0)
        L = st.number_input("Sample Length L (cm)", value=10.0)
        
    with col2:
        A = st.number_input("Cross-section Area A (cm²)", value=5.0)
        dP = st.number_input("Pressure Drop ΔP (atm)", value=1.0)

    # The Calculate Button
    if st.button("Calculate K"):
        # Placeholder for your actual python math logic
        # K = calculate_absolute_permeability(Q, mu, L, A, dP)
        K = (Q * mu * L) / (A * dP) 
        st.success(f"Absolute Permeability K = {K:.4f} mD")

with tab2:
    st.header("Relative Permeability")
    st.write("Relative permeability history matching module will be integrated here.")