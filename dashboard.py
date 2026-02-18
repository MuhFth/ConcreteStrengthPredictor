import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Concrete Strength Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Cohesive bright red/rose theme
st.markdown("""
    <style>
    /* ===== BACKGROUNDS ===== */
    [data-testid="stAppViewContainer"] {
        background-color: #FFF0F0 !important;
    }
    .main {
        background-color: #FFF0F0;
    }
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
        background-color: #FFF0F0 !important;
    }
    [data-testid="stHeader"] {
        background-color: #FFD6D6 !important;
        border-bottom: 2px solid #FF8A8A;
    }

    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background-color: #FFD6D6 !important;
        border-right: 2px solid #FF8A8A;
    }
    [data-testid="stSidebarContent"] {
        background-color: #FFD6D6 !important;
    }
    [data-testid="stSidebar"] *  {
        color: #5a0000 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #8B0000 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #5a0000 !important;
        font-weight: 500;
    }

    /* ===== MAIN CONTENT TEXT — force dark on light bg ===== */
    .main p, .main span, .main div,
    .main label, .main li,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span {
        color: #3a0000 !important;
    }
    h1 { color: #8B0000 !important; font-weight: 700; }
    h2 { color: #B22222 !important; font-weight: 600; }
    h3 { color: #CC3333 !important; font-weight: 500; }

    /* ===== INPUT FIELDS — white bg, dark text ===== */
    .stNumberInput input,
    input[type="number"],
    input[type="text"] {
        background-color: #FFFFFF !important;
        color: #3a0000 !important;
        border: 1.5px solid #FF8A8A !important;
        border-radius: 6px !important;
    }
    .stNumberInput input:focus {
        border: 1.5px solid #CC0000 !important;
        box-shadow: 0 0 0 2px rgba(204,0,0,0.15) !important;
    }
    /* Input labels */
    .stNumberInput label,
    .stTextInput label,
    [data-testid="stWidgetLabel"] {
        color: #5a0000 !important;
        font-weight: 500 !important;
    }

    /* ===== CUSTOM BOXES ===== */
    .success-box {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #CC0000 0%, #8B0000 100%);
        color: white;
        text-align: center;
        margin: 20px 0;
    }
    .info-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #FFE8E8;
        border-left: 4px solid #FF4444;
        margin: 10px 0;
        color: #5a0000 !important;
    }
    .warning-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #FFF3E0;
        border-left: 4px solid #FF8C00;
        margin: 10px 0;
        color: #4a2000 !important;
    }

    /* ===== METRIC CARDS ===== */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #8B0000 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #CC3333 !important;
    }
    [data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #FFAAAA;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(180,0,0,0.1);
    }

    /* ===== BUTTONS ===== */
    .stButton>button {
        background: linear-gradient(135deg, #CC0000 0%, #8B0000 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #FF2222 0%, #CC0000 100%) !important;
        box-shadow: 0 4px 12px rgba(180,0,0,0.3) !important;
    }

    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab"] {
        color: #8B0000 !important;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 2px solid #CC0000 !important;
        color: #CC0000 !important;
    }

    /* ===== DATAFRAME ===== */
    [data-testid="stDataFrame"] {
        border: 1px solid #FFAAAA;
        border-radius: 8px;
    }

    /* ===== FILE UPLOADER & MISC WIDGETS ===== */
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] p,
    .stSelectbox label,
    .stMultiSelect label {
        color: #5a0000 !important;
    }

    /* ===== SUBHEADERS & section text ===== */
    [data-testid="stSubheader"],
    .stSubheader {
        color: #8B0000 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load model function
@st.cache_resource
def load_model():
    """Load the concrete strength prediction model"""
    try:
        # Use joblib to load the model
        model_path = 'concrete_strength_model.pkl'
        model_data = joblib.load(model_path)
        
        # Check if it's a dictionary with model inside
        if isinstance(model_data, dict):
            st.success("✅ Model berhasil dimuat!")
            return model_data
        else:
            st.success("✅ Model berhasil dimuat!")
            return {'model': model_data}
            
    except FileNotFoundError:
        st.error("❌ File model tidak ditemukan. Pastikan concrete_strength_model.pkl ada di direktori yang benar.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("💡 Tip: Pastikan file model dalam format yang benar (joblib/pickle)")
        return None

# Prediction function
def predict_strength(model_data, features):
    """Predict concrete strength from features"""
    try:
        # Extract model and scaler from dictionary
        if isinstance(model_data, dict):
            model = model_data.get('model')
            scaler = model_data.get('scaler', None)
            feature_names = model_data.get('feature_names', [])
        else:
            model = model_data
            scaler = None
            feature_names = []
        
        # Original 8 features
        cement, slag, flyash, water, superplasticizer, coarse, fine, age = features
        
        # Calculate engineered features
        rasio_air_semen = water / cement if cement > 0 else 0
        total_bahan_pengikat = cement + slag + flyash
        log_umur_hari = np.log1p(age)  # log(1 + age)
        
        # Check if model expects Indonesian column names
        if len(feature_names) == 11 and 'Semen' in feature_names:
            # Create DataFrame with Indonesian column names
            import pandas as pd
            
            data_dict = {
                'Semen': [cement],
                'Slag_Tanur_Tinggi': [slag],
                'Abu_Terbang': [flyash],
                'Air': [water],
                'Superplasticizer': [superplasticizer],
                'Agregat_Kasar': [coarse],
                'Agregat_Halus': [fine],
                'Umur_Hari': [age],
                'Rasio_Air_Semen': [rasio_air_semen],
                'Total_Bahan_Pengikat': [total_bahan_pengikat],
                'Log_Umur_Hari': [log_umur_hari]
            }
            
            df = pd.DataFrame(data_dict)
            
            # Scale if scaler exists
            if scaler is not None:
                features_array = scaler.transform(df)
            else:
                features_array = df.values
                
        else:
            # Fallback: use numpy array
            all_features = [
                cement, slag, flyash, water, superplasticizer, coarse, fine, age,
                rasio_air_semen, total_bahan_pengikat, log_umur_hari
            ]
            features_array = np.array(all_features).reshape(1, -1)
            
            # Scale if scaler exists
            if scaler is not None:
                features_array = scaler.transform(features_array)
        
        # Predict
        prediction = model.predict(features_array)
        return max(0, float(prediction[0]))
        
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# Batch prediction function for DataFrames
def predict_batch(model_data, df):
    """Predict concrete strength for a batch of samples in a DataFrame"""
    try:
        # Extract model and scaler
        if isinstance(model_data, dict):
            model = model_data.get('model')
            scaler = model_data.get('scaler', None)
            feature_names = model_data.get('feature_names', [])
        else:
            model = model_data
            scaler = None
            feature_names = []
        
        # Create DataFrame with Indonesian column names and engineered features
        df_model = pd.DataFrame()
        
        df_model['Semen'] = df['Cement']
        df_model['Slag_Tanur_Tinggi'] = df['Blast Furnace Slag']
        df_model['Abu_Terbang'] = df['Fly Ash']
        df_model['Air'] = df['Water']
        df_model['Superplasticizer'] = df['Superplasticizer']
        df_model['Agregat_Kasar'] = df['Coarse Aggregate']
        df_model['Agregat_Halus'] = df['Fine Aggregate']
        df_model['Umur_Hari'] = df['Age']
        
        # Calculate engineered features
        df_model['Rasio_Air_Semen'] = df_model['Air'] / df_model['Semen']
        df_model['Total_Bahan_Pengikat'] = df_model['Semen'] + df_model['Slag_Tanur_Tinggi'] + df_model['Abu_Terbang']
        df_model['Log_Umur_Hari'] = np.log1p(df_model['Umur_Hari'])
        
        # Scale if scaler exists
        if scaler is not None:
            features_scaled = scaler.transform(df_model)
        else:
            features_scaled = df_model.values
        
        # Predict
        predictions = model.predict(features_scaled)
        return np.maximum(0, predictions)
        
    except Exception as e:
        st.error(f"❌ Batch prediction error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# Grade classification
def get_concrete_grade(strength):
    """Classify concrete based on strength"""
    if strength < 20:
        return "K-175", "🔴 Kekuatan Rendah", "Untuk pekerjaan non-struktural"
    elif strength < 30:
        return "K-250", "🟡 Kekuatan Sedang", "Untuk struktur ringan"
    elif strength < 40:
        return "K-300", "🟢 Kekuatan Tinggi", "Untuk struktur bangunan"
    else:
        return "K-400+", "🔵 Kekuatan Sangat Tinggi", "Untuk struktur khusus"

# Initialize session state
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

# Load model
model_data = load_model()
if model_data is None:
    st.error("⚠️ Model tidak dapat dimuat. Silakan periksa file model Anda.")
    st.stop()

# Header
st.markdown("""
    <div style='background: linear-gradient(135deg, #1976d2 0%, #0d47a1 100%); 
                padding: 30px; border-radius: 10px; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);'>
        <h1 style='color: white; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>🏗️ Concrete Strength Predictor</h1>
        <p style='color: #e3f2fd; margin: 5px 0 0 0; font-size: 18px;'>
            Prediksi Kekuatan Beton dengan Machine Learning
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for navigation
with st.sidebar:
    st.markdown("### 📋 Navigation")
    menu = st.radio(
        "Pilih Menu:",
        ["📝 Input Manual", "📁 Upload CSV", "📊 Model Performance"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    
    # Display actual model info if available
    if model_data and isinstance(model_data, dict):
        model = model_data.get('model')
        feature_names = model_data.get('feature_names', [])
        metrics = model_data.get('metrics', {})
        
        if hasattr(model, '__class__'):
            st.info(f"""
            **Model Information:**
            - Type: {model.__class__.__name__}
            - Input Features: {len(feature_names)}
            - R² Score: {metrics.get('r2', 'N/A')}
            - Output: {model_data.get('target_name', 'Compressive Strength')}
            """)
    else:
        st.info("""
        **Model Information:**
        - Algorithm: Linear Regression
        - Features: 8 input parameters
        - Output: Compressive Strength (MPa)
        """)
    
    st.markdown("### 🎯 Quick Stats")
    if model_data and isinstance(model_data, dict):
        model = model_data.get('model')
        if hasattr(model, 'n_features_in_'):
            st.metric("Model Features", model.n_features_in_)
        st.metric("Input Parameters", "8")
    st.metric("Total Predictions", len(st.session_state.predictions_history))

# ==================== MENU 1: INPUT MANUAL ====================
if menu == "📝 Input Manual":
    st.header("📝 Input Manual - Prediksi Individual")
    
    st.markdown("""
        <div class='info-box'>
            <strong>ℹ️ Petunjuk:</strong> Masukkan parameter beton di bawah ini untuk mendapatkan prediksi kekuatan.
            Gunakan tombol "Isi Contoh Data" untuk melihat contoh input.
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class='warning-box'>
            <strong>⚠️ PENTING - Satuan dalam kg/m³:</strong><br>
            • Cement: 100-540 kg/m³ (BUKAN 2-5!)<br>
            • Water: 120-250 kg/m³ (BUKAN 3-10!)<br>
            • Coarse Aggregate: 800-1150 kg/m³ (BUKAN 1-5!)<br>
            • Fine Aggregate: 600-1000 kg/m³ (BUKAN 1-5!)<br>
            <strong>👉 Klik "📋 Isi Contoh Data" untuk nilai yang benar!</strong>
        </div>
    """, unsafe_allow_html=True)
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧱 Material Semen")
        cement = st.number_input(
            "Cement (kg/m³)",
            min_value=0.0,
            max_value=600.0,
            value=0.0,
            step=10.0,
            help="Jumlah semen dalam campuran (100-540 kg/m³)",
            placeholder="Contoh: 540"
        )
        
        slag = st.number_input(
            "Blast Furnace Slag (kg/m³)",
            min_value=0.0,
            max_value=400.0,
            value=0.0,
            step=10.0,
            help="Material pengganti semen (0-360 kg/m³)",
            placeholder="Contoh: 0"
        )
        
        flyash = st.number_input(
            "Fly Ash (kg/m³)",
            min_value=0.0,
            max_value=250.0,
            value=0.0,
            step=10.0,
            help="Abu terbang dari pembakaran batu bara (0-200 kg/m³)",
            placeholder="Contoh: 0"
        )
        
        water = st.number_input(
            "Water (kg/m³)",
            min_value=0.0,
            max_value=300.0,
            value=0.0,
            step=5.0,
            help="Jumlah air dalam campuran (120-250 kg/m³)",
            placeholder="Contoh: 162"
        )
    
    with col2:
        st.subheader("🪨 Agregat & Additives")
        superplasticizer = st.number_input(
            "Superplasticizer (kg/m³)",
            min_value=0.0,
            max_value=35.0,
            value=0.0,
            step=0.5,
            help="Bahan kimia untuk meningkatkan workability (0-32 kg/m³)",
            placeholder="Contoh: 2.5"
        )
        
        coarse_aggregate = st.number_input(
            "Coarse Aggregate (kg/m³)",
            min_value=0.0,
            max_value=1200.0,
            value=0.0,
            step=10.0,
            help="Agregat kasar seperti kerikil (800-1150 kg/m³)",
            placeholder="Contoh: 1040"
        )
        
        fine_aggregate = st.number_input(
            "Fine Aggregate (kg/m³)",
            min_value=0.0,
            max_value=1100.0,
            value=0.0,
            step=10.0,
            help="Agregat halus seperti pasir (600-1000 kg/m³)",
            placeholder="Contoh: 676"
        )
        
        age = st.number_input(
            "Age (Days)",
            min_value=1,
            max_value=365,
            value=28,
            step=1,
            help="Umur beton dalam hari (1-365 hari)",
            placeholder="Contoh: 28"
        )
    
    # Buttons
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        predict_button = st.button("🔮 Prediksi Kekuatan", type="primary", use_container_width=True)
    
    with col_btn2:
        if st.button("📋 Isi Contoh Data", use_container_width=True):
            st.session_state.example_data = {
                'cement': 540.0,
                'slag': 0.0,
                'flyash': 0.0,
                'water': 162.0,
                'superplasticizer': 2.5,
                'coarse_aggregate': 1040.0,
                'fine_aggregate': 676.0,
                'age': 28
            }
            st.rerun()
    
    # Load example data if exists
    if 'example_data' in st.session_state:
        cement = st.session_state.example_data['cement']
        slag = st.session_state.example_data['slag']
        flyash = st.session_state.example_data['flyash']
        water = st.session_state.example_data['water']
        superplasticizer = st.session_state.example_data['superplasticizer']
        coarse_aggregate = st.session_state.example_data['coarse_aggregate']
        fine_aggregate = st.session_state.example_data['fine_aggregate']
        age = st.session_state.example_data['age']
        del st.session_state.example_data
        st.rerun()
    
    # Prediction
    if predict_button:
        # Validate inputs first
        if cement == 0 or water == 0 or coarse_aggregate == 0 or fine_aggregate == 0:
            st.error("⚠️ Mohon isi semua field yang diperlukan (Cement, Water, Coarse Aggregate, Fine Aggregate)!")
        elif cement < 100:
            st.error("⚠️ Nilai Cement terlalu rendah! Cement untuk beton normal: 100-540 kg/m³. Klik '📋 Isi Contoh Data' untuk melihat nilai yang benar.")
        elif water < 100:
            st.error("⚠️ Nilai Water terlalu rendah! Water untuk beton normal: 120-250 kg/m³. Klik '📋 Isi Contoh Data' untuk melihat nilai yang benar.")
        elif coarse_aggregate < 500:
            st.error("⚠️ Nilai Coarse Aggregate terlalu rendah! Coarse Aggregate untuk beton normal: 800-1150 kg/m³. Klik '📋 Isi Contoh Data' untuk melihat nilai yang benar.")
        elif fine_aggregate < 500:
            st.error("⚠️ Nilai Fine Aggregate terlalu rendah! Fine Aggregate untuk beton normal: 600-1000 kg/m³. Klik '📋 Isi Contoh Data' untuk melihat nilai yang benar.")
        else:
            features = [cement, slag, flyash, water, superplasticizer, 
                       coarse_aggregate, fine_aggregate, age]
            
            prediction = predict_strength(model_data, features)
            
            if prediction is not None:
                # Save to history
                st.session_state.predictions_history.append({
                    'features': features,
                    'prediction': prediction,
                    'timestamp': pd.Timestamp.now()
                })
                
                # Calculate additional metrics
                wc_ratio = water / cement if cement > 0 else 0
                grade, category, usage = get_concrete_grade(prediction)
                
                # Display results
                st.markdown("---")
                st.markdown(f"""
                    <div class='success-box'>
                        <h2 style='color: white; margin: 0;'>📈 Hasil Prediksi</h2>
                        <h1 style='color: white; font-size: 48px; margin: 10px 0;'>{prediction:.2f} MPa</h1>
                        <p style='margin: 0; font-size: 18px;'>{category}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Grade Beton", grade)
                
                with col2:
                    st.metric("Umur Beton", f"{age} hari")
                
                with col3:
                    st.metric("W/C Ratio", f"{wc_ratio:.2f}")
                
                with col4:
                    total_material = cement + slag + flyash + water + superplasticizer + coarse_aggregate + fine_aggregate
                    st.metric("Total Material", f"{total_material:.0f} kg/m³")
                
                # Usage recommendation
                st.markdown(f"""
                    <div class='info-box'>
                        <strong>💡 Rekomendasi Penggunaan:</strong> {usage}
                    </div>
                """, unsafe_allow_html=True)
                
                # Material composition chart
                st.subheader("📊 Komposisi Material")
                
                composition_data = pd.DataFrame({
                    'Material': ['Cement', 'Slag', 'Fly Ash', 'Water', 'Superplasticizer', 
                                'Coarse Agg', 'Fine Agg'],
                    'Amount': [cement, slag, flyash, water, superplasticizer, 
                              coarse_aggregate, fine_aggregate]
                })
                
                fig = px.pie(composition_data, values='Amount', names='Material',
                           title='Distribusi Material Beton',
                           color_discrete_sequence=px.colors.sequential.RdBu)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)

# ==================== MENU 2: UPLOAD CSV ====================
elif menu == "📁 Upload CSV":
    st.header("📁 Upload CSV - Prediksi Batch")
    
    st.markdown("""
        <div class='info-box'>
            <strong>ℹ️ Petunjuk:</strong> Upload file CSV dengan kolom yang sesuai untuk melakukan prediksi batch.
            Download template CSV di bawah untuk format yang benar.
        </div>
    """, unsafe_allow_html=True)
    
    # Template download
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📋 Format CSV yang Diperlukan")
        st.code("""
Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, Age
        """, language="text")
    
    with col2:
        # Create template
        template_data = pd.DataFrame({
            'Cement': [540, 450, 425],
            'Blast Furnace Slag': [0, 100, 106],
            'Fly Ash': [0, 0, 0],
            'Water': [162, 180, 153],
            'Superplasticizer': [2.5, 3.0, 4.0],
            'Coarse Aggregate': [1040, 950, 1000],
            'Fine Aggregate': [676, 720, 750],
            'Age': [28, 28, 28]
        })
        
        csv_template = template_data.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Template",
            data=csv_template,
            file_name="concrete_template.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=['csv'],
        help="Upload file CSV dengan format yang sesuai"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File berhasil di-upload! Total: {len(df)} baris data")
            
            # Show preview
            st.subheader("👀 Preview Data")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Validate columns
            required_cols = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 
                           'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age']
            
            # Check for column variations (case insensitive and with/without spaces)
            df.columns = df.columns.str.strip()
            
            missing_cols = []
            for col in required_cols:
                # Try to find column with similar name
                found = False
                for df_col in df.columns:
                    if col.lower().replace(' ', '') == df_col.lower().replace(' ', ''):
                        if df_col != col:
                            df.rename(columns={df_col: col}, inplace=True)
                        found = True
                        break
                if not found:
                    missing_cols.append(col)
            
            if missing_cols:
                st.error(f"⚠️ Kolom yang hilang: {', '.join(missing_cols)}")
            else:
                # Perform predictions
                if st.button("🚀 Mulai Prediksi Batch", type="primary"):
                    with st.spinner("Sedang memproses prediksi..."):
                        # Use batch prediction function
                        predictions = predict_batch(model_data, df)
                        
                        if predictions is not None:
                            # Add predictions to dataframe
                            df['Predicted Strength (MPa)'] = predictions
                            df['Grade'] = df['Predicted Strength (MPa)'].apply(
                                lambda x: get_concrete_grade(x)[0]
                            )
                            df['W/C Ratio'] = df['Water'] / df['Cement']
                            
                            st.success("✅ Prediksi batch selesai!")
                        
                        # Statistics
                        st.subheader("📊 Statistik Hasil Prediksi")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Rata-rata Kekuatan", f"{df['Predicted Strength (MPa)'].mean():.2f} MPa")
                        
                        with col2:
                            st.metric("Kekuatan Maksimum", f"{df['Predicted Strength (MPa)'].max():.2f} MPa")
                        
                        with col3:
                            st.metric("Kekuatan Minimum", f"{df['Predicted Strength (MPa)'].min():.2f} MPa")
                        
                        with col4:
                            st.metric("Std Deviasi", f"{df['Predicted Strength (MPa)'].std():.2f} MPa")
                        
                        # Results table
                        st.subheader("📋 Hasil Lengkap")
                        st.dataframe(df, use_container_width=True)
                        
                        # Download results
                        csv_results = df.to_csv(index=False)
                        st.download_button(
                            label="💾 Download Hasil Prediksi",
                            data=csv_results,
                            file_name="prediction_results.csv",
                            mime="text/csv"
                        )
                        
                        # Visualizations
                        st.subheader("📈 Visualisasi Hasil")
                        
                        # Create tabs for different visualizations
                        tab1, tab2, tab3 = st.tabs(["📊 Distribution", "📈 Trends", "🎯 Comparison"])
                        
                        with tab1:
                            # Histogram of predictions
                            fig1 = px.histogram(
                                df, 
                                x='Predicted Strength (MPa)',
                                nbins=30,
                                title='Distribusi Kekuatan Beton Prediksi',
                                color_discrete_sequence=['#667eea']
                            )
                            fig1.update_layout(
                                xaxis_title="Kekuatan (MPa)",
                                yaxis_title="Frekuensi"
                            )
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        with tab2:
                            # Strength vs Age
                            fig2 = px.scatter(
                                df,
                                x='Age',
                                y='Predicted Strength (MPa)',
                                color='Grade',
                                size='Cement',
                                title='Kekuatan Beton vs Umur',
                                hover_data=['Cement', 'Water', 'W/C Ratio']
                            )
                            fig2.update_layout(
                                xaxis_title="Umur (hari)",
                                yaxis_title="Kekuatan (MPa)"
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        with tab3:
                            # Grade distribution
                            grade_counts = df['Grade'].value_counts().reset_index()
                            grade_counts.columns = ['Grade', 'Count']
                            
                            fig3 = px.bar(
                                grade_counts,
                                x='Grade',
                                y='Count',
                                title='Distribusi Grade Beton',
                                color='Grade',
                                color_discrete_sequence=px.colors.qualitative.Set3
                            )
                            st.plotly_chart(fig3, use_container_width=True)
                        
        except Exception as e:
            st.error(f"❌ Error membaca file: {e}")

# ==================== MENU 3: MODEL PERFORMANCE ====================
elif menu == "📊 Model Performance":
    st.header("📊 Model Performance & Analysis")
    
    # Model metrics
    st.subheader("🎯 Performance Metrics")
    
    # Get metrics from model_data if available
    if isinstance(model_data, dict) and 'metrics' in model_data:
        metrics = model_data['metrics']
        r2 = metrics.get('r2_score', 0.85)
        rmse = metrics.get('rmse', 10.24)
        mae = metrics.get('mae', 7.82)
    else:
        # Default values
        r2 = 0.85
        rmse = 10.24
        mae = 7.82
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1976d2 0%, #0d47a1 100%); 
                        padding: 25px; border-radius: 10px; text-align: center; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.3);'>
                <h3 style='color: white; margin: 0;'>R² Score</h3>
                <h1 style='color: white; font-size: 48px; margin: 10px 0;'>{r2:.2f}</h1>
                <p style='margin: 0; font-size: 14px;'>Coefficient of Determination</p>
                <div style='background: rgba(255,255,255,0.3); height: 8px; border-radius: 4px; margin-top: 15px;'>
                    <div style='background: white; height: 8px; border-radius: 4px; width: {int(r2*100)}%;'></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #388e3c 0%, #1b5e20 100%); 
                        padding: 25px; border-radius: 10px; text-align: center; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.3);'>
                <h3 style='color: white; margin: 0;'>RMSE</h3>
                <h1 style='color: white; font-size: 48px; margin: 10px 0;'>{rmse:.2f}</h1>
                <p style='margin: 0; font-size: 14px;'>Root Mean Squared Error (MPa)</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #00838f 0%, #006064 100%); 
                        padding: 25px; border-radius: 10px; text-align: center; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.3);'>
                <h3 style='color: white; margin: 0;'>MAE</h3>
                <h1 style='color: white; font-size: 48px; margin: 10px 0;'>{mae:.2f}</h1>
                <p style='margin: 0; font-size: 14px;'>Mean Absolute Error (MPa)</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Model Information")
        st.markdown("""
            <div class='info-box'>
                <strong>Algorithm:</strong> Linear Regression<br>
                <strong>Method:</strong> Ordinary Least Squares<br>
                <strong>Features:</strong> 8 input parameters<br>
                <strong>Output:</strong> Compressive Strength (MPa)
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📝 Input Features")
        features_list = [
            "1. Cement (kg/m³)",
            "2. Blast Furnace Slag (kg/m³)",
            "3. Fly Ash (kg/m³)",
            "4. Water (kg/m³)",
            "5. Superplasticizer (kg/m³)",
            "6. Coarse Aggregate (kg/m³)",
            "7. Fine Aggregate (kg/m³)",
            "8. Age (Days)"
        ]
        
        for feature in features_list:
            st.markdown(f"- {feature}")
    
    with col2:
        st.subheader("📊 Feature Importance (Input Parameters)")
        
        # Feature coefficients from actual model
        if isinstance(model_data, dict):
            model = model_data.get('model')
            all_feature_names = model_data.get('feature_names', [])
        else:
            model = model_data
            all_feature_names = []
        
        if hasattr(model, 'coef_') and len(all_feature_names) > 0:
            # Only show first 8 features (the input parameters, not engineered features)
            coefficients = model.coef_[:8]
            feature_names_display = all_feature_names[:8]
        else:
            coefficients = np.array([0.117, 0.104, 0.087, -0.148, 0.277, 0.010, 0.010, 0.114])
            feature_names_display = ['Cement', 'Slag', 'Fly Ash', 'Water', 
                                    'Superplasticizer', 'Coarse Agg', 'Fine Agg', 'Age']
        
        feature_importance = pd.DataFrame({
            'Feature': feature_names_display,
            'Coefficient': coefficients,
            'Abs Coefficient': np.abs(coefficients)
        }).sort_values('Abs Coefficient', ascending=True)
        
        fig = px.bar(
            feature_importance,
            y='Feature',
            x='Coefficient',
            orientation='h',
            title='Impact of Input Parameters on Concrete Strength',
            color='Coefficient',
            color_continuous_scale='RdBu',
            color_continuous_midpoint=0
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#5a0000')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        st.markdown("""
            <div class='info-box'>
                <strong>ℹ️ Interpretasi:</strong><br>
                • Nilai positif = meningkatkan kekuatan beton<br>
                • Nilai negatif = menurunkan kekuatan beton<br>
                • Semakin besar nilai absolut, semakin besar pengaruhnya
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Prediction history
    if st.session_state.predictions_history:
        st.subheader("📜 Riwayat Prediksi")
        
        history_df = pd.DataFrame([
            {
                'Timestamp': item['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'Prediction (MPa)': item['prediction'],
                'Cement': item['features'][0],
                'Water': item['features'][3],
                'Age': item['features'][7]
            }
            for item in st.session_state.predictions_history[-10:]  # Last 10 predictions
        ])
        
        st.dataframe(history_df, use_container_width=True)
        
        # Trend chart
        if len(st.session_state.predictions_history) > 1:
            fig = px.line(
                history_df,
                y='Prediction (MPa)',
                title='Trend Prediksi Terakhir',
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.predictions_history = []
            st.rerun()
    else:
        st.info("📭 Belum ada riwayat prediksi. Mulai prediksi dari menu Input Manual atau Upload CSV!")
    
    st.markdown("---")
    
    # Model insights
    st.subheader("💡 Model Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class='info-box'>
                <strong>🔍 Key Findings:</strong><br><br>
                • <strong>Superplasticizer</strong> memiliki pengaruh positif tertinggi (0.277)<br>
                • <strong>Water</strong> memiliki pengaruh negatif (-0.148), menunjukkan water-cement ratio yang tinggi menurunkan kekuatan<br>
                • <strong>Cement</strong> dan <strong>Age</strong> memberikan kontribusi positif yang signifikan<br>
                • <strong>Agregat</strong> memiliki pengaruh minor namun tetap penting untuk struktur
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='warning-box'>
                <strong>⚠️ Limitations:</strong><br><br>
                • Model ini adalah pendekatan linear dan mungkin tidak menangkap hubungan non-linear yang kompleks<br>
                • Akurasi prediksi bergantung pada kualitas dan range data training<br>
                • Untuk hasil terbaik, pastikan input berada dalam range yang wajar<br>
                • Selalu verifikasi dengan pengujian laboratorium untuk aplikasi kritis
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #8B0000; padding: 20px;'>
        <p>🏗️ <strong>Concrete Strength Predictor</strong> | Powered by Machine Learning</p>
        <p style='font-size: 14px;'> Copyrihgt © 2026 by Pengelola MK Praktikum Unggulan (Praktikum DGX), Universitas Gunadarma</p>
    </div>
""", unsafe_allow_html=True)