import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="🥔 Potato Disease Detection",
    page_icon="🥔",
    layout="wide"
)

# ---------------- CSS ----------------

st.markdown("""
<style>

.stApp{
background:linear-gradient(
135deg,
#E0F2FE,
#DBEAFE,
#F8FAFC
);
}

/* Main heading */
.main-title{
font-size:58px;
font-weight:800;
text-align:center;
background:linear-gradient(
90deg,#2563EB,#16A34A
);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
margin-top:15px;
}

/* subtitle */
.sub{
font-size:20px;
text-align:center;
color:#475569;
margin-bottom:35px;
}

/* Glass card */

.card{
background:rgba(255,255,255,.75);
backdrop-filter:blur(12px);
padding:25px;
border-radius:30px;
box-shadow:
0 8px 30px rgba(0,0,0,.15);
}

/* Prediction box */

.result{
padding:30px;
border-radius:25px;
font-size:32px;
font-weight:bold;
text-align:center;
color:white;
box-shadow:
0 8px 20px rgba(0,0,0,.2);
}

/* uploader */

[data-testid="stFileUploader"]{
background:white;
padding:15px;
border-radius:20px;
border:2px dashed #60A5FA;
}

/* button */

.stButton>button{

width:100%;
height:58px;

font-size:20px;
font-weight:bold;

border:none;
border-radius:18px;

background:linear-gradient(
90deg,
#2563EB,
#0EA5E9
);

color:white;

box-shadow:
0 6px 15px rgba(
37,99,235,.4
);

}

.stButton>button:hover{

transform:scale(1.02);

transition:.3s;

}

/* sidebar */

section[data-testid="stSidebar"]{

background:white;

}

/* footer */

.footer{

text-align:center;
margin-top:60px;
color:#475569;

}

</style>
""",unsafe_allow_html=True)

# ---------------- MODEL ----------------

model=tf.keras.models.load_model(
"potato_model.h5"
)

classes=[
"Early Blight",
"Late Blight",
"Healthy"
]

THRESHOLD=80

# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.image(
    "https://cdn-icons-png.flaticon.com/512/2909/2909762.png",
    width=100
    )

    st.header("Project Details")

    st.write(
    "CNN based Deep Learning model for detecting potato leaf diseases."
    )

    st.markdown("---")

    st.write("✔ Early Blight")
    st.write("✔ Late Blight")
    st.write("✔ Healthy")
    st.write("✔ Unknown Detection")

# ---------------- TITLE ----------------

st.markdown(
'<div class="main-title">🥔 Potato Leaf Disease Detection</div>',
unsafe_allow_html=True
)

st.markdown(
'<div class="sub">AI Powered Smart Agriculture using CNN + Streamlit</div>',
unsafe_allow_html=True
)

# ---------------- COLUMNS ----------------

left,right=st.columns([1,1])

# ---------------- LEFT ----------------

with left:

    st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
    )

    st.subheader(
    "📤 Upload Potato Leaf"
    )

    uploaded=st.file_uploader(
    "",
    type=["jpg","png","jpeg"]
    )

    st.markdown(
    "</div>",
    unsafe_allow_html=True
    )

# ---------------- RIGHT ----------------

with right:

    if uploaded:

        image=Image.open(
        uploaded
        ).convert("RGB")

        st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
        )

        img=image.resize(
        (128,128)
        )

        img=np.array(
        img
        )

        img=img/255.0

        img=np.expand_dims(
        img,
        axis=0
        )

        if st.button(
        "🔍 Analyze Leaf"
        ):

            with st.spinner(
            "Analyzing..."
            ):

                prediction=model.predict(
                img
                )

            index=np.argmax(
            prediction
            )

            confidence=float(
            np.max(
            prediction
            )*100
            )

            if confidence < THRESHOLD:

                result="❌ Unknown / Not Potato"

                color="#DC2626"

            else:

                result=classes[index]

                color="#16A34A"

            st.markdown(
            f"""
            <div
            class="result"
            style="background:{color};">

            Prediction:<br><br>

            {result}

            </div>
            """,
            unsafe_allow_html=True
            )

            st.write(
            "Confidence Score"
            )

            st.progress(
            confidence/100
            )

            st.success(
            f"{confidence:.2f}% confidence"
            )

# ---------------- FOOTER ----------------

st.markdown(
"""
<div class='footer'>

CNN + TensorFlow + Streamlit  
Potato Leaf Disease Detection System

</div>
""",
unsafe_allow_html=True
)